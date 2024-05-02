import random
import os
import wandb
import json
import numpy as np
import torch
import logging
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from typing import List, Tuple, Dict, Any, Literal
import torchvision.transforms.functional as tvF
import tiffslide as ts
from pathlib import Path
import itertools
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, f1_score


from mmm.utils import convert_tile_into_patchbatch, flatten_list_of_dicts
from mmm.mtl_modules.tasks.ClassificationTask import ClassificationTask
from mmm.mtl_modules.shared_blocks.PyramidEncoder import PyramidEncoder
from mmm.interactive import blocks, configs as cfs, data, pipes, tasks, training
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules


class SemiColConfig(cfs.BaseModel):
    write: bool = False
    alternative_path: str = "/output/tissueconcepts/nics/only-nics"
    level: int = 0
    patch_size: int = 224


class TupacConfig(cfs.BaseModel):
    write: bool = False
    alternative_path: str = "/output/tissueconcepts/nics/only-nics"
    level: int = 1
    patch_size: int = 224


class NICConfig(cfs.BaseModel):
    semicol_config: SemiColConfig = SemiColConfig()
    tupac_config: TupacConfig = TupacConfig()
    head_config: blocks.PyramidEncoder.Config = blocks.PyramidEncoder.Config(
        model=cfs.MiniConvNetConfig(num_channels=768),
        # Because of varying input sizes the batchsize is 1 and the accumulation is large.
        # High gradient accumulation means batchnorm is a bad choice
        norm_layer="affinelayernorm",
        # activation_fn=cfs.ActivationFunctionConfig(fn_type=cfs.ActivationFn.ReLU),
        # hidden_dim=64,
        module_name="encoder",
    )


class NICTask(ClassificationTask):
    """
    A classification task that processes neural image representations [C, H, W].
    """

    class Config(ClassificationTask.Config):
        pass

    def _visualize_preds(self, training_ims, step_metrics: Dict, metas: List[Dict]) -> Dict[str, Any]:
        vis_n = min(self._takeout_vis_budget(), training_ims.size(0))

        if vis_n <= 0:
            return {}

        nic = training_ims[0]
        meta = metas[0]
        metastr = json.dumps(meta, default=lambda o: str(o))

        logging.info(nic.shape)
        fmap_index = random.randint(0, nic.shape[0] - 1)

        logging.info(metas)

        return {"fmap": wandb.Image(nic[fmap_index], caption=metastr)}

    def training_step(self, batch: Dict[str, Any], shared_blocks: SharedModules):
        x = batch["image"]
        y = batch["class"]
        # skip if batch is empty
        tensornums = [bool(t.numel()) for t in x]
        if not True in tensornums:
            logging.info("encountered batch without valid training examples, skipping batch")
            return None

        y_hat = shared_blocks.forward(x, self.forward)

        # dealing last batch of a Cachingsubcase Dataset
        assert y.shape[0] == y_hat.shape[0], f"Till found something, {y.shape=}, {y_hat.shape=}"
        # if not y.shape[0] == y_hat.shape[0]:
        #     y = y.long()
        #     y_hat = y_hat.unsqueeze(0)

        batch_loss = self.criterion(y_hat, y) / np.log(len(self.class_names))
        split = np.asarray([b["split"] for b in batch["meta"]])
        step_results: StepMetricDict = {  # type: ignore (.numpy() does not correctly indicate numpy array)
            "targets": y.cpu().numpy(),
            "logits": y_hat.detach().cpu().numpy(),
            "preds": torch.argmax(y_hat.detach().cpu(), dim=1).numpy(),
            "lab": split,
        }
        self.add_step_result(batch_loss.item(), step_results)

        live_vis = self._visualize_preds(
            x.detach().cpu(),
            step_results,
            (batch["meta"] if "meta" in batch else [{} for _ in range(batch["image"].shape[0])]),
        )

        return batch_loss, live_vis

    def log_epoch_metrics(self) -> Tuple[Dict[str, Any], str]:
        log_dict, print_str = super().log_epoch_metrics()
        metrics = flatten_list_of_dicts(self._step_metrics)

        # Change such that we log the metrics overall and per split
        labs = np.unique(metrics["lab"])
        for lab in labs:
            log_dict[f"{lab}_acc"] = accuracy_score(
                y_true=metrics["targets"][np.where(metrics["lab"] == lab)],
                y_pred=metrics["preds"][np.where(metrics["lab"] == lab)],
            )
            log_dict[f"{lab}_f1"] = f1_score(
                y_true=metrics["targets"][np.where(metrics["lab"] == lab)],
                y_pred=metrics["preds"][np.where(metrics["lab"] == lab)],
                average="macro",
            )
            log_dict[f"{lab}_confmat"] = wandb.plot.confusion_matrix(
                preds=metrics["preds"][np.where(metrics["lab"] == lab)],  # type: ignore
                y_true=metrics["targets"][np.where(metrics["lab"] == lab)],  # type: ignore
                class_names=self._get_short_class_names(),
                title=f"{self._prefix}_{self.get_name()}_{lab}",
            )

        return log_dict, print_str


def get_next_divisable(n: int, divisor: int, updown: Literal["larger", "smaller"] = "larger") -> int:
    if updown == "larger":
        return n + (divisor - n % divisor) % divisor
    elif updown == "smaller":
        return n - n % divisor


def get_tile_edge_length(pixels_per_chunk: int = 96 * 224 * 224, patch_edge_length: int = 224) -> int:
    max_edge_length = np.sqrt(pixels_per_chunk)
    divisable_edge_length = int(max_edge_length - (max_edge_length % patch_edge_length))
    return divisable_edge_length


def compute_positions(dimsizes, stepsize):
    return itertools.product(*[enumerate(range(0, dimsize, stepsize)) for dimsize in dimsizes])


def read_regions(
    slide: ts.TiffSlide,
    positions: list,
    tile_edgelength: int,
    patch_edge_length: int,
    level=0,
):
    for (row, x), (col, y) in positions:
        dim1_edgelength = min(
            tile_edgelength,
            get_next_divisable(slide.level_dimensions[level][0] - x, patch_edge_length, "larger"),
        )
        dim2_edgelength = min(
            tile_edgelength,
            get_next_divisable(slide.level_dimensions[level][1] - y, patch_edge_length, "larger"),
        )
        location_downsample_fac = slide.level_downsamples[level]

        logging.debug(
            f"Region {row=}, {col=}, {x=}, {y=}, {dim1_edgelength=}, {dim2_edgelength=}, {level=}, {location_downsample_fac=}"
        )
        region = tvF.to_tensor(
            slide.read_region(
                location=(
                    int(x * location_downsample_fac),
                    int(y * location_downsample_fac),
                ),
                level=level,
                size=(dim1_edgelength, dim2_edgelength),
                as_array=True,  # seems to be a little faster than converting from PIL
            )
        )
        # region = build_test_tensor(dim1_edgelength, dim2_edgelength)
        assert region.shape[0] == 3, "Expected RGB image, after to_tensor it should be channels first"
        yield (row, x), (col, y), region


def convert_region_to_repr(
    enc: PyramidEncoder,
    region: torch.Tensor,
    patch_size: int,
    stride: int,
    from_repr: Literal["pyramid", "z"] = "z",
):
    batch, rows, cols = convert_tile_into_patchbatch(region, patch_size=patch_size, stride=stride)

    with torch.inference_mode():
        pyr, z = enc(batch.to(next(enc.parameters()).device))
        # Here you can encode debug information
        # z[:, 0] = x
        # z[:, 1] = y

    repr = pyr[-1] if from_repr == "pyramid" else z

    # Convert to rows, cols, channels, ...
    repr = repr.reshape(rows, cols, *repr.shape[1:]).swapaxes(1, 0)
    return repr


class RegionLoader(IterableDataset):
    def __init__(
        self,
        slide_paths: List[Path],
        extraction_level: int,
        patch_size: int,
        max_pixels_per_chunk: int,
    ):
        self.slide_paths, self.level, self.patch_size = (
            slide_paths,
            extraction_level,
            patch_size,
        )
        self.max_pixels_per_chunk = max_pixels_per_chunk

    def __iter__(self):
        worker_info = get_worker_info()

        # First move: find out which supercases this worker should process
        if worker_info is None:
            my_slide_paths = self.slide_paths
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id: int = worker_info.id
            my_slide_paths = self.slide_paths[worker_id::num_workers]
            logging.debug(
                f"Worker {worker_id} got {len(my_slide_paths)} supercases: {[x.stem for x in my_slide_paths]}"
            )

        for slide_path in my_slide_paths:
            slide = ts.TiffSlide(slide_path)
            # tf = tifffile.TiffFile(slide_path)

            num_regions_of_this_slide = 0
            tile_edgelength = get_tile_edge_length(
                patch_edge_length=self.patch_size,
                pixels_per_chunk=self.max_pixels_per_chunk,
            )
            positions = list(compute_positions(slide.level_dimensions[self.level], tile_edgelength))
            if worker_id == 0:
                logging.debug(f"I need to process {len(positions)} windows")
            try:
                for x in read_regions(slide, positions, tile_edgelength, self.patch_size, level=self.level):
                    yield slide_path, x
                    num_regions_of_this_slide += 1
            except Exception as e:
                logging.warning(f"Error while reading slide {slide_path}: {e}")

            if worker_info is None or worker_id == 0:
                logging.debug(f"There were {num_regions_of_this_slide} regions in slide {slide_path}")
                num_regions_of_this_slide = 0


def load_representations_for_tiffs(
    enc: PyramidEncoder,
    tiffpaths: List[Path],
    num_workers=4,
    level=0,
    patch_size=224,
    max_pixels_per_batch=96 * 224 * 224,
) -> Dict:
    region_ds = RegionLoader(tiffpaths, level, patch_size, max_pixels_per_batch)
    loader = DataLoader(region_ds, collate_fn=lambda x: x[0], batch_size=1, num_workers=num_workers)
    mem = {}
    for slide_path, ((row, x), (col, y), region) in tqdm(loader):
        if slide_path not in mem:
            mem[slide_path] = {}
        mem[slide_path][(row, col)] = (
            x,
            y,
            convert_region_to_repr(enc, region, patch_size, patch_size),
        )

    return mem


def build_nic(tiff_repr: Dict[Tuple[int, int], Tuple[int, int, torch.Tensor]]) -> torch.Tensor:
    """
    Based on the input dictionary which maps a tile position to a representation tensor,
    build the neural image representation of the whole slide.

    The representation can be converted to channels first by calling `nic.permute(2, 0, 1)`

    Confirm using:

    ```
    x = torch.Tensor([[[0.,  1.,  2.],
                    [3.,  4.,  5.]],

                    [[6.,  7.,  8.],
                    [9., 10., 11.]]])
    assert torch.equal(x[0, 1], x.permute(2, 0, 1)[:, 0, 1]) and torch.equal(x.permute(2, 0, 1)[:, 0, 1], torch.Tensor([3., 4., 5.]))
    ```
    """
    num_cols = [repr.shape[1] for (row, col), (x, y, repr) in tiff_repr.items() if row == 0]
    num_rows = [repr.shape[0] for (row, col), (x, y, repr) in tiff_repr.items() if col == 0]
    depth = tiff_repr[(0, 0)][2].shape[2]

    nic = torch.zeros(size=(sum(num_rows), sum(num_cols), depth), dtype=tiff_repr[(0, 0)][2].dtype)

    for (row, col), (x, y, repr) in tiff_repr.items():
        r = row * num_rows[0]
        c = col * num_cols[0]
        nic[r : r + repr.shape[0], c : c + repr.shape[1], :] = repr

    return nic


def process_tile(nic: torch.Tensor):
    """
    Removes the last row and column of the image and corrects the orientation.

    Probably SemiCOL challenge specific.
    """
    mirrored = nic.rot90(3).flip(1).permute(2, 0, 1)[:, :-1, :-1]

    return mirrored


def get_output_path_of_tile(in_path: Path, outfolder: Path) -> Path:
    return outfolder / in_path.parent.parts[-1] / f"{in_path.name}.pt"


def write_nic_from_list(
    trainfiles: List[Path],
    outfolder: Path,
    encoder: blocks.PyramidEncoder,
    level: int,
    patch_size: int,
    num_workers: int,
):
    """
    Function to write NICs from a list of absolute paths to a files. The files should be supported by the tiffslide package
    https://pypi.org/project/tiffslide/

    For now, we are operating with .tif and .svs files, which both are fully supported by the library.
    """

    while not_yet_existing := [
        filepath for filepath in trainfiles if not get_output_path_of_tile(filepath, outfolder).exists()
    ]:
        filepaths = random.sample(not_yet_existing, min(len(not_yet_existing), num_workers))
        representations = load_representations_for_tiffs(
            encoder,
            filepaths,
            num_workers=num_workers,
            patch_size=patch_size,
            level=level,
            max_pixels_per_batch=96 * 3 * patch_size * patch_size,
        )
        nics = {p: build_nic(representations[p]) for p in representations.keys()}

        for p, nic in nics.items():
            out_path = get_output_path_of_tile(p, outfolder)
            if not out_path.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(nic, out_path)
