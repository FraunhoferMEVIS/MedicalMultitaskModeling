import itertools
import logging
import random
from pathlib import Path
from typing import Any, Dict, Generator, List, Literal, Tuple

import numpy as np
import tiffslide as ts
import torch
import torchvision.transforms.functional as tvF
import torchvision.transforms.functional as F
import wandb
from m3_sdk.DistributedPath import DistributedPath
from m3_sdk.geojson import GeoAnno
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from tqdm.notebook import tqdm

from mmm.data_loading.geojson.NoUsefulWindowException import NoUsefulWindowException
from mmm.interactive import blocks
from mmm.interactive import configs as cfs
from mmm.mtl_modules.shared_blocks.PyramidEncoder import PyramidEncoder
from mmm.mtl_modules.shared_blocks.SharedModules import SharedModules
from mmm.mtl_modules.tasks.ClassificationTask import ClassificationTask
from mmm.utils import convert_tile_into_patchbatch, flatten_list_of_dicts


class WSIWindows(Dataset):
    @staticmethod
    def build_window_gen(for_slide, extraction, levels, sample_geojson, max_instances):
        valid_levels_in_this_slide = [l for l in levels if l < for_slide.level_count]
        window_generators = [
            extraction.iter_valid_windows(
                anno=GeoAnno(json_dict=anno, origin_id=i),
                level_downsamples={l: for_slide.level_downsamples[l] for l in valid_levels_in_this_slide},
            )
            for i, anno in enumerate(sample_geojson["features"])
        ]
        random.shuffle(window_generators)

        def cycle(generators):
            """Takes an iterable of generator objects and yields values from each generator
            in a round-robin fashion. If a generator is exhausted (raises StopIteration), it is removed
            from the list of generators."""
            generators_left = list(generators)
            while generators_left:
                for generator in generators_left:
                    try:
                        yield next(generator)
                    except StopIteration:
                        generators_left.remove(generator)
                    except NoUsefulWindowException:
                        logging.debug(f"No useful window found in {generator}")
                        generators_left.remove(generator)

        circling_generator = cycle(window_generators)

        window_generator = (
            (level, (y, x), patchsize_tuple)
            for level, (x, y), patchsize_tuple in itertools.islice(
                circling_generator,
                max_instances,
            )
        )
        return window_generator

    def __init__(
        self,
        levels_hw_sizes: list[tuple[int, tuple[int, int], tuple[int, int]]] | Generator,
        slide_path: DistributedPath,
    ) -> None:
        self.levels_hw_sizes, self.slide_path = levels_hw_sizes, slide_path
        self.slide_instance: ts.TiffSlide | None = None

    @staticmethod
    def open_slide(p: DistributedPath | Path | str, force_openslide: bool = False):
        if not isinstance(p, DistributedPath):
            dist_path = (
                DistributedPath.from_string(p) if isinstance(p, str) else DistributedPath.from_string(str(p.absolute()))
            )
        else:
            dist_path = p
        slide_needs_openslide = force_openslide or dist_path.upath().suffix in [".mrxs", ".dcm"]
        if slide_needs_openslide:
            try:
                from openslide import OpenSlide
            except ImportError:
                raise Exception(f"Openslide is not installed, but required for {dist_path.uri}")

            assert not dist_path.options, f"Openslide does only support local files, not {dist_path}"

            return OpenSlide(dist_path.to_local_path())
        else:
            return ts.TiffSlide(dist_path.file().open())

    def __len__(self):
        return len(self.levels_hw_sizes)

    def read_instance(self, level, row, col, edge1, edge2):
        self.slide_instance = self.open_slide(self.slide_path) if self.slide_instance is None else self.slide_instance

        # downsample_fac = self.slide_instance.level_downsamples[level]
        # row_l0, col_l0 = row * downsample_fac, col * downsample_fac
        row_l0, col_l0 = row, col
        try:
            is_openslide_slide = not isinstance(self.slide_instance, ts.TiffSlide)
            read_settings = {} if is_openslide_slide else dict(as_array=True)
            img = self.slide_instance.read_region(
                (col_l0, row_l0),  # the dataset is in (y, x) format, but tiffslide is in (x, y) format
                level,
                (edge1, edge2),
                **read_settings,
            )
            if is_openslide_slide:
                img = np.array(img)[..., :3]  # openslide returns RGBA, we only want RGB
        except Exception as e:
            img = np.zeros((edge1, edge2, 3), dtype=np.uint8)
            logging.error(f"Error while reading region {(row, col)}: {e}")
        return {"image": F.to_tensor(img), "row_col": (row, col), "level": level}

    def __getitem__(self, index) -> torch.Tensor:
        level, (row, col), (edge1, edge2) = self.levels_hw_sizes[index]
        return self.read_instance(level, row, col, edge1, edge2), (level, (row, col), (edge1, edge2))

    def __iter__(self):
        for level, (row, col), (edge1, edge2) in self.levels_hw_sizes:
            yield self.read_instance(level, row, col, edge1, edge2), (level, (row, col), (edge1, edge2))


class NICTask(ClassificationTask):
    """
    A classification task that processes neural image representations [C, H, W].
    """

    class Config(ClassificationTask.Config):
        pass

    def _visualize_preds(self, training_ims, step_metrics: Dict, metas: List[Dict]) -> Dict[str, Any]:
        vis_n = min(self.ask_for_visualization(), training_ims.size(0))

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
