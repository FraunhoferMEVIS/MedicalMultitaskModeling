from pathlib import Path
import os
from typing import Optional
from enum import Enum
import numpy as np
import imageio
import wandb
import re

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    go, px = None, None
import torch
import torch.nn as nn

from typing import List, Dict, Callable

import cv2

from mmm.DataSplit import DataSplit
from mmm.mtl_modules.tasks.MTLTask import MTLTask
from mmm.mtl_modules.shared_blocks.PyramidEncoder import PyramidEncoder


def remove_wandb_special_chars(s: str) -> str:
    for special_char in [".", "/"]:
        s = s.replace(special_char, "")
    return s


def convert_mtlimage_to_cv2(im: torch.Tensor) -> np.ndarray:
    """
    CV2 expects channels last for color and no channels at all for gray.
    """
    if im.shape[0] == 1:
        # Gray
        return im[0].numpy()
    else:
        # Our images are channels first, cv2 works with channels last
        channels_last_im = np.moveaxis(im.numpy(), 0, -1)
        return channels_last_im
        # raise NotImplementedError


def convert_cv2_to_mtlimage(cv2_im: np.ndarray) -> torch.Tensor:
    """
    Only works for 2D.
    """
    if len(cv2_im.shape) == 2:
        return torch.from_numpy(cv2_im).unsqueeze(0)
    else:
        # Color image
        channels_first_im = np.moveaxis(cv2_im, -1, 0)
        return torch.from_numpy(channels_first_im)


def resize_if_necessary(im: torch.Tensor, max_spatial_size: int) -> torch.Tensor:
    assert len(im.shape) == 3, "resizing only works for 2D"

    max_edge_length = max(im.shape[1], im.shape[2])
    if max_edge_length > max_spatial_size:
        factor: float = max_spatial_size / max_edge_length
        resized_np = cv2.resize(
            convert_mtlimage_to_cv2(im),
            (0, 0),
            fx=factor,
            fy=factor,
            interpolation=cv2.INTER_AREA,
        )
        return convert_cv2_to_mtlimage(resized_np)
    else:
        return im


def resize_3d_if_necessary(im: torch.Tensor, max_spatial_size: int) -> torch.Tensor:
    assert len(im.shape) == 4, "resizing only works for 3D"
    max_edge_length = max(im.shape[1], im.shape[2])
    if max_edge_length > max_spatial_size:
        factor: float = max_spatial_size / max_edge_length

        slices = [
            cv2.resize(
                convert_mtlimage_to_cv2(im[..., slice_id]),
                (0, 0),
                fx=factor,
                fy=factor,
                interpolation=cv2.INTER_AREA,
            )
            for slice_id in range(im.shape[-1])
        ]

        return torch.stack([torch.from_numpy(x) for x in slices], dim=-1)
    else:
        return im


def build_wandb_image(im: torch.Tensor, caption: str, resample_to_spatial_size: Optional[int] = None) -> wandb.Image:
    if len(im.shape) == 3:
        # 2D images
        im = resize_if_necessary(im, resample_to_spatial_size) if resample_to_spatial_size is not None else im
        return wandb.Image(im, caption=caption)
    elif (len(im.shape)) == 4:
        # Plot a gif over the last dimension
        tmp_dir = Path(wandb.run.dir) / "volumes"
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = tmp_dir / f"out_{len(os.listdir(tmp_dir))}.gif"
        im = resize_3d_if_necessary(im, resample_to_spatial_size) if resample_to_spatial_size is not None else im
        volume = np.moveaxis(np.squeeze(im.numpy() * 255).astype("uint8"), -1, 0)
        imageio.mimsave(tmp_path, volume)
        return wandb.Image(str(tmp_path), caption=caption)

    raise Exception(f"Image is of unexpected shape: {im.shape}")


def build_wandb_image_for_clf(
    im: torch.Tensor,
    gt: np.ndarray,
    pred: np.ndarray,
    class_names: List[str],
    resample_to_spatial_size: Optional[int] = None,
    caption_suffix="",
):
    description = f"{np.min(im.numpy()):.3f}, {np.max(im.numpy()):.3f}; {im.size()}"
    true_str = f"{class_names[gt]} - {gt}"
    pred_str = f"{class_names[pred]} - {pred}"

    wandb_img = build_wandb_image(
        im=im,
        caption=f"{description}\nTrue: {true_str}\nPred: {pred_str}\n{caption_suffix}",
        resample_to_spatial_size=resample_to_spatial_size,
    )

    return wandb_img, description, true_str, pred_str


class PlotModuleMode(Enum):
    full, aggregated, summary = "full", "aggregated", "summary"


def get_module_state(m: nn.Module, comparison_state=None) -> Dict:
    res = {}

    for parameter_name, parameter_tensor in m.named_parameters():
        # clone might not be necessary, but lets be safe until that assumption is tested
        res[f"{parameter_name}"] = parameter_tensor.detach().cpu().clone().flatten()
        if comparison_state and parameter_name in comparison_state:
            res[f"{parameter_name}_change"] = res[f"{parameter_name}"] - comparison_state[parameter_name]

        if parameter_tensor.grad is not None:
            # clone might not be necessary, but lets be safe until that assumption is tested
            res[f"{parameter_name}_grad"] = parameter_tensor.grad.cpu().clone().flatten()
            if comparison_state and f"{parameter_name}_grad" in comparison_state:
                res[f"{parameter_name}_grad_change"] = res[f"{parameter_name}_grad"] - comparison_state[parameter_name]

    return res


def convert_module_state_to_wandb(d: Dict, mode: PlotModuleMode) -> Dict:
    res = {}
    if mode is PlotModuleMode.full:
        for param_name, param_value in d.items():
            if isinstance(param_value, torch.Tensor):
                res[param_name] = wandb.Histogram(param_value)  # type: ignore
            else:
                res[param_name] = param_value
    elif mode is PlotModuleMode.aggregated:
        buffer: Dict[str, List] = {
            "params": [],
            "grads": [],
            "param_changes": [],
            "grad_changes": [],
        }

        aggregator = torch.mean
        for param_name, param_value in d.items():
            if isinstance(param_value, torch.Tensor):
                if param_name.endswith("_grad_change"):
                    buffer["grad_changes"].append(aggregator(param_value))
                elif param_name.endswith("_grad"):
                    buffer["grads"].append(aggregator(param_value))
                elif param_name.endswith("_change"):
                    buffer["param_changes"].append(aggregator(param_value))
                else:
                    buffer["params"].append(aggregator(param_value))
            else:
                res[param_name] = param_value

        assert not buffer["grads"] or len(buffer["params"]) == len(buffer["grads"])
        assert not buffer["param_changes"] or (len(buffer["param_changes"]) == len(buffer["params"]))

        for k, v in buffer.items():
            if v:
                res[f"{k}_aggregated"] = wandb.Histogram(v)  # type: ignore
    else:
        raise NotImplementedError()
    return res


def _extract_data_by_regex(runs: List[Dict], r, convert_to_int: Callable) -> List:
    data = []
    for run in runs:
        summary = run.summary
        summary_keys = list(summary.keys())
        # breakhis20/tl.bestval.breakhis20epoch.acc
        found = list(filter(r.match, summary_keys))
        d = [[convert_to_int(entry), summary[entry]] for entry in found]
        d.sort()
        for x, y in d:
            data.append([x, y])

    data.sort()
    return data


def _create_line_from_data(data: List, name: str, colors, confidence=False) -> List:
    figures = []
    if not confidence:
        x = [a[0] for a in data]
        y = [a[1] for a in data]
        figures.append(go.Scatter(x=x, y=y, name=name, fillcolor=colors[0]))
    if confidence:
        x = [a[0] for a in data]
        higher = []
        lower = []
        y = []
        unique, idxs, counts = np.unique(x, return_index=True, return_counts=True)
        for i, idx in enumerate(idxs):
            # if counts[i]>1:
            points = data[idx : idx + counts[i]]
            y_tmp = np.mean(points, axis=0)
            y.append(y_tmp[-1])
            higher.append(np.max(points, axis=0)[-1])
            lower.append(np.min(points, axis=0)[-1])
        unique = list(unique)
        higher = list(higher)
        lower = list(lower)
        y = list(y)
        figures.append(go.Scatter(x=unique, y=y, name=name, legendgroup=name, line_color=colors[0]))
        figures.append(
            go.Scatter(
                x=unique + unique[::-1],
                y=higher + lower[::-1],
                fill="toself",
                fillcolor=colors[1],
                line_color="rgba(255,255,255,0)",
                legendgroup=name,
                hoverinfo="skip",
            )
        )

    return figures


def create_sample_efficiency_plot(
    run_id: str,
    project_name: str,
    downstream_tasks,
    entity: str = "tissue-concepts",
    metric: str = "acc",
    compare_to_imageNet=None,
    compare_to_st=None,
    include_traditional_models=[],
):
    colors = px.colors.qualitative.Set2
    for color in px.colors.qualitative.Set1:
        colors.append(color)
    # colors = []
    # for color in px_colors:
    #    colors.append(f"rgba{color[3:-1]},0.3)")
    api = wandb.Api()
    run = api.run(f"{entity}/{project_name}/{run_id}/")

    fig = go.Figure()
    color_idx = 0

    # Evaluatr task performance
    for idx, task in enumerate(downstream_tasks):
        # clf head
        r = re.compile(f"{task}(.*)/tl.bestval.{task}(.*)epoch.{metric}")
        data = _extract_data_by_regex([run], r, lambda x: int(x.split("/")[0].split(f"{task}")[-1]))
        for img in _create_line_from_data(data, f"{task}", [colors[color_idx]]):
            fig.add_trace(img)
            color_idx += 1

        # img baseline
        if compare_to_imageNet is not None:
            r = re.compile(f"{task}(.*)/tl.bestval.{task}(.*).{metric}")
            data = _extract_data_by_regex(
                runs=api.runs(
                    path=compare_to_imageNet,
                    filters={"display_name": {"$regex": f"ImageNet-binary"}},
                ),
                r=r,
                convert_to_int=lambda x: int(x.split(f"{task}")[-1].split("_")[0].split("epoch")[0]),
            )
            for img in _create_line_from_data(
                data,
                f"{task}",
                [colors[color_idx], f"rgba{colors[color_idx][3:-1]},0.3)"],
                confidence=True,
            ):
                fig.add_trace(img)
                color_idx += 1

        if compare_to_st is not None:
            r = re.compile(f"{task}(.*)/tl.bestval.{task}(.*).{metric}")
            data = _extract_data_by_regex(
                runs=api.runs(
                    path=compare_to_imageNet,
                    filters={"display_name": {"$regex": f"{task}(.*)"}},
                ),
                r=r,
                convert_to_int=lambda x: int(x.split(f"{task}")[-1].split("_")[0].split("epoch")[0]),
            )
            for img in _create_line_from_data(
                data,
                f"{task}-ST",
                [colors[color_idx], f"rgba{colors[color_idx][3:-1]},0.3)"],
                confidence=True,
            ):
                fig.add_trace(img)
                color_idx += 1

        for model in include_traditional_models:
            r = re.compile(f"{task}(.*).{model}_val_{metric}")
            data = _extract_data_by_regex(
                runs=[run],
                r=r,
                convert_to_int=lambda x: int(x.split(f"{task}")[1].split("_")[0]),
            )
            for img in _create_line_from_data(data, f"{model}-{task}", [colors[color_idx]]):
                fig.add_trace(img)
                color_idx += 1

                if compare_to_imageNet is not None:
                    data = _extract_data_by_regex(
                        runs=api.runs(
                            path=compare_to_imageNet,
                            filters={"display_name": {"$regex": f"ImageNet-binary"}},
                        ),
                        r=r,
                        convert_to_int=lambda x: int(x.split(f"epoch")[0].split(f"{task}")[-1].split("_")[0]),
                    )
                    for img in _create_line_from_data(
                        data,
                        f"{model}-ImageNet",
                        [colors[color_idx], f"rgba{colors[color_idx][3:-1]},0.3)"],
                        confidence=True,
                    ):
                        fig.add_trace(img)
                        color_idx += 1

    fig.update_layout(xaxis_title="Percent Samples", yaxis_title=f"best overall {metric}")

    return fig


def _get_imgnet_defaults(api, path, task, metric):
    runs = api.runs(path=path, filters={"display_name": {"$regex": f"ImageNet"}})
    data = []
    for run in runs:
        summary = run.summary
        summary_keys = list(summary.keys())
        r = re.compile(f"epoch_tl.bestval(.*){task}(.*)_{metric}")
        found = list(filter(r.match, summary_keys))
        d = [[int(entry.split(f"{task}")[-1].split("_")[0]), summary[entry]] for entry in found]

        d.sort()
        for x, y in d:
            data.append([x, y])

    data.sort()
    x = [a[0] for a in data]
    higher = []
    lower = []
    y = []
    unique, idxs, counts = np.unique(x, return_index=True, return_counts=True)
    for i, idx in enumerate(idxs):
        points = data[idx : idx + counts[i]]
        y_tmp = np.mean(points, axis=0)
        y.append(y_tmp[-1])
        higher.append(np.max(points, axis=0)[-1])
        lower.append(np.min(points, axis=0)[-1])
    unique = list(unique)
    higher = list(higher)
    lower = list(lower)
    y = list(y)

    line = go.Scatter(x=unique, y=y, name=f"ImageNet-{task}")

    return [
        go.Scatter(
            x=unique + unique[::-1],
            y=higher + lower[::-1],
            fill="toself",
            fillcolor="rgba(205,205,205,0.5)",
            line_color="rgba(255,255,255,0)",
            showlegend=False,
            hoverinfo="skip",
            name=f"ImageNet-{task}",
        ),
        line,
    ]


def _get_st_plot(api, path, task, metric):
    runs = api.runs(path=path, filters={"display_name": {"$regex": f"{task}[0-9][0-9][0-9]"}})
    data = []
    for run in runs:
        name = run.name
        last = run.summary[f"epoch_mtlval_{metric}"]
        perc = int(name.split(f"{task}")[-1].split("_")[0])
        data.append([perc, last])

    data.sort()
    x = [a[0] for a in data]
    # y = [a[1] for a in data]

    higher = []
    lower = []
    y = []
    unique, idxs, counts = np.unique(x, return_index=True, return_counts=True)
    for i, idx in enumerate(idxs):
        # if counts[i]>1:
        points = data[idx : idx + counts[i]]
        y_tmp = np.mean(points, axis=0)
        y.append(y_tmp[-1])
        higher.append(np.max(points, axis=0)[-1])
        lower.append(np.min(points, axis=0)[-1])
    unique = list(unique)
    higher = list(higher)
    lower = list(lower)
    y = list(y)

    line = go.Scatter(x=unique, y=y, name=f"SingleTask-{task}")

    return [
        go.Scatter(
            x=unique + unique[::-1],
            y=higher + lower[::-1],
            fill="toself",
            fillcolor="rgba(205,205,205,0.5)",
            line_color="rgba(255,255,255,0)",
            showlegend=False,
            hoverinfo="skip",
            name=f"SingleTask-{task}",
        ),
        line,
    ]


@torch.no_grad()
def multitask_embedding_table(ts: List[MTLTask], enc: PyramidEncoder, max_batches: int) -> wandb.Table:
    from mmm.task_sampling import BalancedTaskSampler
    from tqdm.auto import tqdm

    logtable = wandb.Table(columns=["taskid", "tooltip", "repr", "internal_group"])

    for t in ts:
        t.cohort.prepare_epoch(epoch=0)

    tasksampler = BalancedTaskSampler(BalancedTaskSampler.Config(), ts, DataSplit.val)

    for batchcounter, (batch, task) in tqdm(enumerate(tasksampler)):
        _, embs = enc(batch["image"].to(enc.torch_device))
        cls_names: List[str] = task.class_names  # type: ignore
        for batch_idx in range(len(batch)):
            class_name = cls_names[batch["class"][batch_idx].item()]
            img = wandb.Image(batch["image"][batch_idx, ...].detach().cpu(), caption=f"{class_name}")
            logtable.add_data(
                task.get_name(),
                img,
                list(embs[batch_idx].detach().cpu().numpy()),
                f"{task.get_name()}_{class_name}",
            )

        if batchcounter >= max_batches:
            break

    return logtable
