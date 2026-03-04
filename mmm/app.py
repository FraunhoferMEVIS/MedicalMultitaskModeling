# Import big libraries on demand, as it would slow down the startup time of the CLI.
import json
import os
import uuid
from pathlib import Path
from typing import Annotated

import typer
from m3_sdk.types import CompressType

from mmm.mmm_types.ImageEncoding import ImageEncoding
from mmm.mmm_types.LabelType import LabelType

app = typer.Typer(no_args_is_help=True)


@app.command()
def status():
    from mmm.settings import mtl_settings

    typer.secho("MMM Settings", fg=typer.colors.BRIGHT_BLUE)
    typer.secho(mtl_settings.model_dump_json(indent=2))

    try:
        assert mtl_settings.kv.ping()
        typer.secho(f"Redis is available at {mtl_settings.redis_url}!", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Redis not available: {e}", fg=typer.colors.YELLOW)

    try:
        # Test if labelstudio is available
        from label_studio_sdk import Client

        client = Client(url=mtl_settings.labelstudio["base_url"], api_key=mtl_settings.labelstudio["api_key"])
        assert client.check_connection()
        typer.secho(f"Labelstudio is available at {mtl_settings.labelstudio['base_url']}", fg=typer.colors.GREEN)
    except Exception as e:
        typer.secho(f"Labelstudio not available: {e}", fg=typer.colors.YELLOW)

    try:
        from mmm.api.celery import get_number_of_celery_workers, ws

        typer.secho(f"Celery available with {get_number_of_celery_workers()} workers!", fg=typer.colors.BRIGHT_BLUE)
        typer.secho(ws.settings.model_dump_json(indent=2))
    except Exception as e:
        typer.secho(f"Celery not available: {e}", fg=typer.colors.YELLOW)

    # Test if mmm is importable
    typer.secho("Testing if mmm is importable...", fg=typer.colors.BRIGHT_BLUE)
    from mmm.interactive import blocks, configs, data, pipes, tasks, training, utils

    typer.secho("mmm is importable!", fg=typer.colors.GREEN)


@app.command("api")
@app.command(hidden=True)  # for backward compatibility
def inference_app(host: str = "0.0.0.0", port: int = 8000):
    """Starts HTTP API for MMM models."""
    try:
        import uvicorn

        from mmm.inference_app import build_app
    except ImportError as e:
        typer.secho(
            f"Please install the extra dependencies 'api' of medicalmultitaskmodeling: {e}", fg=typer.colors.RED
        )
        return

    uvicorn.run(build_app(), host=host, port=port)


@app.command()
def worker(
    worker_name: str = typer.Argument(
        default_factory=lambda: f"worker_{uuid.uuid4().hex}", help="The name of the worker"
    ),
    queues: list[str] | None = None,
    local_rank: Annotated[
        int,
        typer.Option(
            envvar="LOCAL_RANK",
            help="Used to determine the GPU to use on this node.",
        ),
    ] = 0,
):
    """
    Wraps celery worker commands with the mmm settings. Alternatively, you can start workers similar to:

    `LOCAL_RANK=0 celery -A mmm.api.api_worker:app worker -P solo -c 1 -n rank0@%h`

    Queues:

    - m3worker (default queue) for tasks that are not CPU or GPU heavy and can be parallelized
    - CPU for CPU heavy tasks
    - GPU for tasks that require a CUDA enabled GPU
    """
    import torch

    from mmm.api.api_worker import app as celery_app

    if queues is None:
        queues = ["m3worker", "cpu"]

        if torch.cuda.is_available():
            queues.append(f"gpu")

    heavy_tasks = True in [q in ["cpu", "gpu"] for q in queues]

    options = {
        # threads enables communication with worker during training, but has problems with torch multiprocessing -> solo
        "pool": "solo" if heavy_tasks else "prefork",
        "concurrency": 1 if heavy_tasks else 0,  # 0 means the number of cores
        "hostname": f"{worker_name}{local_rank}@%h",
        "queues": ",".join(queues),
    }
    typer.secho(
        f"Starting worker {worker_name} for queues {queues} for heavy tasks: {heavy_tasks=}", fg=typer.colors.GREEN
    )
    worker = celery_app.Worker(**options)  # type: ignore
    worker.start()


@app.command()
def create_demo_data(
    folder: Annotated[
        str, typer.Argument(envvar="MMM_DEMO_DATA_FOLDER", help="The folder where demo data will be created in")
    ] = "./mmm_demo_data",
    pet_data_root_dir: Annotated[
        str,
        typer.Option(
            envvar="MMM_DEMO_DATA_ROOT_DIR",
            help="Used for storing the pet data in a folder named [INPUT]/oxford-iiit-pet",
        ),
    ] = "/mmm",
    num_cases: int = 20,
    image_encoding: ImageEncoding = ImageEncoding.base64,
    max_bag_size: int = 3,
    for_label: list[LabelType] = [LabelType.seg, LabelType.clf],
    num_classes: int = 3,
    yy: Annotated[bool, typer.Option("-y", envvar="MMM_DEMO_DATA_YES_TO_ALL", help="Accept all")] = False,
):
    from tqdm.auto import tqdm

    from mmm.api.evaluation import TaskDefinition, TaskMetadata
    from mmm.api.PetTestData import PetTestData

    cfg = {
        "for_label": [x.value for x in for_label],
        "max_bag_size": max_bag_size,
        "image_encoding": image_encoding.value,
        "num_classes": num_classes,
        "num_cases": num_cases,
    }

    ds = PetTestData(PetTestData.Config(**cfg), data_directory=pet_data_root_dir)
    data_folder: Path = Path(folder)
    if data_folder.exists():
        typer.secho(f"{data_folder=} already exists", fg=typer.colors.YELLOW)

    # Ask the user if the files should be created
    if yy or typer.confirm(f"Create data for {ds.cfg.model_dump_json(indent=2)} in {data_folder.absolute()}?"):
        # Write to disk
        data_folder.mkdir(parents=True, exist_ok=True)
        td = TaskDefinition(
            meta=TaskMetadata(
                tags=["natural", "unittesting"], license="https://creativecommons.org/licenses/by-sa/4.0/"
            ),
            labeling=ds.get_labeling(),
        )
        (data_folder / "task-definition.json").write_text(td.model_dump_json(indent=2, exclude_none=True))
        typer.secho(f"Classes: {ds.classes}", fg=typer.colors.GREEN)

        (data_folder / td.data_json_files[0].upath()).write_text(
            json.dumps(
                [
                    ds.create_subject_from_case(ds[i], subj_id=f"pets_{i}").model_dump(exclude_none=True)
                    for i in tqdm(range(len(ds)))
                ]
            )
        )
        typer.secho([str(x) for x in data_folder.iterdir()])
        return data_folder / "task-definition.json"
    else:
        typer.secho("Aborted", fg=typer.colors.RED)


@app.command()
def label(
    project_name: str,
    task_definition: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)] = Path.cwd()
    / "task-definition.json",
):
    """
    Commits a task definition to a labelstudio instance specified in the settings.
    """
    from m3_sdk.DistributedPath import DistributedPath

    from mmm.api.evaluation import TaskDefinition
    from mmm.api.LabelStudioFrontend import LabelStudioFrontend
    from mmm.settings import mtl_settings as mtl_settings

    if any(p.title == project_name for p in mtl_settings.ls.get_projects()):  # type: ignore
        typer.confirm(f"Project {project_name} already exists. Do you want to continue?", abort=True)

    task: TaskDefinition = TaskDefinition(**json.loads(task_definition.read_text())).make_relative_paths_absolute(
        DistributedPath.from_string(task_definition.parent.absolute().as_uri())
    )
    lsf = LabelStudioFrontend(
        task.get_subjects(),
        cfg=LabelStudioFrontend.Config(project_name=project_name, labeling=task.get_labeling_config()),
    )
    typer.secho(f"Project {project_name} with {len(lsf.tasks)} subjects at {lsf.get_url()}", fg=typer.colors.GREEN)


@app.command()
def cache(
    subjects_json: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False)] = Path.cwd()
    / "data.json",
    for_type: CompressType = CompressType.token,
    with_labels: None | list[str] = None,
    subjects_per_task: int = 1,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            "-b",
            help="""Batchsize used for compression.
A larger batchsize enables the compression to be faster if dataloading is not a bottleneck.
If the batchsize is bigger than 1, images of different size will be resized.
If the batchsize is too large, the GPU might run out of memory.""",
        ),
    ] = 1,
    num_workers_per_gpu: Annotated[int, typer.Option("--workers-per-gpu", "-w", help="Workers per process (GPU)")] = 0,
):
    """
    Caches a list of subjects into a key-value store using a multitask model.
    """
    from celery.result import AsyncResult

    from mmm.api.api_worker import CacheSubjects, cache_instances, cache_subjects
    from mmm.settings import mtl_settings

    if subjects_per_task < num_workers_per_gpu:
        typer.secho(
            "The number of subjects per task should be larger than the number of workers per GPU (and a multiple)",
            fg=typer.colors.RED,
        )
        return

    all_subjects = [s for s in json.loads(subjects_json.read_text())]

    all_subject_keys: list[bytes] = cache_subjects.delay(CacheSubjects.Args(subjects=all_subjects)).get()
    subj_keys = all_subject_keys

    # Already compressed instances are part of the set compressed:for_type
    if already_compressed := set(map(lambda s: s.decode(), mtl_settings.kv.smembers(f"compressed:{for_type.value}"))):
        subj_keys_already_compressed: list[bytes] = [
            s for s in all_subject_keys if s.decode()[len(mtl_settings.subj_prefix) + 1 :] in already_compressed
        ]
        typer.secho(
            f"Skipping {len(subj_keys_already_compressed)} subjects that are already compressed for {for_type}",
        )
        if len(subj_keys := [s for s in subj_keys if s not in subj_keys_already_compressed]) == 0:
            typer.secho("All subjects are already compressed", fg=typer.colors.GREEN)
            return all_subject_keys

    pending_tasks: list[AsyncResult] = []
    subject_batches = [
        subj_keys[subject_batch : subject_batch + subjects_per_task]
        for subject_batch in range(0, len(subj_keys), subjects_per_task)
    ]
    typer.secho(
        f"Processing {len(subj_keys)} subjects in batches: {[len(b) for b in subject_batches]}",
        fg=typer.colors.GREEN,
    )
    for chunk_subjects in subject_batches:
        pending_tasks.append(
            cache_instances.delay(
                for_type,
                chunk_subjects,
                with_labels,
                batch_size=batch_size,
                num_workers=num_workers_per_gpu,
                skip_if_exists=True,
            )
        )

    p = mtl_settings.kv.pubsub()
    p.subscribe("compress")

    while pending_tasks:
        if ready_tasks := [task for task in pending_tasks if task.ready()]:
            for t in ready_tasks:
                pending_tasks.remove(t)
                num_new_instances, num_all_instances = t.get()
                typer.secho(
                    f"Finished task by adding {num_new_instances} new instances of {num_all_instances}. "
                    f"Pending & Ongoing tasks: {len(pending_tasks)}",
                    fg=typer.colors.GREEN,
                )

        msg = p.get_message(timeout=1.5)
        if msg is not None and not isinstance(msg["data"], int):
            d = json.loads(msg["data"].decode())
            time = f"{d['data_loading_time_s']:.3f}, {d['case_processing_time_s']:.3f}, model:{d['model_time_s']:.3f}"
            typer.secho(f"Iteration: {time}")

    return all_subject_keys


@app.command()
def reset(
    extra_patterns: Annotated[
        list[str], typer.Option("--extra", "-e", help="Extra patterns such as 'customkeys:*'")
    ] = None,
    default_patterns: Annotated[bool, typer.Option(help="Delete default M3 API patterns")] = True,
):
    """
    Deletes all keys managed by the M3 API from the DB. Extra arguments are treated as extra patterns to delete.
    """
    from mmm.settings import mtl_settings

    if extra_patterns is None:
        extra_patterns = []

    delete_patterns = (
        [
            "gt:*",
            "datasets:*",
            "repr:*",
            "task:*",
            "subject:*",
            "compressed:*",
            "m3api:*",
            "celery_api_worker:*",
        ]
        if default_patterns
        else []
    )

    patterns_without_keys = []
    for pattern in delete_patterns + extra_patterns:
        if delkeys := list(mtl_settings.kv.keys(pattern)):
            typer.secho(f"Deleting {len(delkeys)} keys for pattern {pattern}", fg=typer.colors.GREEN)
            mtl_settings.kv.delete(*delkeys)
        else:
            patterns_without_keys.append(pattern)

    if patterns_without_keys:
        typer.secho(f"No keys found for patterns: {patterns_without_keys}", fg=typer.colors.YELLOW)


@app.command()
def st():
    dashboard_path = Path(__file__).parent / "api" / "st_main.py"
    assert dashboard_path.exists(), f"Dashboard path {dashboard_path} was not found"

    os.system(f"python -m streamlit run {dashboard_path}")


if __name__ == "__main__":
    app()
