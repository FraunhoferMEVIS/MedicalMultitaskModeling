import json
from pathlib import Path
from typing import Generator, Hashable, Literal

from m3_sdk.DistributedPath import DistributedPath
from pydantic import Field
from typing_extensions import Annotated

from mmm.api.models import MSubject
from mmm.api.mtl_adapter import LabelingConfig
from mmm.BaseModel import BaseModel
from mmm.data_loading.utils import train_val_split


class TaskMetadata(BaseModel):
    # task metadata
    license: str | None = Field(default=None, description="If applicable, just use 'CC BY 4.0' or MIT.")
    description: str | None = Field(
        default=None,
        description="1-2 sentence description such as 'x-ray classification of abdomen for the subset from munich'.",
    )
    source: str | None = Field(default=None, description="If possible a link.")
    official_name: str | None = Field(
        default=None, description="The name of the data source that is used in literature."
    )
    dataset_size: str | None = Field(
        default=None, description="For example, '200 3D-CT volumes' or '1000 natural photos'."
    )
    tags: list[str] = Field(default=[], description="Free tags such as 'histo', 'nonmedical', 'breast'")

    def get_keyvalue_properties(self) -> dict[str, str]:
        """
        Returns fields that M3 has no special treatment for.
        Intended for unstructured metadata.
        """
        res = self.model_dump(exclude_none=True, exclude_defaults=True)
        res.pop("tags", None)
        return res


class CrossValidation(BaseModel):
    split_type: Literal["cross_validation"] = "cross_validation"
    n_splits: int = 5
    splitting_seed: int = 42
    shuffle: bool = True
    train_val_perc: float = 0.7

    def compute_splits(
        self, cases: list[Hashable]
    ) -> Generator[tuple[str, list[int], list[int] | None, list[int]], None, None]:
        from sklearn.model_selection import KFold

        for i, (train_indices, test_indices) in enumerate(
            KFold(
                n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.splitting_seed if self.shuffle else None
            ).split(cases)
        ):
            traintrain_indices, trainval_indices = train_val_split(
                list(train_indices), perc=self.train_val_perc, seed=self.splitting_seed
            )
            yield f"kfold_{i}", traintrain_indices, trainval_indices, list(test_indices)


class ByMetaInfo(BaseModel):
    split_type: Literal["by_meta_value"] = "by_meta_value"
    train_criterion: tuple[str, str] = Field(description="Key-pattern pair within meta info.")
    val_criterion: tuple[str, str] | None = Field(
        default=None, description="Key-pattern pair within meta info. (optional)"
    )
    test_criterion: tuple[str, str] = Field(description="Key-pattern pair within meta info.")

    def compute_splits(
        self, cases: list[MSubject]
    ) -> Generator[tuple[str, list[int], list[int] | None, list[int]], None, None]:
        import re

        train_indices = []
        val_indices = []
        test_indices = []
        for i, case in enumerate(cases):
            if re.match(self.train_criterion[1], case.meta[self.train_criterion[0]]):
                train_indices.append(i)
            if self.val_criterion and re.match(self.val_criterion[1], case.meta[self.val_criterion[0]]):
                val_indices.append(i)
            if re.match(self.test_criterion[1], case.meta[self.test_criterion[0]]):
                test_indices.append(i)
        yield f"meta_{self.train_criterion[0]}", train_indices, val_indices or None, test_indices


SplitType = ByMetaInfo | CrossValidation


class TaskDefinition(BaseModel):
    meta: TaskMetadata | None = Field(default=None, description="Meta information about the task.")
    splits: None | list[Annotated[SplitType, Field(discriminator="split_type")]] = Field(
        default=[CrossValidation()],
        description="By default, KFold CV is applied",
    )
    data_json_files: list[DistributedPath] = Field(
        default=[DistributedPath(uri="data.json")],
        description="Should be a list of dictionary objects that can be validated with `mmm.api.models.MSubject`",
        examples=[
            [{"url": "data-new-york.json"}, {"url": "data-los-angeles.json"}],
        ],
        title="Data Collections",
    )
    labeling: LabelingConfig | DistributedPath = Field(
        examples=[DistributedPath(uri="labeling.xml")],
    )

    def add_to_wandb(
        self,
        project: str,
        by_reference: bool = True,
        registry_name: str = "Universal Evaluation",
        artifact_type: str = "m3-api-data",
        collection: str = "",
    ):
        """
        Uploads the task definition and associated data json files to Weights and Biases as an artifact.

        Args:
            project (str): The W&B project name where the producing run will exist.
            by_reference (bool): If subject data is large or private, only a reference will be uploaded to W&B.
        """
        import tempfile
        from urllib.parse import urlparse

        import wandb

        assert self.meta is not None, "Metadata must be provided to upload to WandB."
        assert self.meta.official_name is not None, "Official name must be provided in metadata to upload to WandB."
        assert self.meta.description is not None, "Description must be provided in metadata to upload to WandB."
        with wandb.init(
            project=project, name=self.meta.official_name, tags=self.meta.tags, reinit="return_previous"
        ) as run:
            artifact = wandb.Artifact(
                name=self.meta.official_name,
                type=artifact_type,
                description=self.meta.description,
                metadata=self.meta.get_keyvalue_properties(),
            )

            new_paths = []  # If the file is included directly, the path needs to be relative
            for distpath in self.data_json_files:
                if by_reference:
                    artifact.add_reference(distpath.uri)
                    new_paths.append(distpath)
                else:
                    artifact.add_file(urlparse(distpath.uri).path, name=distpath.upath().name)
                    new_paths.append(DistributedPath.from_string(distpath.upath().name))
            self.data_json_files = new_paths

            with tempfile.TemporaryDirectory() as tmpdir:
                (p := Path(tmpdir, "task-definition.json")).write_text(
                    self.model_dump_json(indent=2, exclude_none=True)
                )
                artifact.add_file(str(p.absolute()), name="task-definition.json")

            logged_artifact = run.log_artifact(artifact)
            if not collection:
                collection = self.meta.official_name
            run.link_artifact(logged_artifact, f"wandb-registry-{registry_name}/{collection}")

        return run.url

    def get_minimum_cohort_size(self) -> int:
        if not self.splits:
            return 1

        res = 0
        for split in self.splits:
            if isinstance(split, CrossValidation):
                res += split.n_splits

        return max(res, 1)

    def get_labeling_config(self) -> LabelingConfig:
        if isinstance(self.labeling, LabelingConfig):
            return self.labeling
        else:
            return LabelingConfig(xml=self.labeling.upath().read_text())

    def make_relative_paths_absolute(self, data_root: DistributedPath | Path):
        if isinstance(data_root, Path):
            data_root = DistributedPath.from_string(data_root.as_posix())

        if (not isinstance(self.labeling, LabelingConfig)) and (not self.labeling.upath().is_absolute()):
            self.labeling = data_root / self.labeling

        json_files = []
        for json_file in self.data_json_files:
            if not json_file.upath().is_absolute():
                json_files.append(data_root / json_file)
            else:
                json_files.append(json_file)
        self.data_json_files = json_files
        return self

    def generate_splits(
        self, cases: list[MSubject] | None = None
    ) -> Generator[tuple[str, list[int], list[int] | None, list[int]], None, None]:
        if cases is None:
            cases = self.get_subjects()
        if not self.splits:
            yield "default", list(range(len(cases))), None, list(range(len(cases)))
        else:
            for split in self.splits:
                if isinstance(split, CrossValidation | ByMetaInfo):
                    for split_id, train, val, test in split.compute_splits(cases):
                        yield split_id, train, val, test
                else:
                    raise NotImplementedError(f"Splitting method {split} not implemented")

    def get_subjects(self, data_sources=None) -> list[MSubject]:
        if data_sources is None:
            data_sources = []
        subjects = []
        for data_source in self.data_json_files + data_sources:
            if isinstance(data_source, DistributedPath):
                subjects.extend(json.loads(data_source.upath().read_text()))
            elif hasattr(data_source, "read_text"):
                subjects.extend(json.loads(data_source.read_text()))
            elif isinstance(data_source, dict):
                subjects.append(data_source)
            else:
                assert isinstance(data_source, list)
                subjects.extend(data_source)
        return [MSubject(**subject) for subject in subjects]
