import json
import uuid
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from mmm.api.models import MSubject
from mmm.api.mtl_adapter import LabelingConfig
from mmm.settings import mtl_settings as s

try:
    from label_studio_sdk import Client, Project

except ImportError:
    if not TYPE_CHECKING:
        Client, Project = None, None
    else:
        raise  # Avoids errors in type checking tools


class LabelStudioFrontend:
    """
    Creates a project in Labelstudio for a list of MSubjects.
    """

    class Config(BaseModel):
        project_name: str = Field(default_factory=lambda: f"LSF_{uuid.uuid4().hex[:8]}")
        color: str = "#0000FE"
        labeling: LabelingConfig

    def __init__(self, tasks: list[MSubject] | MSubject, initial_task: int = 0, cfg: Config | None = None) -> None:
        self.cfg = cfg if cfg is not None else self.Config()
        self.tasks = tasks if isinstance(tasks, list) else [tasks]
        self.active_task = initial_task

        self.ls_id_list, self.project = self._create_project()

    def _create_project(self) -> tuple[list[int], Project]:
        res = None
        for p in s.ls.get_projects():
            if p.title == self.cfg.project_name:
                res = p
        if res is None:
            res = s.ls.create_project(
                title=self.cfg.project_name,
                label_config=self.cfg.labeling.xml,
                color=self.cfg.color,
            )

        ls_id_list = res.import_tasks([s.model_dump(exclude_none=True) for s in self.tasks])

        return ls_id_list, res

    def get_url(self) -> str:
        return f"{s.labelstudio['base_url']}/projects/{self.project.id}/data?task={self.ls_id_list[self.active_task]}"

    @staticmethod
    def cleanup_projects(for_prefix: str = "LSF_") -> None:
        for project in s.ls.get_projects():
            if project["title"].startswith(for_prefix):
                s.ls.delete_project(project["id"])
