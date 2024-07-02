import logging
from pydantic import Field
from typing import TYPE_CHECKING, Any
from deepdiff import DeepDiff


from mmm.BaseModel import BaseModel

try:
    from label_studio_sdk import Client, Project, parse_config
except ImportError:
    if not TYPE_CHECKING:
        Client, Project = Any, Any


class LSProject:
    """
    Creating data happens by using LSProject.ls.import_tasks.
    """

    class Config(BaseModel):
        name: str = Field(
            default="uniqueprojectname",
            description="Unique identifier of the project w.r.t. this labelstudio instance",
        )

    def __init__(
        self,
        args: Config,
        ls_client: Client,
        template: str = "",
        ls_project: Project | None = None,
    ):
        self.args, self.client = args, ls_client
        self.template = template
        # If project already exists, load it
        self.ls: Project = self.get_project() if ls_project is None else ls_project
        if self.ls is None:
            self.ls = self._create_new()

    @staticmethod
    def exists(client, project_name: str) -> bool:
        return any(p.title == project_name for p in client.get_projects())

    def get_project(self) -> Project | None:
        for p in self.client.get_projects():
            if p.title == self.args.name:
                if not self.template:
                    self.template = p.get_params()["label_config"]
                else:
                    if differences := DeepDiff(
                        parse_config(self.template),
                        p.get_params()["parsed_label_config"],
                    ):
                        logging.warning(f"Found differences between expected and actual template: {differences}")
                return p
        return None

    def _create_new(self):
        assert self.template, "Template must be set to create a new project"
        p = self.client.start_project(
            title=f"{self.args.name}",
            label_config=self.template,
        )
        return p

    async def delele_all_predictions(self) -> list[int]:
        prediction_ids = [
            p["id"] for p in self.client.make_request("GET", f"/api/predictions?project={self.ls.id}").json()
        ]
        for pred_id in prediction_ids:
            self.client.make_request("DELETE", f"/api/predictions/{pred_id}")
        return prediction_ids

    def __repr__(self):
        return f"LSProject(name={self.args.name}, id={self.ls.id})"
