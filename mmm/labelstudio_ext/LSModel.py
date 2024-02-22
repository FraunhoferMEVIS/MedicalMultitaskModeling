"""
LS machine learning integration. Useful resources:

- Possible routes: https://github.com/HumanSignal/label-studio-ml-backend/blob/master/label_studio_ml/api.py

The instruction in your project can be as an example:

<style>
.ls-modal__content {
    width: auto;
    height: "800px";
}
</style>
This needs to be reachable from the local user machine:
<iframe src="http://localhost:8000/lgbm/settingsgui" width="100%" height="700px"></iframe>
"""

import json
import base64
import logging
import jinja2
from typing import Any, TypeVar, Generic, Literal
import numpy as np

from mmm.BaseModel import BaseModel
from .projects import LSProject
from .LabelstudioCredentials import LabelstudioCredentials

try:
    # API extras
    from label_studio_sdk import Client, Project
    from label_studio_converter.brush import (
        decode_rle,
        encode_rle,
        decode_from_annotation,
    )
    from fastapi import APIRouter, BackgroundTasks, Request
    from fastapi.responses import HTMLResponse
except ImportError:
    Client, Project = Any, Any
    APIRouter, BackgroundTasks, Request, HTMLResponse = Any, Any, Any, Any

# bound to be a pydantic model
T = TypeVar("T", bound=BaseModel)


class LabelStudioTask(BaseModel):
    data: dict
    predictions: list[dict] | None = None
    annotations: list[dict] | None = None
    meta: dict | None = None


class SerializedArray(BaseModel):
    """
    >>> import numpy as np
    >>> from mmm.labelstudio_ext.LSModel import SerializedArray
    >>> for dtype in [np.int8, np.int32, np.int64, np.float32, np.float64]:
    ...     testarr = np.random.rand(50, 52).astype(dtype)
    ...     seria_arr = SerializedArray.from_numpy(testarr)
    ...     assert np.array_equal(testarr, seria_arr.to_numpy())
    """

    data: str
    dtype: str
    shape: tuple[int, ...]
    meta: dict | None = None

    @classmethod
    def from_numpy(cls, arr: np.ndarray, meta: dict | None = None):
        return cls(
            data=base64.b64encode(arr).decode("utf-8"),
            dtype=str(arr.dtype),
            shape=arr.shape,
            meta=meta,
        )

    def to_numpy(self):
        return np.frombuffer(base64.b64decode(self.data), dtype=self.dtype).reshape(self.shape)


class LSModel(Generic[T]):
    """
    For predict to be called,
    select "Retrieve predictions when loading a task automatically" in the Machine learning settings.

    Train gets called by Labelstudio when the user clicks "Train" in the UI.
    Alternatively, enable the setting: "Start model training after any annotation are submitted or updated".
    """

    class Config(BaseModel):
        category: str = "unknownmodel"

    def __init__(self, cfg: T, ls_client: Client, prefix: str) -> None:
        self.cfg, self.ls_client, self.prefix = cfg, ls_client, prefix
        self.mtl_templates = jinja2.Environment(loader=jinja2.PackageLoader("mmm", "resources"))
        self.model_version = None

    async def get_user_helper(self) -> str:
        return f"Please configure model {self.cfg.category} hosted at {self.prefix}"

    async def get_user_html(self) -> str:
        return f"<h1>Model {self.cfg.category} hosted at {self.prefix}</h1>"

    async def settings_gui(self, request: Request) -> str:
        # Render the jsonform.jinja2 template
        return HTMLResponse(
            content=self.mtl_templates.get_template("jsonform.jinja2").render(
                {
                    "helper": await self.get_user_helper(),
                    "post_url": f"{self.prefix}/settings",
                    "data": (await self.get_current_settings()).model_dump_json(),
                    "schema": json.dumps(await self.get_settings_schema()),
                    "user_html": await self.get_user_html(),
                }
            ),
            status_code=200,
        )

    async def get_settings_schema(self) -> dict:
        return self.cfg.model_json_schema()

    async def get_current_settings(self) -> T:
        return self.cfg

    async def set_settings(self, body: dict) -> T:
        self.cfg = type(self.cfg)(**body)
        return self.cfg

    async def predict(self, project: LSProject, tasks: list[int], context: dict | None = None):
        """
        If this is called with a single task, we assume that the respective user just clicked on this task.

        https://labelstud.io/api#tag/Predictions/operation/api_predictions_create
        """
        logging.info(f"Predict not implemented, called for project {project} and {len(tasks)} tasks")
        return {}

    async def train(
        self,
        project: LSProject,
        tasks: list[dict],
        mode: Literal["retrain", "continue"] = "retrain",
    ):
        logging.info(f"Train not implemented, called for project {project} and {len(tasks)} tasks with mode {mode}")
        return {}

    async def get_version(self):
        if self.model_version is not None:
            return {"model_version": self.model_version}
        return {"model_version": "unknown"}

    async def get_versions(self):
        d = await self.get_version()
        return {"versions": [d["model_version"]]}

    async def _train_handler(self, request: Request):
        requestbody = await request.json()
        project_id: str = requestbody["project"].split(".")[0]
        ls_project: Project = self.ls_client.get_project(project_id)
        project: LSProject = LSProject(
            LSProject.Config(name=ls_project.title),
            self.ls_client,
            ls_project=ls_project,
        )
        return await self.train(
            project,
            requestbody["annotations"] if "annotations" in requestbody else [],
            mode="retrain",
        )

    async def _predict_handler(self, request: Request):
        req = await request.json()
        project_id = req["project"].split(".")[0]
        context = req["params"]["context"]
        if context is not None:
            logging.info(context)
        ls_project: Project = self.ls_client.get_project(project_id)
        project: LSProject = LSProject(
            LSProject.Config(name=ls_project.title),
            self.ls_client,
            ls_project=ls_project,
        )
        logging.info(f"Called predict with {len(req['tasks'])} tasks: {[t['id'] for t in req['tasks']]}")
        predictions = await self.predict(project, req["tasks"], context=context)
        return predictions

    async def _setup_handler(self, request: Request):
        req_body = await request.json()
        project_id: str = req_body["project"].split(".")[0]
        ls_project = self.ls_client.get_project(project_id)
        instruction = ls_project.params["expert_instruction"]
        self.model_version = req_body["model_version"]

        try:
            f_key, f_instr = await self.get_instruction(ls_project, instruction)

            if not f_key(instruction):
                logging.info(f"Adding instruction for {self.prefix}")
                ls_project.set_params(**{"expert_instruction": f_instr(ls_project, instruction)})
        except NotImplementedError:
            logging.info(f"Instruction for {self.prefix} not implemented")

        return await self.get_version()

    async def get_instruction(self, project: Project, previous_instruction: str) -> tuple:
        """
        First function is the key that is used to see if the instruction is already there.
        Second function builds the instruction itself.
        """
        raise NotImplementedError()

    async def webhook(self, request: Request, background_tasks: BackgroundTasks):
        body = await request.json()
        if body["action"] in ["ANNOTATION_UPDATED", "ANNOTATION_CREATED"]:
            logging.info(f"Annotation updated: {body['task']['id']}")
            project_id: str = body["project"]["id"]
            ls_project: Project = self.ls_client.get_project(project_id)
            project: LSProject = LSProject(
                LSProject.Config(name=ls_project.title),
                self.ls_client,
                ls_project=ls_project,
            )
            ls_task = body["task"]
            ls_task["annotations"] = [body["annotation"]]
            background_tasks.add_task(self.train, project, [ls_task], mode="continue")
        return {}

    async def health_handler(self):
        return {"status": "UP", "model_class": self.cfg.category}

    def get_tag(self):
        return f"Model {self.cfg.category}"

    async def invocation(self, ls_task: LabelStudioTask) -> SerializedArray:
        """
        Enables to perform inference without requiring a labelstudio project.
        """
        raise NotImplementedError

    async def labeling_priority(self, ls_task: LabelStudioTask) -> float:
        """
        Returns a high number for tasks that should be labeled.
        """
        raise NotImplementedError

    def build_router(self) -> APIRouter:
        router = APIRouter()
        tag = self.get_tag()
        # Required for labelstudio
        labelstudio_tag = f"Labelstudio {self.cfg.category}"
        router.post("/predict", tags=[labelstudio_tag])(self._predict_handler)
        router.post("/train", tags=[labelstudio_tag])(self._train_handler)
        router.get("/health", tags=[labelstudio_tag])(self.health_handler)
        router.post("/webhook", tags=[labelstudio_tag])(self.webhook)
        router.post("/setup", tags=[labelstudio_tag])(self._setup_handler)
        router.get("/versions", tags=[labelstudio_tag])(self.get_versions)

        # User interaction
        router.get("/settingsgui", tags=[labelstudio_tag])(self.settings_gui)
        router.get("/settingsschema", tags=[labelstudio_tag])(self.get_settings_schema)
        router.get("/settings", tags=[labelstudio_tag])(self.get_current_settings)
        router.post("/settings", tags=[labelstudio_tag])(self.set_settings)

        # Invocation without labelstudio
        router.post("/compute_score/", tags=[tag])(self.labeling_priority)
        router.post("/invocation", tags=[tag])(self.invocation)

        return router
