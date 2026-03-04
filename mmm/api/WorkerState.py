import os

import logfire

from .WorkerSettings import WorkerSettings


class WorkerState:
    def __init__(self) -> None:
        self.settings = WorkerSettings()
        self.rank = int(os.getenv("LOCAL_RANK", default=0))
        self._fm = None

    @property
    def fm(self):
        if self._fm is None:
            import torch

            torch.cuda.set_device(ws.rank)
            from mmm.api.M3Model import M3Model

            # Device identifier is cuda:0 on first celery worker, cuda:1 on second, etc.
            with logfire.span(
                "Worker loading model {modules_path}",
                rank=self.rank,
                modules_path=self.settings.modules_path,
                all_settings=self.settings,
            ) as span:
                self._fm = M3Model(self.settings.modules_path, device_identifier="cuda")
                span.set_attribute("loaded_tasks", self._fm.get_task_keys())
                span.set_attribute("loaded_shared_blocks", self._fm.get_sharedblock_keys())
        return self._fm


ws = WorkerState()
