import redis

from mmm.api.WorkerState import WorkerState
from mmm.BaseModel import BaseModel


class ApiFunction:
    class Args(BaseModel):
        pass

    class Results(BaseModel):
        pass

    @staticmethod
    def invoke(args: Args, ws: WorkerState, kv: redis.Redis) -> Results:
        raise NotImplementedError
