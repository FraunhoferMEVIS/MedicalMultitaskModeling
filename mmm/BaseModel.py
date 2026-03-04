"""
Pydantic ignores extra fields by default. This has caused silent bugs in the past.
"""

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict


class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(extra="forbid")
