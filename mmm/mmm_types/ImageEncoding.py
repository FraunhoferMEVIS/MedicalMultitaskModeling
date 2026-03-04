from enum import Enum


class ImageEncoding(str, Enum):
    base64 = "base64"
    abs_filepath = "abs_filepath"
