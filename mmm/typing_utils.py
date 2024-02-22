import torch
import random
import colorsys
import cv2
import numpy as np
import base64


def get_colors(num_colors):
    colors = []
    for i in np.arange(0.0, 360.0, 360.0 / num_colors):
        hue = i / 360.0
        lightness = (50 + np.random.rand() * 10) / 100.0
        saturation = (90 + np.random.rand() * 10) / 100.0
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def rgbnumpy_to_base64(img: np.ndarray) -> str:
    """
    Expects a numpy array with shape (H, W, C) and converts it to a base64 string
    """
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    encoded_img = buffer.tobytes()
    base64_img_string = base64.b64encode(encoded_img).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_img_string}"
