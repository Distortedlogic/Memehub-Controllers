from pathlib import Path
from typing import Callable, Tuple

import cv2
import numpy as np
import requests
from PIL import Image
from src.constants import IMG_HEIGHT, IMG_WIDTH
from torch import Tensor
from torchvision import transforms

transformations: Callable[..., Tensor] = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_img_from_url(url: str) -> Tensor:
    raw = requests.get(url, stream=True).raw
    image = Image.open(raw).resize((224, 224)).convert("RGB")
    return transformations(image)
    # arr = np.array(image, dtype=np.float32)
    # return np.moveaxis(arr, 2, 0)


def download_img_from_url(
    url: str,
    path: str,
    size: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH),
    verbose: bool = False,
) -> bool:
    try:
        if Path(path).is_file():
            return True
        resp = requests.get(url, timeout=5).content
        arr = np.array(bytearray(resp), dtype=np.uint8)
        raw_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(raw_image, size)
        assert resized_image.shape == (224, 224, 3)
        if not resized_image.any():
            raise Exception("no image")
        cv2.imwrite(path, resized_image)
        return True
    except Exception as e:
        if verbose:
            print(e)
        return False
