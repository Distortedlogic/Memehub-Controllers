from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
from PIL.Image import Image as Img
from src.constants import IMG_HEIGHT, IMG_WIDTH
from torch import Tensor
from torchvision import transforms

transformations: Callable[..., Tensor] = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
deleted = np.array(Image.open("src/data/delete.png"))


class isDeletedException(Exception):
    pass


def load_tensor_from_url(url: str, is_deleted: bool) -> Tensor:
    raw = requests.get(url, stream=True).raw
    try:
        image = Image.open(raw).resize((224, 224)).convert("RGB")
    except UnidentifiedImageError as e:
        raise e
    if is_deleted and isDeleted(image):
        raise isDeletedException
    return transformations(image)


def load_img_from_url(url: str, is_deleted: bool) -> Img:
    raw = requests.get(url, stream=True).raw
    try:
        image = Image.open(raw).resize((224, 224)).convert("RGB")
    except UnidentifiedImageError as e:
        print("url", url)
        print(Image.open(raw))
        raise e
    if is_deleted and isDeleted(image):
        raise isDeletedException
    return image
    # arr = np.array(image, dtype=np.float32)
    # return np.moveaxis(arr, 2, 0)


def isDeleted(image: Img) -> bool:
    # _ = download_img_from_url(
    #     "https://i.redd.it/v9kwwkec3kn61.png", "src/data/delete.png"
    # )
    try:
        diff = np.array(image) - deleted
        isDeleted = np.abs(diff).sum()
        return not bool(isDeleted)
    except Exception as e:
        print("image", image)
        print("deleted", deleted)
        raise e


def download_img_from_url(
    url: str,
    path: str,
    size: Tuple[int, int] = (IMG_HEIGHT, IMG_WIDTH),
    verbose: bool = False,
) -> bool:
    try:
        if Path(path).is_file():
            return True
        # resp = requests.get(url, timeout=5).content
        # arr = np.array(bytearray(resp), dtype=np.uint8)
        # raw_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        # # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # resized_image = cv2.resize(raw_image, size)
        raw = requests.get(url, stream=True).raw
        resized_image = Image.open(raw).resize((224, 224)).convert("RGB")
        assert np.array(resized_image).shape == (224, 224, 3)
        if not np.array(resized_image).any():
            raise Exception("no image")
        resized_image.save(path)
        return True
    except Exception as e:
        if verbose:
            print(e)
        return False

