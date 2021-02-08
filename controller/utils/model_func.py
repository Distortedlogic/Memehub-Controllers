import numpy as np
import requests
from PIL import Image


def load_img_from_url(url):
    raw = requests.get(url, stream=True).raw
    image = Image.open(raw).resize((224, 224)).convert("RGB")
    arr = np.array(image, dtype=np.float32)
    return np.rollaxis(arr, 2, 0)
