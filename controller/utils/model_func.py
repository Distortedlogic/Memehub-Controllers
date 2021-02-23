import numpy as np
import requests
from PIL import Image
from torchvision import transforms

transformations = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_img_from_url(url):
    raw = requests.get(url, stream=True).raw
    image = Image.open(raw).resize((224, 224)).convert("RGB")
    return transformations(image)
    # arr = np.array(image, dtype=np.float32)
    # return np.moveaxis(arr, 2, 0)
