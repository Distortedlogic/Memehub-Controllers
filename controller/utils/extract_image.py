from controller.constants import IMG_HEIGHT, IMG_WIDTH
import requests
import numpy as np
import cv2


def extract_img(url):
    try:
        resp = requests.get(url, timeout=5).content
        image = np.asarray(bytearray(resp), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        assert image.shape == (224, 224, 3)
        return image
    except Exception as e:
        # print(e)
        return np.array([])
