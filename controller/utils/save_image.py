from uuid import uuid4

import cv2
import numpy as np
import requests

from controller.constants import IMG_HEIGHT, IMG_WIDTH


def save_img(url, path, filename=None):
    try:
        resp = requests.get(url, timeout=5).content
        image = np.asarray(bytearray(resp), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        assert image.shape == (224, 224, 3)
        if not filename:
            path = path + str(uuid4()) + ".jpg"
        else:
            path = path + filename + ".jpg"
        cv2.imwrite(path, image)
        return True
    except Exception as e:
        print(e)
        return False
