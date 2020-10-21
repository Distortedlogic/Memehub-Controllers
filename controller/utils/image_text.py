import cv2
import numpy as np
import pytesseract
import requests

from controller.constants import IMG_HEIGHT, IMG_WIDTH


def image_text(url):
    try:
        resp = requests.get(url, timeout=5).content
        image = np.asarray(bytearray(resp), dtype=np.uint8)

        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        meme_text = pytesseract.image_to_string(image)
        image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        return image, meme_text
    except Exception as e:
        print(e)
        print("image error")
        return np.array([]), None
