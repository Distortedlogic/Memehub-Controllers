import cv2
import numpy as np
import pytesseract
import requests
from controller.constants import IMG_HEIGHT, IMG_WIDTH


def extract_img(url, path: str = "", verbose=False, text=False):
    try:
        if path:
            image = cv2.imread(path)
            try:
                if image.any():
                    if text:
                        meme_text = pytesseract.image_to_string(image)
                        return image, meme_text
                    else:
                        return image
            except:
                pass
        resp = requests.get(url, timeout=5).content
        image = np.asarray(bytearray(resp), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if text:
            meme_text = pytesseract.image_to_string(image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        assert image.shape == (224, 224, 3)
        if not image.any():
            return np.array([])
        if path:
            cv2.imwrite(path, image)
        if text:
            return image, meme_text
        else:
            return image
    except Exception as e:
        if verbose:
            print(e)
        return np.array([])
