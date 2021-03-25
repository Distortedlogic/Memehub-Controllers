from datetime import date
from typing import cast

import numpy as np
from cv2 import cv2


def round_hour(ts: int):
    q, r = divmod(ts, 3600)
    return (q + 1 if r >= 1800 else q) * 3600


def round_hour_down(ts: int):
    q, _ = divmod(ts, 3600)
    return q * 3600


def dump_datetime(value: date):
    """Deserialize datetime object into string form for JSON processing."""
    if value is None:
        return None
    return [value.strftime("%Y-%m-%d"), value.strftime("%H:%M:%S")]


def isDeleted(image_cv: np.ndarray) -> bool:
    deleted = cv2.imread("assets/deleted_img/image404.png")
    deleted_nb = cv2.imread("assets/deleted_img/image404_nb.png")
    try:
        diff = cv2.subtract(image_cv, deleted)
    except Exception:
        diff = True
    try:
        diff_nb = cv2.subtract(image_cv, deleted_nb)
    except Exception:
        diff_nb = True
    return cast(bool, np.all(diff == 0) | np.all(diff_nb == 0))
