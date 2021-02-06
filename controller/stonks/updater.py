import json

import numpy as np
import redisai
import requests
from controller.constants import STATIC_PATH, VERSION
from controller.generated.models import Meme, db
from PIL import Image
from sqlalchemy import or_


def update():
    rai = redisai.Client(host="redis", port="6379")
    with open(STATIC_PATH, "rb") as f:
        static = json.load(f)
    while memes := (
        db.session.query(Meme)
        .filter(or_(Meme.stonk == None, Meme.version != VERSION))
        .order_by(Meme.createdAt.desc())
        .limit(100)
        .all()
    ):
        images = []
        for meme in memes:
            raw = requests.get(meme.url, stream=True).raw
            image = Image.open(raw).resize((224, 224)).convert("RGB")
            arr = np.array(image, dtype=np.float32)
            rolled_channels = np.rollaxis(arr, 2, 0)
            images.append(rolled_channels)
        images = np.array(images, dtype=np.float32)
        rai.tensorset("images", images)
        rai.modelrun("MemeClf", ["images"], ["out"])
        pred = [static["num_name"][str(num)] for num in rai.tensorget("out")]
        print(pred)
        break
