import json

import numpy as np
import redisai
from controller.constants import STATIC_PATH, VERSION
from controller.generated.models import Meme, db
from controller.utils.model_func import load_img_from_url
from sqlalchemy import or_


def update():
    rai = redisai.Client(host="redis", port="6379")
    with open(STATIC_PATH, "rb") as f:
        static = json.load(f)
    while memes := (
        site_db.query(Meme)
        .filter(or_(Meme.stonk == None, Meme.version != VERSION))
        .order_by(Meme.createdAt.desc())
        .limit(100)
        .all()
    ):
        images = [load_img_from_url(meme.url) for meme in memes]
        rai.tensorset("images", np.array(images, dtype=np.float32))
        rai.modelrun("MemeClf", ["images"], ["out"])
        names = (static["num_name"][str(num)] for num in rai.tensorget("out"))
        for meme, name in zip(memes, names):
            meme.stonk = name
            meme.version = VERSION
        print(pred)
        break
