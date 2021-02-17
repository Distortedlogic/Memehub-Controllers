import json
from multiprocessing import cpu_count

import numpy as np
from constants import STATIC_PATH, VERSION
from controller.generated.models import RedditMeme, db
from PIL import Image
from redisai import Client
from sqlalchemy import and_
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from utils.model_func import load_img_from_url


class RedditSet(IterableDataset):
    def __init__(self, limit):
        self.limit = limit

    def __iter__(self):
        yield from (
            (load_img_from_url(meme["url"]), meme)
            for meme in db.session.query(RedditMeme)
            .filter(and_(RedditMeme.upvotes >= self.limit, RedditMeme.template == None))
            .order_by(RedditMeme.created_at.desc())
        )


def reddit_stonks(limit=1000):
    rai = Client(host="redis", port="6379")
    with open(STATIC_PATH, "rb") as f:
        static = json.load(f)
    for image_memes in DataLoader(
        RedditSet(limit), batch_size=64, num_workers=cpu_count(), shuffle=True
    ):
        images, memes = list(map(list, zip(*image_memes)))
        rai.tensorset("images", np.array(images).astype(np.float32))
        rai.modelrun("features", ["images"], ["features"])
        rai.modelrun("dense", ["features"], ["dense_out"])
        names = (static["num_name"][str(num)] for num in rai.tensorget("dense_out"))
        for name in list(set(names)):
            rai.modelrun(name, ["features"], [f"{name}_out"])
        for idx, meme, name in enumerate(zip(memes, names)):
            if rai.tensorget(f"{name}_out")[idx]:
                meme.template = name
            else:
                meme.template = "None"
            meme.version = VERSION
        db.session.commit()

