from multiprocessing import cpu_count

import numpy as np
from controller.generated.models import RedditMeme, db
from PIL import Image
from redisai import Client
from sqlalchemy import and_
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

rai = Client(host="redis", port="6379")


class RedditSet(IterableDataset):
    def __init__(self, limit):
        self.limit = limit

    def __iter__(self):
        yield from (
            (Image.open(meme["url"]), meme)
            for meme in db.session.query(RedditMeme)
            .filter(and_(RedditMeme.upvotes >= self.limit, RedditMeme.template == None))
            .order_by(RedditMeme.created_at.desc())
        )


def reddit_stonks(limit=1000):
    for image_memes in DataLoader(
        RedditSet(limit), batch_size=64, num_workers=cpu_count(), shuffle=True
    ):
        images, memes = list(map(list, zip(*image_memes)))
        rai.tensorset("images", np.array(images).astype(np.float32))
        rai.modelrun("meme_features", ["images"], ["features"])
        for stonk in []:
            rai.modelrun(stonk, ["features"], ["out"])
            for meme in np.where(rai.tensorget("out") == 1, memes):
                if not meme.template:
                    meme.template = stonk
                else:
                    meme.template += "," + stonk
        for meme in memes:
            if not meme.template:
                meme.template = "not_found"
        db.session.commit()

