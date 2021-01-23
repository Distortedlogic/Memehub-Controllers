import numpy as np
from controller.generated.models import RedditMeme, Redditor, db
from controller.utils.image_funcs import extract_img
from redisai import Client
from sqlalchemy import and_

rai = Client(host="redis", port="6379")


def get_memes(limit):
    return (
        db.session.query(RedditMeme)
        .filter(and_(RedditMeme.upvotes >= limit, RedditMeme.template == None))
        .order_by(RedditMeme.created_at.desc())
        .limit(limit)
        .all()
    )


def classify_stonks(limit=1000):
    while (memes := get_memes(limit)) :
        for batch in np.array_split(memes, len(memes) // 100 + 1):
            rai.tensorset(
                "images",
                np.array([extract_img(meme["url"]) for meme in batch]).astype(
                    np.float32
                ),
            )
            rai.modelrun("meme_features", ["images"], ["features"])
            for stonk in []:
                rai.modelrun("meme_features", ["features"], ["out"])
                for meme in np.where(rai.tensorget("out") == 1, batch):
                    if not meme.template:
                        meme.template = stonk
                    else:
                        meme.template += "," + stonk
            for meme in batch:
                if not meme.template:
                    meme.template = "not_found"
            db.session.commit()

