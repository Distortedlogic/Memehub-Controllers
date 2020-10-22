import numpy as np
import redisai
from controller.generated.models import RedditMeme, db
from controller.utils.image_text import image_text
from sqlalchemy import and_
from tqdm import tqdm


def build_features(self):
    rai = redisai.Client(host="redis", port="6379")
    size = 100
    while True:
        meme_q = (
            db.session.query(RedditMeme)
            .filter(and_(RedditMeme.features != None, RedditMeme.features != [-1]))
            .limit(size)
        )
        all_memes = meme_q.all()
        if not all_memes:
            break
        result = tqdm(
            (image_text(meme.url) for meme in all_memes), total=len(all_memes)
        )
        memes = []
        images = []
        for meme, (image, text) in zip(all_memes, result):
            if image.any():
                meme.meme_text = text
                memes.append(meme)
                images.append(image)
            else:
                meme.meme_text = "NOT FOUND"
                meme.features = [-1]
        images = np.array(images).astype(np.float32)
        rai.tensorset("images", images)
        rai.modelrun("vgg16", ["images"], ["out"])
        for meme, features in zip(memes, rai.tensorget("out")):
            meme.features = features.flatten().tolist()
        db.session.commit()
