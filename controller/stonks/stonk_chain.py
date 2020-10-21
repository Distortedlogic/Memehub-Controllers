from controller.constants import MODELS_REPO
from itertools import compress

import numpy as np
import redisai as rai
from sqlalchemy import and_
from sqlalchemy.orm import load_only

from os import listdir
from os.path import splitext

from controller.extensions import db
from controller.reddit.schema import RedditMeme
from controller.stonks.schema import Template


class StonkChain:
    def __init__(self):
        self.rai = rai.Client(host="redis", port="6379")
        self.names = [
            item
            for item in set(splitext(file)[0] for file in listdir(MODELS_REPO))
            if item != "vgg16"
        ]

    def predict(self, name):
        self.rai.modelrun(name, "features", "out")
        return self.rai.tensorget("out")

    def fetch(self, size=100):
        while True:
            data = (
                db.session.query(RedditMeme)
                .options(load_only("template", "features", "url"))
                .filter(and_(RedditMeme.template == None, RedditMeme.features != None))
                .limit(size)
                .all()
            )
            print(data)
            if not data:
                break
            yield data

    def run(self):
        for memes in self.fetch():
            self.rai.tensorset(
                "features",
                np.array([meme.features for meme in memes]).astype(np.float32),
            )
            for name in self.names:
                print(name, np.around(self.predict(name)))
            #     for idx, meme in enumerate(compress(memes, self.predict(name))):
            #         print(f'found {idx + 1} for {name}')
            #         if meme.template:
            #             print('dup found', meme.url, meme.template, name)
            #         meme.template = name
            # for meme in memes:
            #     if not meme.template:
            #         meme.template = 'NOT FOUND'
            # db.session.commit()
