import arrow
import cv2 as cv2
import keras
import numpy as np
import onnx
import pandas as pd
import pytesseract
import redis
import redisai as rai
import requests
import tensorflow as tf
from billiard import Pool, cpu_count
from numpy import ndarray
from tqdm import tqdm

from controller.constants import FULL_SUB_LIST, THE_BEGINNING
from controller.extensions import db
from controller.reddit.functions.database import get_subs_to_scrape, redditmeme_max_ts
from controller.reddit.functions.misc import round_hour_down
from controller.reddit.functions.praw_mp import initializer, praw_by_id
from controller.reddit.functions.pushshift import query_pushshift
from controller.reddit.schema import RedditMeme, Redditor
from controller.utils.image_text import image_text


class RedditController:
    def __init__(self, verbose: bool = False, device: str = "cpu"):
        self.verbose = verbose
        self.full = False
        self.rai = rai.Client(host="redis", port="6379")

    def clear_features(self):
        meme_q = db.session.query(RedditMeme).filter(RedditMeme.features != None)
        all_memes = meme_q.all()
        for meme in all_memes:
            meme.features = None
        db.session.commit()

    def get_features(self, raw_memes):
        result = tqdm(image_text(meme["url"]) for meme in raw_memes)
        memes = []
        images = []
        for meme, (image, text) in zip(raw_memes, result):
            if image.any():
                meme["meme_text"] = text
                memes.append(meme)
                images.append(image)
            else:
                meme["meme_text"] = "NOT FOUND"
                meme["features"] = [-1]
        feature_list = []
        print("len(images)", len(images))
        for chunk in np.array_split(
            np.array(images), (len(images) // 100) if len(images) > 150 else 1
        ):
            chunk = np.array(chunk).astype(np.float32)
            self.rai.tensorset("images", chunk)
            self.rai.modelrun("vgg16", ["images"], ["out"])
            feature_list.append(self.rai.tensorget("out"))
        for meme, features in zip(memes, feature_list):
            meme["features"] = features.flatten().tolist()

        return raw_memes

    def stream(self, subreddit: str, start_time: int, end_time: int):
        for id_iter in query_pushshift(subreddit, start_time, end_time):
            with Pool(cpu_count(), initializer) as workers:
                if self.verbose:
                    yield list(tqdm(workers.imap_unordered(praw_by_id, id_iter)))
                else:
                    yield list(workers.imap_unordered(praw_by_id, id_iter))

    def engine(self, sub: str, max_ts: int) -> None:
        for raw_memes in self.stream(sub, max_ts, self.now):
            try:
                max_ts = max(
                    max_ts, max(item["timestamp"] for item in raw_memes if item)
                )
            except:
                print("empty data from reddit stream")
                return
            raw_memes = [
                meme for meme in raw_memes if meme and meme["username"] != "None"
            ]
            if not self.full:
                raw_memes = self.get_features(raw_memes)
            for meme in raw_memes:
                try:
                    redditor = (
                        db.session.query(Redditor)
                        .filter_by(username=meme["username"])
                        .one()
                    )
                except:
                    redditor = Redditor(username=meme["username"])
                    db.session.add(redditor)
                    db.session.commit()
                db.session.add(
                    RedditMeme(**meme, subreddit=sub, redditor_id=redditor.id)
                )
            db.session.commit()

    def update(self, full: bool = False) -> None:
        self.full = full
        self.now = round_hour_down(arrow.utcnow().timestamp)
        subs = FULL_SUB_LIST if full else get_subs_to_scrape()
        for sub in subs:
            if self.verbose:
                print(sub)
            max_ts = redditmeme_max_ts(sub)
            if not max_ts:
                max_ts = THE_BEGINNING
            self.engine(sub, max_ts)
