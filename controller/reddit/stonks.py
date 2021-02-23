import json
from multiprocessing import cpu_count
from time import time

import boto3
import IPython
import numpy as np
import pandas as pd
import requests
from controller.constants import STATIC_PATH, VERSION
from controller.generated.models import RedditMeme, db
from controller.utils.model_func import load_img_from_url
from controller.utils.secondToText import secondsToText
from decouple import config
from IPython.core.display import clear_output
from IPython.display import display
from matplotlib.pyplot import imshow
from PIL import Image
from redisai import Client
from sqlalchemy import and_, func
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torchvision import transforms

# s3 = boto3.client(
#     "s3", aws_access_key_id=config("AWS_ID"), aws_secret_access_key=config("AWS_KEY"),
# )

s3 = boto3.resource(
    "s3", aws_access_key_id=config("AWS_ID"), aws_secret_access_key=config("AWS_KEY")
)
bucket = s3.Bucket("memehub")


class RedditSet(IterableDataset):
    def __init__(self):
        self.bad_images = 0
        self.q = (
            db.session.query(RedditMeme)
            .filter(RedditMeme.meme_clf == None)
            .order_by(RedditMeme.created_at.desc())
        )

    def __iter__(self):
        for meme in self.q:
            try:
                image = load_img_from_url(meme.url)
                yield (image, meme.id)
            except:
                self.bad_images += 1
                print("bad images", self.bad_images)
        # yield from ((load_img_from_url(meme.url), meme.id) for meme in self.q)

    def count(self):
        return self.q.count()


def audit_reddit_stonks():
    rai = Client(host="redis", port="6379")
    dataset = RedditSet()
    bad_templates = []
    with open(STATIC_PATH, "rb") as f:
        static = json.load(f)
    for round, (images, ids) in enumerate(
        DataLoader(dataset, batch_size=128, num_workers=0)  # cpu_count()
    ):
        rai.tensorset("images", np.array(images).astype(np.float32))
        rai.modelrun("features", "images", "features_out")
        rai.modelrun("dense", ["features_out"], ["dense_out"])
        names = [
            static["num_name"][str(np.argmax(arr))]
            for arr in rai.tensorget("dense_out")
        ]
        for name in list(set(names)):
            rai.modelrun(name, ["features_out"], [f"{name}_out"])
        for idx, (id, image, name) in enumerate(
            zip(ids.numpy().astype(int).tolist(), images, names)
        ):
            meme = db.session.query(RedditMeme).filter_by(id=id).first()
            is_stonk = np.round(rai.tensorget(f"{name}_out")[idx]).astype(bool)
            clear_output()
            print(name)
            print(np.round(is_stonk).astype(bool))
            print("Meme")
            display(transforms.ToPILImage()(image))
            try:
                object = bucket.Object("memehub/templates/" + name)
                response = object.get()
                file_stream = response["Body"]
                im = Image.open(file_stream)
                print("Template")
                display(im)
            except Exception as e:
                bad_templates.append(name)
            finally:
                pass
            print(bad_templates)
            if (
                keypress := input("hit enter for next image -- q to break -- k to kill")
            ) in "qk" and keypress:
                break


def reddit_stonks():
    rai = Client(host="redis", port="6379")
    dataset = RedditSet()
    bad_templates = []
    with open(STATIC_PATH, "rb") as f:
        static = json.load(f)
    for round, (images, ids) in enumerate(
        DataLoader(dataset, batch_size=128, num_workers=0)  # cpu_count()
    ):
        now = time()
        rai.tensorset("images", np.array(images).astype(np.float32))
        rai.modelrun("features", "images", "features_out")
        rai.modelrun("dense", ["features_out"], ["dense_out"])
        names = [
            static["num_name"][str(np.argmax(arr))]
            for arr in rai.tensorget("dense_out")
        ]
        for name in list(set(names)):
            rai.modelrun(name, ["features_out"], [f"{name}_out"])
        for idx, (id, name) in enumerate(zip(ids.numpy().astype(int).tolist(), names)):
            meme = db.session.query(RedditMeme).filter_by(id=id).first()
            is_stonk = np.round(rai.tensorget(f"{name}_out")[idx]).astype(bool)
            meme.meme_clf = name
            meme.stonk = is_stonk
            meme.version = VERSION
        db.session.commit()
        if round % 10 == 0:
            clear_output()
            memes_done = (
                db.session.query(RedditMeme).filter(RedditMeme.version != None).count()
            )
            print(f"memes_done - {memes_done}")
            memes_found = (
                db.session.query(RedditMeme).filter(RedditMeme.stonk == True).count()
            )
            print(f"memes_found - {memes_found}")
            print(f"round - {round}")
            round_time = int(time() - now)
            count = dataset.count()
            print(f"round time - {secondsToText(round_time)}")
            print(f"num left - {count}")
            print(f"ETA - {secondsToText(round_time*count)}")


def print_market():
    df = pd.read_sql(
        db.session.query(
            RedditMeme.meme_clf.label("name"),
            func.count(RedditMeme.meme_clf).label("num_posts"),
            func.sum(RedditMeme.upvotes).label("total_upvotes"),
        )
        .filter(RedditMeme.stonk == True)
        .group_by(RedditMeme.meme_clf)
        .order_by(func.count(RedditMeme.meme_clf).desc())
        .statement,
        db.session.bind,
    )
    df["upvotes_per_post"] = (
        df["total_upvotes"].divide(df["num_posts"]).apply(np.round).astype(int)
    )
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        IPython.display.display(df)
    print(df["name"].tolist())
