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
from sqlalchemy.sql.elements import or_
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
            .filter(RedditMeme.version == None)
            .order_by(RedditMeme.created_at.desc())
        )

    def __iter__(self):
        for meme in self.q:
            try:
                image = load_img_from_url(meme.url)
                yield (image, meme.id)
            except:
                db.session.delete(meme)

    def count(self):
        return self.q.count()


def display_meme(meme):
    try:
        image = transforms.ToPILImage()(load_img_from_url(meme.url))
    except:
        db.session.delete(meme)
        return
    clear_output()
    print(
        f"""num correct - {db.session.query(RedditMeme).filter(and_(
                RedditMeme.version == VERSION,
                RedditMeme.stonk_correct == True,
            )).count()}"""
    )
    print(
        f"""num wrong - {db.session.query(RedditMeme).filter(and_(
                RedditMeme.version == VERSION,
                RedditMeme.stonk_correct == False,
            )).count()}"""
    )
    print(meme.meme_clf)
    print(meme.stonk)
    try:
        object = bucket.Object("memehub/templates/" + meme.meme_clf)
        response = object.get()
        file_stream = response["Body"]
        template = Image.open(file_stream)
        print("Meme/Template")
        display(image, template)
    except Exception as e:
        display(image)


def handle_keypress_audit(meme, prev_id):
    if (keypress := input("hit enter for next image -- q to break -- k to kill")) == "":
        meme.meme_clf_correct = True
        meme.stonk_correct = True
    elif keypress == "n":
        meme.meme_clf_correct = False
        meme.stonk_correct = False
    elif "gb" in keypress:
        if "n" in keypress:
            meme.meme_clf_correct = False
            meme.stonk_correct = False
        else:
            meme.meme_clf_correct = True
            meme.stonk_correct = True
        if prev_id:
            prev_meme = db.session.query(RedditMeme).filter_by(id=prev_id).first()
            display(transforms.ToPILImage()(load_img_from_url(prev_meme.url)))
            handle_keypress_audit(prev_meme, None)
    elif keypress == "q":
        return True
    db.session.commit()


def audit_reddit_stonks():
    prev_id = None
    with open(STATIC_PATH, "rb") as f:
        static = json.load(f)
    folder_count_df = pd.DataFrame(static["folder_count"], columns=["name", "count"])
    for name in folder_count_df.sort_values("count")["name"]:
        for meme in db.session.query(RedditMeme).filter(
            and_(
                RedditMeme.meme_clf == name,
                RedditMeme.stonk == True,
                RedditMeme.version == VERSION,
                RedditMeme.stonk_correct == None,
            )
        ):
            display_meme(meme)
            if handle_keypress_audit(meme, prev_id):
                break
            prev_id = meme.id


def handle_keypress_is_template(meme, prev_id):
    if (keypress := input("hit enter for next image -- q to break -- k to kill")) == "":
        meme.is_template = False
    elif keypress == "n":
        meme.is_template = True
    elif keypress == "y":
        meme.stonk = True
        meme.stonk_correct = True
        meme.meme_clf_correct = True
    elif "gb" in keypress:
        if "n" in keypress:
            meme.is_template = True
        else:
            meme.is_template = False
        if "y" in keypress:
            meme.stonk = True
            meme.stonk_correct = True
            meme.meme_clf_correct = True
        if prev_id:
            prev_meme = db.session.query(RedditMeme).filter_by(id=prev_id).first()
            display(transforms.ToPILImage()(load_img_from_url(prev_meme.url)))
            handle_keypress_is_template(prev_meme, None)
    elif keypress == "q":
        return True
    db.session.commit()


def audit_is_template():
    prev_id = None
    with open(STATIC_PATH, "rb") as f:
        static = json.load(f)
    folder_count_df = pd.DataFrame(
        static["folder_count"].items(), columns=["name", "count"]
    )
    for name in folder_count_df.sort_values("count")["name"]:
        for meme in db.session.query(RedditMeme).filter(
            and_(
                RedditMeme.meme_clf == name,
                RedditMeme.is_template == None,
                RedditMeme.stonk_correct == False
                # or_(RedditMeme.stonk == False, RedditMeme.stonk_correct == False),
            )
        ):
            display_meme(meme)
            if handle_keypress_is_template(meme, prev_id):
                break
            prev_id = meme.id


def reddit_stonks():
    batch_size = 128
    start = time()
    rai = Client(host="redis", port="6379")
    dataset = RedditSet()
    with open(STATIC_PATH, "rb") as f:
        static = json.load(f)
    for round, (images, ids) in enumerate(
        DataLoader(dataset, batch_size=batch_size, num_workers=0)  # cpu_count()
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
            uptime = int(time() - start)
            count = dataset.count()
            print(f"uptime - {secondsToText(uptime)}")
            print(f"num left - {count}")
            print(f"ETA - {secondsToText(uptime*count//((round+1)*batch_size))}")


def print_market():
    df = pd.read_sql(
        db.session.query(
            RedditMeme.meme_clf.label("name"),
            func.count(RedditMeme.meme_clf).label("num_posts"),
            func.sum(RedditMeme.upvotes).label("total_upvotes"),
        )
        .filter(and_(RedditMeme.stonk == True, RedditMeme.version == VERSION))
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
