import json
from time import time
from typing import Any, Iterator, List, Tuple, cast

import boto3
import numpy as np
import pandas as pd
from controller.constants import STATIC_PATH, VERSION
from controller.generated.models import RedditMeme
from controller.session import site_db
from controller.utils.display_df import display_df
from controller.utils.model_func import load_img_from_url
from controller.utils.secondToText import secondsToText
from decouple import config
from IPython.core.display import clear_output
from IPython.display import display
from PIL import Image
from redisai import Client
from sqlalchemy import and_, func
from sqlalchemy.sql.elements import ClauseElement
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torchvision import transforms

s3 = boto3.resource(
    "s3", aws_access_key_id=config("AWS_ID"), aws_secret_access_key=config("AWS_KEY")
)
bucket = s3.Bucket("memehub")


def display_meme(meme: RedditMeme):
    try:
        image = transforms.ToPILImage()(load_img_from_url(meme.url))
    except:
        site_db.delete(meme)
        return
    clear_output()
    print(
        f"""num correct - {site_db.query(RedditMeme).filter(and_(
                cast(ClauseElement,RedditMeme.version == VERSION),
                cast(ClauseElement,RedditMeme.stonk_correct == True),
            )).count()}"""
    )
    print(
        f"""num wrong - {site_db.query(RedditMeme).filter(and_(
                cast(ClauseElement,RedditMeme.version == VERSION),
                cast(ClauseElement,RedditMeme.stonk_correct == False),
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
        _ = display(image, template)
    except:
        _ = display(image)


def update_meme(meme: RedditMeme, keypress: str):
    if "nn" in keypress:
        meme.meme_clf_correct = False  # type: ignore
        meme.stonk_correct = False  # type: ignore
        meme.is_a_template = True  # type: ignore
    elif "n" in keypress:
        meme.meme_clf_correct = False  # type: ignore
        meme.stonk_correct = False  # type: ignore
        meme.is_a_template = False  # type: ignore
    elif "y" in keypress:
        meme.stonk = True  # type: ignore
        meme.stonk_correct = True  # type: ignore
        meme.meme_clf_correct = True  # type: ignore
        meme.is_a_template = True  # type: ignore


def handle_keypress_audit(meme: RedditMeme, prev_ids: List[int]):
    if (keypress := input("hit enter for next image -- q to break -- k to kill")) == "":
        meme.meme_clf_correct = True  # type: ignore
        meme.stonk_correct = True  # type: ignore
        meme.is_a_template = True  # type: ignore
    else:
        update_meme(meme, keypress)
        if "gb" in keypress:
            update_meme(meme, keypress)
            if (
                prev_meme := site_db.query(RedditMeme)
                .filter_by(id=prev_ids[-1])
                .first()
            ) :
                _ = display(transforms.ToPILImage()(load_img_from_url(prev_meme.url)))
                _ = handle_keypress_audit(prev_meme, prev_ids[:-1])
        elif keypress == "q":
            return True
    site_db.commit()


def audit_reddit_stonks():
    with open(STATIC_PATH, "rb") as f:
        static = json.load(f)
    folder_count_df = pd.DataFrame(
        list(static["folder_count"].items()), columns=["name", "count"]
    )
    prev_ids: List[int] = []
    for name in cast(List[str], folder_count_df.sort_values("count")["name"]):
        print(name)
        for meme in site_db.query(RedditMeme).filter(
            and_(
                cast(ClauseElement, RedditMeme.meme_clf == name),
                cast(ClauseElement, RedditMeme.stonk == True),
                cast(ClauseElement, RedditMeme.version == VERSION),
                cast(ClauseElement, RedditMeme.stonk_correct == None),
            )
        ):
            display_meme(meme)
            if handle_keypress_audit(meme, prev_ids):
                break
            prev_ids.append(meme.id)


def clear_reddit_stonks():
    for meme in site_db.query(RedditMeme).filter(
        cast(ClauseElement, RedditMeme.version != None)
    ):
        meme.version = None  # type: ignore
        meme.meme_clf = None  # type: ignore
        meme.meme_clf_correct = None  # type: ignore
        meme.stonk = None  # type: ignore
        meme.stonk_correct = None  # type: ignore
        meme.is_template = None  # type: ignore
    site_db.commit()


class RedditSet(IterableDataset[Tensor]):
    def __init__(self):
        pass

    def __iter__(self):
        for meme in (
            site_db.query(RedditMeme)
            .filter(cast(ClauseElement, RedditMeme.version == None))
            .order_by(cast(ClauseElement, RedditMeme.created_at.desc()))
        ):
            try:
                image = load_img_from_url(meme.url)
                yield (image, meme.id)
            except:
                site_db.delete(meme)

    def count(self):
        return (
            site_db.query(RedditMeme)
            .filter(cast(ClauseElement, RedditMeme.version == None))
            .order_by(func.random())
        ).count()


def reddit_stonks():
    batch_size = 128
    rai = Client(host="redis", port=6379)
    dataset = RedditSet()
    with open(STATIC_PATH, "rb") as f:
        static = json.load(f)
    start = time()
    round_start_time = time()
    for round, (images, ids) in enumerate(
        cast(
            Iterator[Tuple[Tensor, Tensor]],
            DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=1,
                collate_fn=cast(Any, None),
            ),
        )
    ):
        _ = rai.tensorset("images", np.array(images).astype(np.float32))
        _ = rai.modelrun("features", "images", "features_out")
        _ = rai.modelrun("dense", "features_out", "dense_out")
        names: List[str] = [
            static["num_name"][str(cast(int, np.argmax(arr)))]
            for arr in cast(List[Tensor], rai.tensorget("dense_out"))
        ]
        for name in list(set(names)):
            _ = rai.modelrun(name, "features_out", f"{name}_out")
        for idx, (id, name) in cast(
            Iterator[Tuple[int, Tuple[int, str]]],
            enumerate(zip(ids.numpy().astype(int).tolist(), names)),
        ):
            if (meme := site_db.query(RedditMeme).filter_by(id=id).first()) :
                is_stonk = cast(
                    bool, np.round(rai.tensorget(f"{name}_out")[idx]).astype(bool)
                )
                meme.meme_clf = name  # type: ignore
                meme.stonk = is_stonk  # type: ignore
                meme.version = VERSION  # type: ignore
        site_db.commit()
        if round % 10 == 0:
            clear_output()
            memes_done = (
                site_db.query(RedditMeme)
                .filter(cast(ClauseElement, RedditMeme.version != None))
                .count()
            )
            print(f"memes_done - {memes_done}")
            memes_found = (
                site_db.query(RedditMeme)
                .filter(cast(ClauseElement, RedditMeme.stonk == True))
                .count()
            )
            print(f"memes_found - {memes_found}")
            print(f"round - {round}")
            uptime = int(time() - start)
            count = dataset.count()
            print(f"uptime - {secondsToText(uptime)}")
            print(f"round_time - {secondsToText(int(time()-round_start_time)//10)}")
            round_start_time = time()
            print(f"num left - {count}")
            print(f"ETA - {secondsToText(uptime*count//((round+1)*batch_size))}")


def print_market():
    df = pd.read_sql(
        cast(
            str,
            site_db.query(
                RedditMeme.meme_clf.label("name"),
                func.count(RedditMeme.meme_clf).label("num_posts"),
                func.sum(RedditMeme.upvotes).label("total_upvotes"),
            )
            .filter(
                and_(
                    cast(ClauseElement, RedditMeme.stonk == True),
                    cast(ClauseElement, RedditMeme.version == VERSION),
                )
            )
            .group_by(RedditMeme.meme_clf)
            .order_by(func.count(RedditMeme.meme_clf).desc())
            .statement,
        ),
        site_db.bind,
    )
    df["upvotes_per_post"] = (
        df["total_upvotes"].divide(df["num_posts"]).apply(np.round).astype(int)
    )
    display_df(df)
    print(cast(List[str], df["name"].tolist()))

