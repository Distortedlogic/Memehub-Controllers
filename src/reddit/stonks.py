from time import time
from typing import Any, Iterator, List, Tuple, Union, cast

import boto3
import numpy as np
import pandas as pd
from decouple import config
from IPython.core.display import clear_output
from redisai import Client
from sqlalchemy import and_, func
from sqlalchemy.sql.elements import ClauseElement
from src.constants import LOAD_VERSION
from src.generated.models import RedditMeme
from src.session import site_db
from src.utils.display import display_df, display_meme, display_template
from src.utils.image_funcs import load_img_from_url
from src.utils.model_func import get_static_names
from src.utils.secondToText import secondsToText
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

s3 = boto3.resource(
    "s3", aws_access_key_id=config("AWS_ID"), aws_secret_access_key=config("AWS_KEY")
)
bucket = s3.Bucket("memehub")


class Auditor:
    def __init__(self):
        pass


def print_stats():
    print(
        f"""num correct - {site_db.query(RedditMeme).filter(and_(
                cast(ClauseElement,RedditMeme.version == LOAD_VERSION),
                cast(ClauseElement,RedditMeme.stonk_correct == True),
            )).count()}"""
    )
    print(
        f"""num wrong - {site_db.query(RedditMeme).filter(and_(
                cast(ClauseElement,RedditMeme.version == LOAD_VERSION),
                cast(ClauseElement,RedditMeme.stonk_correct == False),
            )).count()}"""
    )


def update_meme(meme: RedditMeme, prev_ids: List[int]) -> Union[bool, None]:
    if (keypress := input("go back? or quit?")) == "y":
        if (prev_meme := site_db.query(RedditMeme).get(prev_ids[-1])) :
            _ = display_meme(prev_meme)
            return update_meme(prev_meme, prev_ids[:-1])
    elif keypress == "q":
        return True
    else:
        stonk_correct = not bool(input("stonk_correct?"))
        meme.stonk_correct = stonk_correct  # type: ignore
        if stonk_correct:
            if meme.stonk:
                meme.meme_clf_correct = True  # type: ignore
                meme.is_a_template = True  # type: ignore
            else:
                meme.meme_clf_correct = False  # type: ignore
                meme.is_a_template = not bool(input("is_a_template?"))  # type: ignore
        else:
            if meme.stonk:
                meme.meme_clf_correct = False  # type: ignore
                meme.is_a_template = not bool(input("is_a_template?"))  # type: ignore
            else:
                meme.meme_clf_correct = True  # type: ignore
                meme.is_a_template = True  # type: ignore
    site_db.commit()


def auditor(q: Any, prev_ids: List[int]):
    for meme in q:
        display_meme(meme)
        display_template(meme.meme_clf)
        if update_meme(meme, prev_ids):
            break
        prev_ids.append(meme.id)
    return prev_ids


def audit_reddit_stonks():
    folder_count_df = pd.DataFrame(
        list(get_static_names(LOAD_VERSION)["folder_count"].items()),
        columns=["name", "count"],
    )
    prev_ids: List[int] = []
    if (
        (
            keypress := input(
                """
    What do u want to audit?
    c - correct
    i - incorrect
    nmc - not_a_meme correct
    nmi - not_a_meme incorrect
    ntc - not_a_template correct
    nti - not_a_template incorrect
    """
            )
        )
        == "c"
    ):
        for name in cast(List[str], folder_count_df.sort_values("count")["name"]):
            print(name)
            prev_ids = auditor(
                site_db.query(RedditMeme).filter(
                    and_(
                        cast(ClauseElement, RedditMeme.meme_clf == name),
                        cast(ClauseElement, RedditMeme.stonk == True),
                        cast(ClauseElement, RedditMeme.version == LOAD_VERSION),
                        cast(ClauseElement, RedditMeme.stonk_correct == None),
                    )
                ),
                prev_ids,
            )
    elif keypress == "i":
        for name in cast(List[str], folder_count_df.sort_values("count")["name"]):
            print(name)
            prev_ids = auditor(
                site_db.query(RedditMeme).filter(
                    and_(
                        cast(ClauseElement, RedditMeme.meme_clf == name),
                        cast(ClauseElement, RedditMeme.stonk == False),
                        cast(ClauseElement, RedditMeme.version == LOAD_VERSION),
                        cast(ClauseElement, RedditMeme.stonk_correct == None),
                    )
                ),
                prev_ids,
            )
    elif "nm" in keypress:
        correct = "c" in keypress
        _ = auditor(
            site_db.query(RedditMeme).filter(
                and_(
                    cast(ClauseElement, RedditMeme.meme_clf == "not_a_meme"),
                    cast(ClauseElement, RedditMeme.stonk == correct),
                    cast(ClauseElement, RedditMeme.version == LOAD_VERSION),
                    cast(ClauseElement, RedditMeme.stonk_correct == None),
                )
            ),
            [],
        )
    elif "nt" in keypress:
        correct = "c" in keypress
        _ = auditor(
            site_db.query(RedditMeme).filter(
                and_(
                    cast(ClauseElement, RedditMeme.meme_clf == "not_a_template"),
                    cast(ClauseElement, RedditMeme.stonk == correct),
                    cast(ClauseElement, RedditMeme.version == LOAD_VERSION),
                    cast(ClauseElement, RedditMeme.stonk_correct == None),
                )
            ),
            [],
        )


def clear_reddit_stonks():
    for meme in site_db.query(RedditMeme).filter(
        cast(ClauseElement, RedditMeme.version != None)
    ):
        meme.version = None  # type: ignore
        meme.meme_clf = None  # type: ignore
        meme.meme_clf_correct = None  # type: ignore
        meme.stonk = None  # type: ignore
        meme.stonk_correct = None  # type: ignore
        meme.is_a_template = None  # type: ignore
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
    num_name = get_static_names(LOAD_VERSION)["num_name"]
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
            num_name[str(cast(int, np.argmax(arr)))]
            for arr in cast(List[Tensor], rai.tensorget("dense_out"))
        ]
        for name in list(set(names)):
            _ = rai.modelrun(name, "features_out", f"{name}_out")
        for idx, (id, name) in cast(
            Iterator[Tuple[int, Tuple[int, str]]],
            enumerate(zip(ids.numpy().astype(int).tolist(), names)),
        ):
            if (meme := site_db.query(RedditMeme).get(id)) :
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
                    cast(ClauseElement, RedditMeme.version == LOAD_VERSION),
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

