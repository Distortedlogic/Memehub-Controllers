from time import time
from typing import Any, Iterator, List, Tuple, cast

import arrow
import numpy as np
import pandas as pd
from arrow.arrow import Arrow
from IPython.core.display import clear_output
from PIL import UnidentifiedImageError
from redisai import Client
from sqlalchemy import and_, func
from sqlalchemy.sql.elements import ClauseElement, or_
from src.constants import DONT_USE_TEMPLATES, LOAD_MEME_CLF_VERSION
from src.generated.models import Meme, RedditMeme
from src.session import site_db
from src.stonks.utils.evaluate import evaluated, unevaluated
from src.utils.display import display_df, display_template, pretty_print_dict
from src.utils.image_funcs import isDeletedException, load_tensor_from_url
from src.utils.model_func import get_static_names
from src.utils.secondToText import secondsToText
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset


class MemeSet(IterableDataset[Tensor]):
    def __init__(self, entity: Any, eval_mode: bool, is_celery: bool):
        self.entity = entity
        self.num_deleted = 0
        self.num_bad_images = 0
        if eval_mode:
            self.clause = and_(
                or_(
                    cast(ClauseElement, self.entity.version == None),
                    cast(ClauseElement, self.entity.version != LOAD_MEME_CLF_VERSION),
                ),
                cast(ClauseElement, self.entity.is_a_template_official != None),
            )
        else:
            if is_celery:
                self.clause = and_(
                    cast(
                        ClauseElement,
                        self.entity.createdAt
                        >= cast(Arrow, arrow.utcnow())
                        .replace(minute=0, second=0, millisecond=0)
                        .shift(days=-2),
                    ),
                    or_(
                        cast(ClauseElement, self.entity.version == None),
                        cast(
                            ClauseElement, self.entity.version != LOAD_MEME_CLF_VERSION
                        ),
                    ),
                )
            else:
                self.clause = or_(
                    cast(ClauseElement, self.entity.version == None),
                    cast(ClauseElement, self.entity.version != LOAD_MEME_CLF_VERSION),
                )

    def __iter__(self):
        for meme in (
            site_db.query(self.entity)
            .filter(self.clause)
            .order_by(cast(ClauseElement, self.entity.created_at.desc()))
        ):
            try:
                image = load_tensor_from_url(meme.url, is_deleted=True)
                yield (image, meme.id)
            except UnidentifiedImageError:
                self.num_bad_images += 1
            except isDeletedException:
                self.num_deleted += 1
                # site_db.delete(meme)
                # site_db.commit()

    def count(self):
        return (
            site_db.query(self.entity).filter(
                cast(ClauseElement, self.entity.version == None)
            )
        ).count()


class StonkMarket:
    def __init__(self, is_celery: bool = False):
        self.rai = Client(host="redis", port=6379)
        self.batch_size = 128
        self.is_celery = is_celery
        self.celery_tag = "_celery" if is_celery else ""

    def reddit_engine(self):
        self.entity = RedditMeme
        self.eval_mode = False
        self.engine()

    def site_engine(self):
        self.entity = Meme
        self.eval_mode = False
        self.engine()

    def evaluate(self):
        self.entity = RedditMeme
        self.eval_mode = True
        self.engine()

    def update_meme(self, idx_id_name: Tuple[int, Tuple[int, str]]):
        idx, (id, name) = idx_id_name
        if (meme := site_db.query(self.entity).get(id)) :
            is_stonk = bool(
                round(
                    cast(
                        List[int], self.rai.tensorget(f"{name}_out" + self.celery_tag)
                    )[idx]
                )
            )
            meme.meme_clf = name  # type: ignore
            meme.stonk = is_stonk  # type: ignore
            meme.version = LOAD_MEME_CLF_VERSION  # type: ignore
        site_db.commit()

    def engine(self):
        if input("Fresh?"):
            self.clear()
        self.dataset = MemeSet(self.entity, self.eval_mode, self.is_celery)
        num_name = get_static_names(LOAD_MEME_CLF_VERSION)["num_name"]
        self.start = time()
        self.now = time()
        for self.iteration, (images, ids) in enumerate(
            cast(
                Iterator[Tuple[Tensor, Tensor]],
                DataLoader(
                    self.dataset,
                    batch_size=self.batch_size,
                    num_workers=1,
                    collate_fn=cast(Any, None),
                ),
            )
        ):
            _ = self.rai.tensorset(
                "images" + self.celery_tag, np.array(images).astype(np.float32)
            )
            _ = self.rai.modelrun(
                "features", "images" + self.celery_tag, "features_out" + self.celery_tag
            )
            _ = self.rai.modelrun(
                "dense", "features_out" + self.celery_tag, "dense_out" + self.celery_tag
            )
            names: List[str] = [
                num_name[str(cast(int, np.argmax(arr)))]
                for arr in cast(
                    List[Tensor], self.rai.tensorget("dense_out" + self.celery_tag)
                )
            ]
            for name in set(names):
                _ = self.rai.modelrun(
                    name,
                    "features_out" + self.celery_tag,
                    f"{name}_out" + self.celery_tag,
                )
            for item in cast(
                Iterator[Tuple[int, Tuple[int, str]]],
                enumerate(zip(ids.numpy().astype(int).tolist(), names)),
            ):
                self.update_meme(item)
            if self.iteration % 10 == 0:
                self.print_stats()
        self.print_stats()

    def print_stats(self):
        clear_output()
        self.count = self.dataset.count()
        if self.eval_mode:
            evaluated()
        else:
            unevaluated(
                dict(
                    bad_images=self.dataset.num_bad_images,
                    is_deleted=self.dataset.num_deleted,
                    num_left=self.count,
                    iteration=self.iteration,
                )
            )
            self.time_stats()

    def time_stats(self):
        uptime = int(time() - self.start)
        pretty_print_dict(
            dict(
                round_time=secondsToText(int(time() - self.now) // 10),
                uptime=secondsToText(uptime),
                eta=secondsToText(
                    uptime * self.count // ((self.iteration + 1) * self.batch_size)
                ),
            )
        )
        self.now = time()

    def print_market(self):
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
                        cast(
                            ClauseElement, RedditMeme.version == LOAD_MEME_CLF_VERSION
                        ),
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

    def limbo(self):
        inTheMarket = [
            name
            for name, in cast(
                List[str],
                site_db.query(RedditMeme.meme_clf.label("name"),)
                .filter(
                    and_(
                        cast(ClauseElement, RedditMeme.stonk == True),
                        cast(
                            ClauseElement, RedditMeme.version == LOAD_MEME_CLF_VERSION
                        ),
                    )
                )
                .distinct(RedditMeme.meme_clf),
            )
        ]
        notInTheMarket = [
            name
            for name in get_static_names(LOAD_MEME_CLF_VERSION)["names"]
            if name not in inTheMarket
        ]
        for name in notInTheMarket:
            if name not in inTheMarket + list(DONT_USE_TEMPLATES):
                clear_output()
                print("notInTheMarket", len(notInTheMarket))
                print(name)
                _ = display_template(name)
                _ = input("next")

    @staticmethod
    def clear():
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
