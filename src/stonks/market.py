from time import time
from typing import Any, Iterator, List, Tuple, cast

import numpy as np
import pandas as pd
from IPython.core.display import clear_output
from redisai import Client
from sqlalchemy import and_, func
from sqlalchemy.sql.elements import ClauseElement
from src.constants import LOAD_MEME_CLF_VERSION
from src.generated.models import Meme, RedditMeme
from src.session import site_db
from src.utils.display import display_df, pretty_print_dict
from src.utils.image_funcs import isDeletedException, load_tensor_from_url
from src.utils.model_func import get_static_names
from src.utils.secondToText import secondsToText
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset


class MemeSet(IterableDataset[Tensor]):
    def __init__(self, entity: Any):
        self.entity = entity

    def __iter__(self):
        for meme in (
            site_db.query(self.entity)
            .filter(cast(ClauseElement, self.entity.version == None))
            .order_by(cast(ClauseElement, self.entity.created_at.desc()))
        ):
            try:
                image = load_tensor_from_url(meme.url, is_deleted=True)
                yield (image, meme.id)
            except isDeletedException:
                site_db.delete(meme)
                site_db.commit()

    def count(self):
        return (
            site_db.query(self.entity).filter(
                cast(ClauseElement, self.entity.version == None)
            )
        ).count()


class StonkMarket:
    def __init__(self):
        self.rai = Client(host="redis", port=6379)
        self.batch_size = 128

    def reddit_engine(self):
        self.entity = RedditMeme
        self.engine()

    def site_engine(self):
        self.entity = Meme
        self.engine()

    def run_model_name(self, name: str):
        _ = self.rai.modelrun(name, "features_out", f"{name}_out")

    def update_meme(self, idx_id_name: Tuple[int, Tuple[int, str]]):
        idx, (id, name) = idx_id_name
        if (meme := site_db.query(self.entity).get(id)) :
            is_stonk = bool(
                round(cast(List[int], self.rai.tensorget(f"{name}_out"))[idx])
            )
            meme.meme_clf = name  # type: ignore
            meme.stonk = is_stonk  # type: ignore
            meme.version = LOAD_MEME_CLF_VERSION  # type: ignore
        site_db.commit()

    def engine(self):
        self.dataset = MemeSet(self.entity)
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
            _ = self.rai.tensorset("images", np.array(images).astype(np.float32))
            _ = self.rai.modelrun("features", "images", "features_out")
            _ = self.rai.modelrun("dense", "features_out", "dense_out")
            names: List[str] = [
                num_name[str(cast(int, np.argmax(arr)))]
                for arr in cast(List[Tensor], self.rai.tensorget("dense_out"))
            ]
            for name in set(names):
                _ = self.run_model_name(name)
            for item in cast(
                Iterator[Tuple[int, Tuple[int, str]]],
                enumerate(zip(ids.numpy().astype(int).tolist(), names)),
            ):
                self.update_meme(item)
            if self.iteration % 10 == 0:
                self.print_stats()

    def print_stats(self):
        clear_output()
        uptime = int(time() - self.start)
        count = self.dataset.count()
        memes_done = (
            site_db.query(self.entity)
            .filter(cast(ClauseElement, self.entity.version != None))
            .count()
        )
        memes_found = (
            site_db.query(self.entity)
            .filter(cast(ClauseElement, self.entity.stonk == True))
            .count()
        )
        pretty_print_dict(
            dict(
                memes_done=memes_done,
                memes_found=memes_found,
                ratio=memes_found / memes_done,
                num_left=count,
                iteration=self.iteration,
                round_time=secondsToText(int(time() - self.now) // 10),
                uptime=secondsToText(uptime),
                eta=secondsToText(
                    uptime * count // ((self.iteration + 1) * self.batch_size)
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

