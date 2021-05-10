from typing import Any, Dict, Tuple, Union, cast

import arrow
import pandas as pd
from arrow.arrow import Arrow
from sqlalchemy.sql.elements import ClauseElement
from sqlalchemy.sql.expression import and_
from sqlalchemy.sql.functions import func
from src.constants import FULL_SUB_LIST
from src.generated.models import RedditMeme
from src.session import site_db


def calc_percentiles(reset: bool = False):
    if reset:
        while (
            memes := site_db.query(RedditMeme)
            .filter(cast(ClauseElement, RedditMeme.percentile != None))
            .limit(1000)
            .all()
        ) :
            for meme in memes:
                meme.percentile = None  # type: ignore
            site_db.commit()
    for subreddit in FULL_SUB_LIST:
        calc_percentile_sub(subreddit)


def update_meme(
    idx_row: Tuple[ClauseElement, Tuple[int, Dict[str, Union[str, float]]]]
):
    subreddit_clause, (_, row) = idx_row
    if meme := (
        site_db.query(RedditMeme)
        .filter(
            and_(
                cast(ClauseElement, RedditMeme.reddit_id == row["reddit_id"]),
                cast(ClauseElement, RedditMeme.percentile == None),
                subreddit_clause,
            )
        )
        .first()
    ):
        meme.percentile = row["percentile"]  # type: ignore


def calc_percentile_sub(subreddit: str):
    subreddit_clause = cast(ClauseElement, RedditMeme.subreddit == subreddit)
    if not (
        timestamp := cast(
            Union[int, None],
            site_db.query(func.max(RedditMeme.timestamp))
            .filter(
                and_(
                    subreddit_clause, cast(ClauseElement, RedditMeme.percentile != None)
                )
            )
            .scalar(),
        )
    ):
        max_ts = (
            cast(
                Arrow,
                arrow.get(
                    cast(
                        int,
                        site_db.query(func.min(RedditMeme.timestamp))
                        .filter(subreddit_clause)
                        .scalar(),
                    )
                ),
            )
            .ceil("hour")
            .shift(days=1)
        )
    else:
        max_ts = cast(Arrow, arrow.get(timestamp)).ceil("hour").shift(hours=1)
    df = pd.read_sql(
        cast(
            str,
            site_db.query(
                RedditMeme.reddit_id,
                RedditMeme.upvotes,
                RedditMeme.created_at,
                RedditMeme.percentile,
            )
            .filter(
                and_(
                    subreddit_clause,
                    cast(
                        ClauseElement,
                        RedditMeme.created_at > max_ts.shift(days=-1).datetime,
                    ),
                )
            )
            .statement,
        ),
        site_db.bind,
        columns=["reddit_id", "upvotes", "created_at", "percentile"],
    )
    while max_ts < cast(Arrow, arrow.utcnow()).floor("hour").shift(days=-1, seconds=-1):
        print(cast(str, max_ts.format("YYYY-MM-DD HH:mm:ss")))
        frame = df.loc[
            cast(
                Any,
                (df["created_at"] > max_ts.naive)
                & (df["created_at"] < max_ts.shift(days=1).naive),
            ),
            ["upvotes", "reddit_id", "percentile"],
        ]
        empty_percentiles = frame["percentile"].isna()
        frame["percentile"] = frame["upvotes"].rank(pct=True)
        for _, row in frame.loc[empty_percentiles].iterrows():
            df.loc[
                df["reddit_id"] == cast(str, row["reddit_id"]), ["percentile"]
            ] = cast(float, row["percentile"])
            if meme := (
                site_db.query(RedditMeme)
                .filter(
                    and_(
                        cast(ClauseElement, RedditMeme.reddit_id == row["reddit_id"]),
                        cast(ClauseElement, RedditMeme.percentile == None),
                        subreddit_clause,
                    )
                )
                .first()
            ):
                meme.percentile = row["percentile"]  # type: ignore
        site_db.commit()
        max_ts: Arrow = max_ts.shift(hours=1)
