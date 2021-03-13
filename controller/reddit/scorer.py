from datetime import datetime
from typing import cast

import pandas as pd
from controller.constants import FULL_SUB_LIST, HOUR_TD, MONTH_TD
from controller.generated.models import RedditMeme, RedditScore
from controller.reddit.functions.database import (
    redditmeme_max_ts,
    redditmeme_min_ts,
    redditscore_max_ts,
)
from controller.reddit.functions.dataframe import score_df, score_kwargs_gen
from controller.reddit.functions.misc import round_hour
from controller.session import site_db
from sqlalchemy import and_
from sqlalchemy.orm import load_only
from sqlalchemy.sql.expression import ClauseElement


def score(sub: str, start_ts: int, end_ts: int):
    cols = ["username", "upvotes", "upvote_ratio"]
    df = pd.read_sql(
        cast(
            str,
            site_db.query(RedditMeme)
            .filter(
                and_(
                    cast(ClauseElement, start_ts < RedditMeme.timestamp),
                    cast(ClauseElement, RedditMeme.timestamp < end_ts),
                )
            )
            .filter_by(subreddit=sub)
            .options(load_only(*cols))
            .statement,
        ),
        site_db.bind,
    )
    scored = score_df(df)
    site_db.add_all(
        RedditScore(
            **kwargs,
            timestamp=end_ts,
            datetime=datetime.fromtimestamp(end_ts),
            subreddit=sub,
            time_delta=end_ts - start_ts
        )
        for kwargs in score_kwargs_gen(scored)
    )
    site_db.commit()


def score_redditors(interval: int = HOUR_TD, td: int = MONTH_TD, verbose: bool = False):
    if verbose:
        print("Scoring")
    for sub in FULL_SUB_LIST:
        meme_min_ts = redditmeme_min_ts(sub)
        meme_max_ts = redditmeme_max_ts(sub)
        score_max_ts = redditscore_max_ts(sub)
        if not score_max_ts:
            score_max_ts = round_hour(meme_min_ts) + td - interval
        if score_max_ts < round_hour(meme_max_ts) - interval:
            while score_max_ts <= round_hour(meme_max_ts) - interval:
                next_step = score_max_ts + interval
                score(sub, next_step - td, next_step)
                score_max_ts = next_step
