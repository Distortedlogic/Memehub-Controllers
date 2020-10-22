from datetime import datetime

import pandas as pd
from sqlalchemy import and_
from sqlalchemy.orm import load_only

from controller.constants import HOUR_TD, MONTH_TD
from controller.generated.models import RedditMeme, RedditScore, db
from controller.reddit.functions.database import (
    get_subs_to_scrape,
    redditmeme_max_ts,
    redditmeme_min_ts,
    redditscore_max_ts,
)
from controller.reddit.functions.dataframe import score_df, score_kwargs_gen
from controller.reddit.functions.misc import round_hour


class RedditScorer:
    def update(self, interval=HOUR_TD, td=MONTH_TD):
        for self.subreddit in get_subs_to_scrape():
            meme_min_ts = redditmeme_min_ts(self.subreddit)
            meme_max_ts = redditmeme_max_ts(self.subreddit)
            score_max_ts = redditscore_max_ts(self.subreddit)
            if not score_max_ts:
                score_max_ts = round_hour(meme_min_ts) + td - interval
            if score_max_ts < round_hour(meme_max_ts) - interval:
                while score_max_ts <= round_hour(meme_max_ts) - interval:
                    next_step = score_max_ts + interval
                    self.score(next_step - td, next_step)
                    score_max_ts = next_step

    def score(self, start_ts, end_ts):
        cols = ["username", "upvotes", "upvote_ratio"]
        df = pd.read_sql(
            db.session.query(RedditMeme)
            .filter(
                and_(start_ts < RedditMeme.timestamp, RedditMeme.timestamp < end_ts)
            )
            .filter_by(subreddit=self.subreddit)
            .options(load_only(*cols))
            .statement,
            db.session.bind,
        )
        scored = score_df(df)
        db.session.add_all(
            RedditScore(
                **kwargs,
                timestamp=end_ts,
                datetime=datetime.fromtimestamp(end_ts),
                subreddit=self.subreddit,
                time_delta=end_ts - start_ts
            )
            for kwargs in score_kwargs_gen(scored)
        )
        db.session.commit()
