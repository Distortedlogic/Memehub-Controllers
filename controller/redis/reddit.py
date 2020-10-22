import json

from controller.constants import DAY_TD, FULL_SUB_LIST, MONTH_TD, WEEK_TD
from controller.generated.models import db
from controller.reddit import RedditMeme, RedditScore
from controller.reddit.functions import *
from controller.reddit.functions.database import (
    num_posts,
    percent_change,
    redditmeme_max_ts,
    redditmeme_min_ts,
    redditscore_max_ts,
    redditscore_min_ts,
)
from sqlalchemy import and_, func

from redis import Redis

timeframes = [
    ("Daily", DAY_TD),
    ("Weekly", WEEK_TD),
    ("Monthly", MONTH_TD),
    ("Ever", None),
]


class RedditReDB:
    def __init__(self):
        self.redis = Redis(host="redis", port=6379)

    def _set(self, name, obj):
        self.redis.set(name, json.dumps(obj))

    @property
    def current(self):
        return [
            {"overview": self.redis.get("overview")},
            {"top_month": self.redis.get("top_month")},
        ]

    @property
    def serialize(self):
        return [{"overview": self.overview}, {"top_month": self.top_month}]

    def update(self):
        print("started")
        self.subreddit_list = FULL_SUB_LIST
        self.overview = {
            "totals": {"Daily": [], "Weekly": [], "Monthly": [], "Ever": []}
        }
        self.top_month = {"data": []}
        for idx, subreddit in enumerate(self.subreddit_list):
            min_ts, max_ts = redditmeme_min_ts(subreddit), redditmeme_max_ts(subreddit)
            if max_ts:
                for tf, delta in timeframes:
                    delta = delta if delta else max_ts - min_ts
                    self.overview["totals"][tf].append(
                        {
                            "sub": subreddit,
                            "amount": num_posts(subreddit, max_ts - delta, max_ts),
                            "percent": percent_change(
                                subreddit, max_ts - delta, max_ts
                            ),
                        }
                    )
            min_ts, max_ts = redditscore_min_ts(subreddit), redditscore_max_ts(
                subreddit
            )
            if max_ts:
                data = [
                    data.stats
                    for data in db.session.query(RedditScore)
                    .filter(
                        and_(
                            RedditScore.timestamp == max_ts,
                            RedditScore.subreddit == subreddit,
                        )
                    )
                    .order_by(RedditScore.final_score.desc())
                    .limit(5)
                    .all()
                ]
                self.top_month["data"].append(data)
                for data in self.top_month["data"][idx]:
                    data["sub"] = subreddit
                    data["url"] = (
                        db.session.query(RedditMeme.url)
                        .filter(RedditMeme.username == data["username"])
                        .order_by(func.random())
                        .first()[0]
                    )
        self._set("overview", self.overview)
        self._set("top_month", self.top_month)
