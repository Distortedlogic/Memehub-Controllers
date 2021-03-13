from pathlib import Path
from typing import cast

import arrow

VERSION = "0.2.3"
MODEL_REPO = f"{VERSION}/jit/"
Path(MODEL_REPO).mkdir(parents=True, exist_ok=True)
STATIC_PATH = f"{VERSION}/static.json"

MONTH_TD = 60 * 60 * 24 * 30
WEEK_TD = 60 * 60 * 24 * 7
DAY_TD = 60 * 60 * 24
HOUR_TD = 60 * 60


def get_beginning() -> int:
    return cast(
        int,
        arrow.utcnow().shift(days=-31).replace(hour=0, minute=0, second=0).timestamp,
    )


FULL_SUB_LIST = ["dankmemes", "memes"]

PUSHSHIFT_URI = r"https://api.pushshift.io/reddit/search/submission?subreddit={}&after={}&before={}&size={}"
