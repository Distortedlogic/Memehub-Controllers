from typing import cast

from controller.generated.models import RedditMeme, RedditScore
from controller.session import site_db
from sqlalchemy.sql.functions import func


def redditmeme_min_ts(subreddit: str) -> int:
    try:
        min_ts = cast(
            int,
            site_db.query(func.min(RedditMeme.timestamp))
            .filter_by(subreddit=subreddit)
            .scalar(),
        )
        return min_ts
    except:
        return 0


def redditmeme_max_ts(subreddit: str) -> int:
    try:
        max_ts = cast(
            int,
            site_db.query(func.max(RedditMeme.timestamp))
            .filter_by(subreddit=subreddit)
            .scalar(),
        )
        return max_ts
    except:
        return 0


def redditscore_min_ts(subreddit: str) -> int:
    try:
        min_ts = cast(
            int,
            site_db.query(func.min(RedditScore.timestamp))
            .filter_by(subreddit=subreddit)
            .scalar(),
        )
        return min_ts
    except:
        return 0


def redditscore_max_ts(subreddit: str) -> int:
    try:
        max_ts = cast(
            int,
            site_db.query(func.max(RedditScore.timestamp))
            .filter_by(subreddit=subreddit)
            .scalar(),
        )
        return max_ts
    except:
        return 0
