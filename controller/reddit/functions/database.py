from typing import List
from controller.extensions import db
from sqlalchemy import and_, func
import time


def num_posts(subreddit: str, start: int, end: int) -> int:
    from controller.reddit import RedditMeme

    try:
        return (
            db.session.query(RedditMeme)
            .filter(
                and_(
                    start < RedditMeme.timestamp,
                    RedditMeme.timestamp < end,
                    RedditMeme.subreddit == subreddit,
                )
            )
            .count()
        )
    except:
        return 0


def percent_change(subreddit: str, start: int, end: int) -> float:
    from controller.reddit import RedditMeme

    try:
        current = (
            db.session.query(RedditMeme)
            .filter(
                and_(
                    start < RedditMeme.timestamp,
                    RedditMeme.timestamp < end,
                    RedditMeme.subreddit == subreddit,
                )
            )
            .count()
        )
        previous = (
            db.session.query(RedditMeme)
            .filter(
                and_(
                    (2 * start - end) < RedditMeme.timestamp,
                    RedditMeme.timestamp < start,
                    RedditMeme.subreddit == subreddit,
                )
            )
            .count()
        )
        return round(100 * (current - previous) / previous, 2)
    except:
        return 0


def redditmeme_min_ts(subreddit: str) -> int:
    from controller.reddit import RedditMeme

    try:
        min_ts = int(
            db.session.query(db.func.min(RedditMeme.timestamp))
            .filter_by(subreddit=subreddit)
            .scalar()
        )
        return min_ts
    except:
        return 0


def redditmeme_max_ts(subreddit: str) -> int:
    from controller.reddit import RedditMeme

    try:
        max_ts = int(
            db.session.query(db.func.max(RedditMeme.timestamp))
            .filter_by(subreddit=subreddit)
            .scalar()
        )
        return max_ts
    except:
        return 0


def redditscore_min_ts(subreddit: str) -> int:
    from controller.reddit import RedditScore

    try:
        min_ts = int(
            db.session.query(db.func.min(RedditScore.timestamp))
            .filter_by(subreddit=subreddit)
            .scalar()
        )
        return min_ts
    except:
        return 0


def redditscore_max_ts(subreddit: str) -> int:
    from controller.reddit import RedditScore

    try:
        max_ts = int(
            db.session.query(db.func.max(RedditScore.timestamp))
            .filter_by(subreddit=subreddit)
            .scalar()
        )
        return max_ts
    except:
        return 0


def get_subs_to_scrape() -> List[str]:
    from controller.reddit import RedditMeme

    try:
        subs: List[str] = [
            data[0]
            for data in db.session.query(RedditMeme.subreddit)
            .group_by(RedditMeme.subreddit)
            .all()
            if (
                redditmeme_max_ts(data[0])
                and redditmeme_max_ts(data[0]) > int(time.time() - 60 * 60 * 24)
            )
        ]
        return subs
    except:
        raise Exception("no subs found")
