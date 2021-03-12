from controller.generated.models import RedditMeme, RedditScore, db


def redditmeme_min_ts(subreddit: str) -> int:
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
    try:
        max_ts = int(
            db.session.query(db.func.max(RedditScore.timestamp))
            .filter_by(subreddit=subreddit)
            .scalar()
        )
        return max_ts
    except:
        return 0
