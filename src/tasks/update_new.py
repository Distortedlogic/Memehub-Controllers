from src.celery_app import CELERY
from src.generated.models import RedditNew
from src.reddit.functions.praw_mp import init_reddit, praw_by_id
from src.session import site_db


@CELERY.task(name="RedditNew", unique_on=[], lock_expiry=60 * 60 * 12)
def update_new():
    reddit = init_reddit(1)
    _ = site_db.query(RedditNew).delete()
    submissions = (praw_by_id(id) for id in reddit.subreddit("dankmemes").new())
    site_db.add_all(
        RedditNew(
            username=submission["username"],
            reddit_id=submission["reddit_id"],
            title=submission["title"],
            url=submission["url"],
            created_at=submission["created_at"],
        )
        for submission in submissions
        if submission
    )
    site_db.commit()

