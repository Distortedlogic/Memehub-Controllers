from src.celery_app import CELERY
from src.reddit.scorer import score_redditors
from src.reddit.scraper import RedditScrapper


@CELERY.task(name="Reddit", unique_on=[], lock_expiry=60 * 60 * 12)
def Reddit():
    RedditScrapper().scrape_reddit_memes()
    score_redditors()
