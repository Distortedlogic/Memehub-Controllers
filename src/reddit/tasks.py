from src import CELERY
from src.reddit.scorer import score_redditors
from src.reddit.scraper import scrape_reddit_memes


@CELERY.task(name="Reddit", unique_on=[], lock_expiry=60 * 60 * 12)
def Reddit():
    scrape_reddit_memes()
    score_redditors()
