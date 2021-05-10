from src.celery_app import CELERY
from src.reddit.percentile import calc_percentiles
from src.reddit.scorer import score_redditors
from src.reddit.scraper import RedditScrapper
from src.stonks.market import StonkMarket


@CELERY.task(name="Reddit", unique_on=[], lock_expiry=60 * 60 * 12)
def Reddit():
    RedditScrapper().scrape_reddit_memes()
    score_redditors()
    StonkMarket(is_celery=True).reddit_engine()
    calc_percentiles()
