from src.celery_app import CELERY


@CELERY.task(name="Stonks", unique_on=[], lock_expiry=60 * 60 * 12)
def Stonks():
    # stonks = StonkMarket()
    # stonks.reddit_engine()
    # stonks.site_engine()
    pass
