from controller import CELERY
from controller.reddit.controller import RedditController
from controller.reddit.scorer import RedditScorer
from controller.redis.reddit import RedditReDB


@CELERY.task(name="Reddit", unique_on=[], lock_expiry=60 * 60 * 12)
def Reddit(verbose=None, full=True):
    rc = RedditController()
    rs = RedditScorer()
    redb = RedditReDB()
    rc.update(full=full, verbose=verbose)
    rs.update()
    redb.update()
