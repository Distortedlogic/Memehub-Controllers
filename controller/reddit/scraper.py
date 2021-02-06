import arrow
from billiard import Pool, cpu_count
from controller.constants import FULL_SUB_LIST, get_beginning
from controller.generated.models import RedditMeme, Redditor, db
from controller.reddit.functions.database import redditmeme_max_ts
from controller.reddit.functions.praw_mp import initializer, praw_by_id
from controller.reddit.functions.pushshift import query_pushshift
from tqdm import tqdm


def stream(sub: str, max_ts: int, now: int, verbose):
    for id_iter in query_pushshift(sub, max_ts, now):
        with Pool(cpu_count(), initializer) as workers:
            if verbose:
                memes = list(tqdm(workers.imap_unordered(praw_by_id, id_iter)))
            else:
                memes = list(workers.imap_unordered(praw_by_id, id_iter))
        yield [meme for meme in memes if meme and meme["username"] != "None"]


def engine(sub, verbose):
    now = arrow.utcnow().shift(days=-1).replace(second=0, minute=0).timestamp
    max_ts = redditmeme_max_ts(sub)
    if not max_ts:
        max_ts = get_beginning()
    if verbose:
        print(sub)
    for raw_memes in stream(sub, max_ts, now, verbose):
        for meme in raw_memes:
            try:
                redditor = (
                    db.session.query(Redditor)
                    .filter_by(username=meme["username"])
                    .one()
                )
            except:
                redditor = Redditor(username=meme["username"])
                db.session.add(redditor)
                db.session.commit()
            try:
                meme = db.session.query(RedditMeme).filter_by(url=meme["url"]).one()
            except:
                db.session.add(
                    RedditMeme(**meme, subreddit=sub, redditor_id=redditor.id)
                )
        db.session.commit()


def scrape_reddit_memes(verbose=False):
    for sub in FULL_SUB_LIST:
        engine(sub, verbose=verbose)
