import json
import time

import arrow
import requests
from billiard import Pool, cpu_count
from controller.constants import FULL_SUB_LIST, PUSHSHIFT_URI, get_beginning
from controller.generated.models import RedditMeme, Redditor, db
from controller.reddit.functions.database import redditmeme_max_ts
from controller.reddit.functions.praw_mp import initializer, praw_by_id
from retry import retry
from tqdm import tqdm


@retry(tries=5, delay=2)
def make_request(uri):
    with requests.get(uri) as resp:
        return json.loads(resp.content)


def query_pushshift(subreddit, start_at, end_at):
    SIZE = 100
    n = SIZE
    while n == SIZE:
        collection = []
        while n == SIZE and len(collection) < 1000:
            url = PUSHSHIFT_URI.format(subreddit, start_at, end_at, SIZE)
            if not (raw := make_request(url)):
                break
            if not (posts := raw["data"]):
                break
            start_at = posts[-1]["created_utc"] - 10
            n = len(posts)
            collection.extend(posts)
        yield list(map(lambda post: post["id"], collection))


def stream(sub: str, max_ts: int, now: int, verbose):
    prev_ids = []
    for ids in query_pushshift(sub, max_ts, now):
        with Pool(cpu_count(), initializer) as workers:
            if verbose:
                memes = list(tqdm(workers.imap_unordered(praw_by_id, ids)))
            else:
                memes = list(workers.imap_unordered(praw_by_id, ids))
        yield [meme for meme in memes if meme and meme["username"] != "None"]


def engine(sub, verbose):
    now = arrow.utcnow().shift(days=-1).replace(second=0, minute=0).timestamp
    max_ts = redditmeme_max_ts(sub)
    if not max_ts:
        max_ts = get_beginning()
    print(sub)
    for raw_memes in stream(sub, max_ts, now, verbose):
        print(arrow.get(max(meme["timestamp"] for meme in raw_memes)).format())
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
