import json
from multiprocessing.pool import Pool as mpPool
from typing import Any, Dict, Iterator, List, Union, cast

import arrow
import requests
from billiard import Pool, cpu_count  # type:ignore
from retry import retry
from src.constants import FULL_SUB_LIST, PUSHSHIFT_URI, get_beginning
from src.generated.models import RedditMeme, Redditor
from src.reddit.functions.database import redditmeme_max_ts, redditmeme_min_ts
from src.reddit.functions.praw_mp import initializer, praw_by_id
from src.session import site_db
from tqdm import tqdm


@retry(tries=5, delay=2)
def make_request(uri: str) -> Dict[str, List[Dict[str, Any]]]:
    with requests.get(uri) as resp:
        return json.loads(resp.content)


def query_pushshift(subreddit: str, start_at: int, end_at: int) -> Iterator[List[str]]:
    SIZE = 100
    n = SIZE
    while n == SIZE:
        collection: List[Dict[str, Any]] = []
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


def stream(
    sub: str, max_ts: int, now: int, verbose: bool
) -> Iterator[List[Dict[str, Any]]]:
    for ids in query_pushshift(sub, max_ts, now):
        with cast(mpPool, Pool(cpu_count(), initializer)) as workers:
            if verbose:
                memes: List[Union[Dict[str, Any], None]] = list(
                    tqdm(workers.imap_unordered(praw_by_id, ids))
                )
            else:
                memes = list(workers.imap_unordered(praw_by_id, ids))
        yield [meme for meme in memes if meme and meme["username"] != "None"]


def engine(sub: str, max_ts: int, now: int, verbose: bool):
    for raw_memes in stream(sub, max_ts, now, verbose):
        for meme in raw_memes:
            try:
                redditor = (
                    site_db.query(Redditor).filter_by(username=meme["username"]).one()
                )
            except:
                redditor = Redditor(username=meme["username"])
                site_db.add(redditor)
                site_db.commit()
            try:
                meme = site_db.query(RedditMeme).filter_by(url=meme["url"]).one()
            except:
                site_db.add(RedditMeme(**meme, subreddit=sub, redditor_id=redditor.id))
        site_db.commit()


def scrape_reddit_memes(verbose: bool = False):
    for sub in FULL_SUB_LIST:
        now: int = arrow.utcnow().shift(days=-1).replace(second=0, minute=0).timestamp
        if not (max_ts := redditmeme_max_ts(sub)):
            max_ts = get_beginning()
        engine(sub, max_ts, now, verbose=verbose)


def scrape_reddit_memes_backwards(verbose: bool = False):
    for sub in FULL_SUB_LIST:
        print(sub) if verbose else None
        if not (min_ts := redditmeme_min_ts(sub)):
            min_ts = get_beginning()
        engine(sub, min_ts - 60 * 60 * 24 * 30, min_ts, verbose=verbose)
