import json
from itertools import chain
from multiprocessing.pool import Pool as mpPool
from typing import Any, Dict, Iterator, List, Union, cast

import arrow
import requests
from arrow.arrow import Arrow
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


class RedditScrapper:
    def __init__(self):
        self.num_ids = 100

    def query_pushshift(self) -> Iterator[List[str]]:
        n = self.num_ids
        empty = False
        while not empty and n == self.num_ids:
            collection: List[Dict[str, Any]] = []
            while n == self.num_ids and len(collection) < 1000:
                url = PUSHSHIFT_URI.format(
                    self.sub, self.start_at, self.end_at, self.num_ids
                )
                if not (raw := make_request(url)):
                    break
                if not (posts := raw["data"]):
                    empty = True
                    break
                self.start_at: int = posts[-1]["created_utc"] - 10
                n = len(posts)
                collection.extend(posts)
            yield list(map(lambda post: post["id"], collection))

    def praw_memes(self, verbose: bool) -> Iterator[List[Dict[str, Any]]]:
        for ids in self.query_pushshift():
            with cast(mpPool, Pool(cpu_count(), initializer)) as workers:
                if verbose:
                    memes: List[Union[Dict[str, Any], None]] = list(
                        tqdm(workers.imap_unordered(praw_by_id, ids))
                    )
                else:
                    memes = list(workers.imap_unordered(praw_by_id, ids))
            yield [meme for meme in memes if meme and meme["username"] != "None"]

    def add_meme_to_db(self, meme: Dict[str, Any]):
        try:
            redditor = (
                site_db.query(Redditor).filter_by(username=meme["username"]).one()
            )
        except Exception:
            redditor = Redditor(username=meme["username"])
            site_db.add(redditor)
            site_db.commit()
        try:
            meme = site_db.query(RedditMeme).filter_by(url=meme["url"]).one()
        except Exception:
            site_db.add(RedditMeme(**meme, subreddit=self.sub, redditor_id=redditor.id))
        site_db.commit()

    def engine(self, verbose: bool):
        for meme in chain.from_iterable(self.praw_memes(verbose)):
            try:
                redditor = (
                    site_db.query(Redditor).filter_by(username=meme["username"]).one()
                )
            except Exception:
                redditor = Redditor(username=meme["username"])
                site_db.add(redditor)
                site_db.commit()
            try:
                meme = site_db.query(RedditMeme).filter_by(url=meme["url"]).one()
            except Exception:
                site_db.add(
                    RedditMeme(**meme, subreddit=self.sub, redditor_id=redditor.id)
                )
            site_db.commit()

    def scrape_reddit_memes(self, verbose: bool = False):
        print("STARTING CELERY")
        for self.sub in FULL_SUB_LIST:
            self.end_at = (
                cast(Arrow, arrow.utcnow())
                .shift(days=-1)
                .replace(second=0, minute=0)
                .timestamp
            )
            self.start_at = redditmeme_max_ts(self.sub)
            if not (self.start_at):
                self.start_at = get_beginning()
            self.engine(verbose=verbose)

    def scrape_reddit_memes_backwards(self, verbose: bool = False):
        for self.sub in FULL_SUB_LIST:
            print(self.sub) if verbose else None
            self.end_at = redditmeme_min_ts(self.sub)
            if not self.end_at:
                self.end_at = get_beginning()
            self.start_at = self.end_at - 60 * 60 * 24 * 30
            self.engine(verbose=verbose)
