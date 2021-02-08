import json
import time

import arrow
import requests
from controller.constants import PUSHSHIFT_URI
from retry import retry


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
            raw = make_request(url)
            if not raw:
                break
            posts = raw["data"]
            if not posts:
                break
            start_at = posts[-1]["created_utc"] - 10
            n = len(posts)
            collection.extend(posts)
        ids = list(map(lambda post: post["id"], collection))
        return ids
