import json
import time

import arrow
import requests
from controller.constants import PUSHSHIFT_URI


def make_request(uri):
    current_tries = 0
    while current_tries < 50:
        response = None
        try:
            response = requests.get(uri)
            return json.loads(response.content)
        except Exception as e:
            time.sleep(current_tries)
            current_tries += 1
            print(e)
            if response:
                print(response)
            else:
                print("cant print response")
            print(f"Pushshift request FAILED - retries {current_tries}")


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
