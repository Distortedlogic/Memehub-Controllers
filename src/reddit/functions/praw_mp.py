import random
from datetime import datetime
from json import loads
from typing import Any, Dict, TypedDict, cast

from decouple import config
from praw import Reddit
from praw.reddit import Submission


def init_reddit(id: int):
    reddit_oauth = loads(cast(str, config("reddit_oauth_" + str(id))))
    return Reddit(
        client_id=reddit_oauth["CLIENT_ID"],
        client_secret=reddit_oauth["CLIENT_SECRET"],
        password=reddit_oauth["PASSWORD"],
        user_agent=reddit_oauth["USER_AGENT"],
        username=reddit_oauth["USERNAME"],
    )


NUM_REDDIT_INSTANCES = 8
reddit = None
reddit_objs = [init_reddit(i) for i in range(8)]


def initializer():
    worker_id = random.choice(range(NUM_REDDIT_INSTANCES))
    global reddit
    tries = 0
    while True:
        try:
            reddit = reddit_objs[worker_id]
            break
        except Exception:
            try:
                reddit = init_reddit(worker_id)
                break
            except Exception:
                worker_id += 1 % NUM_REDDIT_INSTANCES
                tries += 1
                if tries > 2 * NUM_REDDIT_INSTANCES:
                    raise Exception("reddit instance error")


def praw_by_id(submission_id: str):
    try:
        submission: Submission = cast(Reddit, reddit).submission(id=submission_id)
        if not submission.stickied:
            if any(
                submission.url.endswith(filetype)
                for filetype in [".jpg", ".jpeg", ".png"]
            ):
                return extract_data(submission)
    except Exception:
        pass


class redditData(TypedDict):
    reddit_id: str
    title: str
    username: str
    timestamp: int
    created_at: datetime
    url: str
    upvote_ratio: float
    upvotes: int
    downvotes: int
    num_comments: int


def extract_data(submission: Submission) -> redditData:
    return {
        "reddit_id": submission.id,
        "title": submission.title,
        "username": str(submission.author),
        "timestamp": submission.created_utc,
        "created_at": datetime.fromtimestamp(submission.created_utc),
        "url": submission.url,
        "upvote_ratio": submission.upvote_ratio,
        "upvotes": submission.score,
        "downvotes": round(submission.score / submission.upvote_ratio)
        - submission.score,
        "num_comments": submission.num_comments,
    }
