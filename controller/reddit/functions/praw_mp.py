from datetime import datetime
from json import loads

from billiard import current_process
from controller.generated.models import RedditMeme, db
from decouple import config
from praw import Reddit
from praw.reddit import Submission


def init_reddit(id):
    reddit_oauth = loads(config("reddit_oauth_" + str(id)))
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
    try:
        pid = current_process().name.split("-", 1)[1].split(":", 1)[1]
        process_id = (int(pid) - 1) % NUM_REDDIT_INSTANCES
    except:
        pid = current_process().name.split("-", 1)[1]
        process_id = (int(pid) - 1) % NUM_REDDIT_INSTANCES
    worker_id = process_id
    global reddit
    tries = 0
    while True:
        try:
            reddit = reddit_objs[worker_id]
            break
        except:
            try:
                reddit = init_reddit(worker_id)
                break
            except:
                worker_id += 1 % NUM_REDDIT_INSTANCES
                tries += 1
                if tries > 2 * NUM_REDDIT_INSTANCES:
                    raise Exception("reddit instance error")


def praw_by_id(submission_id):
    try:
        submission: Submission = reddit.submission(id=submission_id)
        if not submission.stickied:
            if any(
                submission.url.endswith(filetype)
                for filetype in [".jpg", ".jpeg", ".png"]
            ):
                return extract_data(submission)
    except:
        pass


def extract_data(submission):
    return dict(
        reddit_id=submission.id,
        title=submission.title,
        username=str(submission.author),
        timestamp=submission.created_utc,
        created_at=datetime.fromtimestamp(submission.created_utc),
        url=submission.url,
        upvote_ratio=submission.upvote_ratio,
        upvotes=submission.score,
        downvotes=round(submission.score / submission.upvote_ratio) - submission.score,
        num_comments=submission.num_comments,
    )
