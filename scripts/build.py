from controller import APP
from controller.reddit.controller import RedditController
from controller.reddit.scorer import RedditScorer
from controller.redis.reddit import RedditReDB
from controller.tasks.reddit import Reddit

if __name__ == "__main__":
    with APP.app_context():
        Reddit.delay(verbose=True)
