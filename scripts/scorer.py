from controller import APP
from controller.reddit.scorer import RedditScorer

if __name__ == "__main__":
    with APP.app_context():
        RedditScorer().update()
