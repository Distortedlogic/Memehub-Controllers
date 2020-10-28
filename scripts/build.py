from controller import APP
from controller.reddit.controller import RedditController
from controller.reddit.scorer import RedditScorer
from controller.redis.reddit import RedditReDB

if __name__ == "__main__":
    with APP.app_context():
        rc = RedditController(verbose=True)
        rs = RedditScorer()
        redb = RedditReDB()

        rc.update(full=True)
        rs.update()
        redb.update()
        # rc.build_features()
