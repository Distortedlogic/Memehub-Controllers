from controller import APP
from controller.reddit.scorer import score_redditors
from controller.reddit.scraper import scrape_reddit_memes

if __name__ == "__main__":
    with APP.app_context():
        scrape_reddit_memes(verbose=True)
        score_redditors()
