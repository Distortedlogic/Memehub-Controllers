from controller import APP
from controller.reddit.scraper import scrape_reddit_memes_backwards

if __name__ == "__main__":
    with APP.app_context():
        scrape_reddit_memes_backwards(verbose=True)
