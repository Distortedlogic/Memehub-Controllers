from controller import APP
from controller.reddit.stonks import audit_reddit_stonks, print_market, reddit_stonks


def mh_print_market():
    with APP.app_context():
        print_market()


def mh_audit_reddit_stonks():
    with APP.app_context():
        audit_reddit_stonks()


def mh_reddit_stonks():
    with APP.app_context():
        reddit_stonks()


if __name__ == "__main__":
    mh_reddit_stonks()
