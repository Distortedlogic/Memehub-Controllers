from controller.redis.leaderboard import LeaderBoard
from controller import APP

if __name__ == "__main__":
    with APP.app_context():
        LeaderBoard().update()