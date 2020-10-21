import json

import arrow
from controller.extensions import db
from controller.generated.models import Meme, MemeVote, User
from rejson import Client
from sqlalchemy import case, func

from redis import Redis

serialize = lambda user, ups, downs: dict(
    username=user.username,
    avatar=user.avatar,
    userId=user.id,
    ups=ups,
    downs=downs,
)


class LeaderBoard:
    def __init__(self):
        self.redis = Redis(host="redis", port=6379)

    def update(self):
        self.redis.set(
            "leaderboards:daily",
            json.dumps(
                {
                    "data": [
                        serialize(user, ups, downs)
                        for (user, ups, downs) in self.leader_data_by_days(1)
                    ]
                }
            ),
        )
        self.redis.set(
            "leaderboards:weekly",
            json.dumps(
                {
                    "data": [
                        serialize(user, ups, downs)
                        for (user, ups, downs) in self.leader_data_by_days(7)
                    ]
                }
            ),
        )
        self.redis.set(
            "leaderboards:monthly",
            json.dumps(
                {
                    "data": [
                        serialize(user, ups, downs)
                        for (user, ups, downs) in self.leader_data_by_days(30)
                    ]
                }
            ),
        )
        self.redis.set(
            "leaderboards:ever",
            json.dumps(
                {
                    "data": [
                        serialize(user, ups, downs)
                        for (user, ups, downs) in self.leader_data_by_days()
                    ]
                }
            ),
        )

    def leader_data_by_days(self, days=None):
        if days:
            return (
                db.session.query(
                    User,
                    func.sum(case([(MemeVote.upvote == True, 1)], else_=0)),
                    func.sum(case([(MemeVote.upvote == False, 1)], else_=0)),
                )
                .join(User.memes)
                .join(Meme.meme_votes)
                .group_by(User)
                .order_by(
                    (
                        func.sum(case([(MemeVote.upvote == True, 1)], else_=0))
                        - func.sum(case([(MemeVote.upvote == False, 1)], else_=0))
                    ).desc()
                )
                .filter(MemeVote.createdAt > arrow.utcnow().shift(days=-1).datetime)
                .limit(3)
                .all()
            )
        else:
            return (
                db.session.query(
                    User,
                    func.sum(case([(MemeVote.upvote == True, 1)], else_=0)),
                    func.sum(case([(MemeVote.upvote == False, 1)], else_=0)),
                )
                .join(User.memes)
                .join(Meme.meme_votes)
                .group_by(User)
                .order_by(func.count(MemeVote.upvote).desc())
                .limit(3)
                .all()
            )
