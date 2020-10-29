import datetime
import os
import random

import arrow
from controller.generated.models import (
    Comment,
    CommentVote,
    Meme,
    MemeVote,
    RedditMeme,
    User,
    db,
)
from sqlalchemy import and_, func, not_

from .utils import *

clear = lambda: os.system("clear")


class Simulator:
    def __init__(self, verbose=True):
        init_db()
        self.verbose = verbose
        self.interval = 1
        self.meme_repo = (
            db.session.query(RedditMeme.url)
            .distinct(RedditMeme.url)
            .order_by(RedditMeme.url, RedditMeme.upvotes.desc())
            .all()
        )

    def update_user_settings(self):
        self.followable = (
            db.session.query(User.id)
            .filter(not_(User.followers.any(Follow.followerId == self.current_userId)))
            .order_by(func.random())
            .limit(2)
            .all()
        )
        self.commentable_memes = (
            db.session.query(Meme.id)
            .filter(
                not_(Meme.comments.any(Comment.userId == self.current_userId)),
            )
            .order_by(func.random())
            .limit(10)
            .all()
        )
        self.votable_memes = (
            db.session.query(Meme.id)
            .filter(not_(Meme.meme_votes.any(MemeVote.userId == self.current_userId)))
            .group_by(Meme)
            .order_by(func.random())
            .limit(30)
            .all()
        )
        self.votable_comments = (
            db.session.query(Comment.id)
            .filter(
                not_(
                    Comment.comment_votes.any(CommentVote.userId == self.current_userId)
                )
            )
            .group_by(Comment)
            .order_by(func.random())
            .limit(20)
            .all()
        )

    def engine(self):
        for _ in range(5):
            create_user(self.t1, self.t2)
        for user in db.session.query(User):
            self.AUS = 0.2
            if roll_dice(self.AUS):
                continue
            self.current_userId = user.id
            self.update_user_settings()
            for (userId,) in self.followable:
                follow_user(user.id, userId, self.t1, self.t2)
            for _ in range(random.choice([0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2])):
                meme = self.meme_repo.pop()
                post_meme(user.id, meme, self.t1, self.t2)
            for (memeId,) in self.commentable_memes:
                post_comment(user.id, memeId, self.t1, self.t2)
            for (memeId,) in self.votable_memes:
                vote_meme(user.id, memeId, self.t1, self.t2)
            for (commentId,) in self.votable_comments:
                vote_comment(user.id, commentId, self.t1, self.t2)
            db.session.commit()

    def dbseed(self):
        farback = 40

        start = arrow.utcnow()
        self.t2 = arrow.utcnow().shift(days=-farback)
        self.t1 = self.t2.shift(days=-farback + self.interval)
        self.rounds_ran = 0
        self.total_runtime = datetime.timedelta(seconds=0)
        while arrow.utcnow() > self.t2:
            now = arrow.utcnow()
            self.t2 = self.t2.shift(days=self.interval)
            self.t1 = self.t1.shift(days=self.interval)
            self.engine()
            self.rounds_ran += 1
            self.runtime = arrow.utcnow() - now
            self.total_runtime += self.runtime
            self.print_stats()
        now = arrow.utcnow()
        self.t2 = now
        self.t1 = now.shift(days=-self.interval)
        self.engine()
        self.rounds_ran += 1
        self.runtime = arrow.utcnow() - now
        self.total_runtime += self.runtime
        self.print_stats()
        now = arrow.utcnow()
        user_cleanup()
        meme_cleanup()
        # create_clans(50)
        comment_cleanup()
        clear()
        end = arrow.utcnow()
        print(f"Total exec time - {end-start}")
        print(f"DB Seed exec time - {now-start}")
        print(f"Clean up exec time - {end-now}")

        # while True:
        #     now = arrow.utcnow()
        #     self.t2 = arrow.utcnow().shift(hours=-1)
        #     self.t1 = arrow.utcnow().shift(hours=-2)
        #     self.engine()
        #     self.rounds_ran += 1
        #     self.runtime = arrow.utcnow() - now
        #     self.total_runtime += self.runtime
        #     self.print_stats()
        #     input("Hit Enter to Run Round")

    def print_stats(self):
        if self.verbose:
            clear()
            print(
                f"""
            Current Time: {self.t1}\n
            Total Users: {db.session.query(User.id).count()}\n
            Total Memes: {db.session.query(Meme.id).count()}\n
            Total Meme Votes: {db.session.query(MemeVote.userId).count()}\n
            Total Comments: {db.session.query(Comment.id).count()}\n
            Total Comment Votes: {db.session.query(CommentVote.userId).count()}\n\n
            Rounds Ran: {self.rounds_ran}\n\n
            Round Runtime: {self.runtime}\n
            Total Runtime: {self.total_runtime}
            """,
                end="",
            )
