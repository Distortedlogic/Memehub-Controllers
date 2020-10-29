import random
import uuid

import arrow
import bcrypt
from controller.generated.models import (
    Comment,
    CommentVote,
    Follow,
    Meme,
    MemeVote,
    User,
    db,
)
from faker import Faker
from sqlalchemy import func
from sqlalchemy.sql.expression import and_

fake = Faker()
Faker.seed(42)


def roll_dice(prob):
    return random.random() < prob


def init_db():
    db.session.query(Follow).delete()
    db.session.commit()
    db.session.query(CommentVote).delete()
    db.session.commit()
    db.session.query(Comment).delete()
    db.session.commit()
    db.session.query(MemeVote).delete()
    db.session.commit()
    db.session.query(Meme).delete()
    db.session.commit()
    db.session.query(User).delete()
    db.session.commit()

    db.session.add(
        User(
            id=str(uuid.uuid4()),
            username="jermeek",
            email="jermeek@gmail.com",
            createdAt=arrow.utcnow().shift(days=-40).naive,
            avatar=fake.image_url(),
            password=bcrypt.hashpw("123456".encode("utf-8"), bcrypt.gensalt(10)),
        )
    )
    db.session.commit()
    userId = db.session.query(User).first().id
    db.session.add(
        Meme(
            id=str(uuid.uuid4()),
            userId=userId,
            url=fake.image_url(),
            createdAt=arrow.utcnow().shift(days=-40).naive,
        )
    )
    db.session.commit()
    memeId = db.session.query(Meme).first().id
    db.session.add(
        Comment(
            id=str(uuid.uuid4()),
            userId=userId,
            memeId=memeId,
            text="init",
            createdAt=arrow.utcnow().shift(days=-40).naive,
        )
    )
    db.session.commit()


def create_user(t1, t2):
    profile = fake.profile(fields=["username", "mail"])
    if (
        not db.session.query(User).filter(User.username == profile["username"]).first()
        and not db.session.query(User).filter(User.email == profile["mail"]).first()
    ):
        db.session.add(
            User(
                id=str(uuid.uuid4()),
                username=profile["username"],
                email=profile["mail"],
                createdAt=fake.date_time_between(t1.naive, t2.naive),
                avatar=fake.image_url(),
                password=bcrypt.hashpw(
                    fake.password().encode("utf-8"), bcrypt.gensalt(10)
                ),
            )
        )
        db.session.commit()


def follow_user(followerId, followingId, t1, t2):
    follow = Follow(
        followerId=followerId,
        followingId=followingId,
        createdAt=fake.date_time_between(t1.naive, t2.naive),
    )
    db.session.add(follow)


def post_meme(userId, url, t1, t2):
    db.session.add(
        Meme(
            id=str(uuid.uuid4()),
            title=fake.text(max_nb_chars=50),
            community=random.choice(
                [
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    "original",
                    "hive",
                    "wholesome",
                    "dark",
                    "political",
                ]
            ),
            userId=userId,
            url=url,
            createdAt=fake.date_time_between(t1.naive, t2.naive),
        )
    )


def vote_meme(userId, memeId, t1, t2):
    db.session.add(
        MemeVote(
            userId=userId,
            memeId=memeId,
            upvote=roll_dice(0.75),
            createdAt=fake.date_time_between(t1.naive, t2.naive),
        )
    )


def post_comment(userId, memeId, t1, t2):
    db.session.add(
        Comment(
            id=str(uuid.uuid4()),
            userId=userId,
            memeId=memeId,
            text=fake.text(),
            createdAt=fake.date_time_between(t1.naive, t2.naive),
        )
    )


def vote_comment(userId, commentId, t1, t2):
    db.session.add(
        CommentVote(
            userId=userId,
            commentId=commentId,
            createdAt=fake.date_time_between(t1.naive, t2.naive),
            upvote=roll_dice(0.75),
        )
    )


action_to_points = dict(
    memeVoteGiven=1,
    memeUpvoteRecieved=10,
    memeDownvoteRecieved=-12,
    memeCommentRecieved=2,
    commentVoteGiven=1,
    commentUpvoteRecieved=5,
    commentDownvoteRecieved=-6,
    followRecieved=7,
)


def user_cleanup():
    for user in db.session.query(User):
        user.numFollowers = (
            db.session.query(Follow).filter(Follow.followingId == user.id).count()
        )
        user.numFollowing = (
            db.session.query(Follow).filter(Follow.followerId == user.id).count()
        )
        user.numMemeVotesGiven = (
            db.session.query(MemeVote).filter_by(userId=user.id).count()
        )
        user.numMemeUpvotesRecieved = (
            db.session.query(MemeVote)
            .join(MemeVote.meme)
            .filter(and_(Meme.userId == user.id, MemeVote.upvote == True))
            .count()
        )
        user.numMemeDownvotesRecieved = (
            db.session.query(MemeVote)
            .join(MemeVote.meme)
            .filter(and_(Meme.userId == user.id, MemeVote.upvote == False))
            .count()
        )
        user.numMemeCommentsRecieved = (
            db.session.query(Comment)
            .join(Comment.meme)
            .filter(Meme.userId == user.id)
            .count()
        )
        user.numCommentVotesGiven = (
            db.session.query(CommentVote).filter_by(userId=user.id).count()
        )
        user.numCommentUpvotesRecieved = (
            db.session.query(CommentVote)
            .join(CommentVote.comment)
            .filter(and_(Comment.userId == user.id, CommentVote.upvote == True))
            .count()
        )
        user.numCommentDownvotesRecieved = (
            db.session.query(CommentVote)
            .join(CommentVote.comment)
            .filter(and_(Comment.userId == user.id, CommentVote.upvote == False))
            .count()
        )
        user.totalPoints = (
            user.numMemeVotesGiven * action_to_points["memeVoteGiven"]
            + user.numMemeUpvotesRecieved * action_to_points["memeUpvoteRecieved"]
            + user.numMemeDownvotesRecieved * action_to_points["memeDownvoteRecieved"]
            + user.numMemeCommentsRecieved * action_to_points["memeCommentRecieved"]
            + user.numCommentVotesGiven * action_to_points["commentVoteGiven"]
            + user.numCommentUpvotesRecieved * action_to_points["commentUpvoteRecieved"]
            + user.numCommentDownvotesRecieved
            * action_to_points["commentDownvoteRecieved"]
        )
    db.session.commit()


def meme_cleanup():
    for meme in db.session.query(Meme):
        meme.ups = (
            db.session.query(MemeVote.upvote)
            .filter_by(memeId=meme.id, upvote=True)
            .count()
        )
        meme.downs = (
            db.session.query(MemeVote.upvote)
            .filter_by(memeId=meme.id, upvote=False)
            .count()
        )
        meme.numComments = (
            db.session.query(Comment.id).filter_by(memeId=meme.id).count()
        )
        try:
            meme.ratio = round(meme.ups / (meme.ups + meme.downs), 3)
        except:
            meme.ratio = 1
    db.session.commit()


def comment_cleanup():
    for comment in db.session.query(Comment):
        comment.ups = (
            db.session.query(CommentVote.upvote)
            .filter_by(commentId=comment.id, upvote=True)
            .count()
        )
        comment.downs = (
            db.session.query(CommentVote.upvote)
            .filter_by(commentId=comment.id, upvote=False)
            .count()
        )
        try:
            comment.ratio = round(comment.ups / (comment.ups + comment.downs), 3)
        except:
            comment.ratio = 1
    db.session.commit()
