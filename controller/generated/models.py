# coding: utf-8
from typing import Any, Dict

from controller.reddit.functions.misc import dump_datetime
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.dialects.postgresql.base import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.schema import FetchedValue

Base: Any = declarative_base()


class CommentVote(Base):
    __tablename__ = "comment_votes"

    userId = Column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    commentId = Column(
        ForeignKey("comments.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    upvote = Column(Boolean, nullable=False)
    createdAt = Column(DateTime, nullable=False, server_default=FetchedValue())

    comment = relationship(
        "Comment",
        primaryjoin="CommentVote.commentId == Comment.id",
        backref="comment_votes",
    )
    user = relationship(
        "User", primaryjoin="CommentVote.userId == User.id", backref="comment_votes"
    )


class Comment(Base):
    __tablename__ = "comments"

    id = Column(UUID, primary_key=True)
    text = Column(String, nullable=False)
    isHive = Column(Boolean, nullable=False, server_default=FetchedValue())
    permlink = Column(String)
    userId = Column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    memeId = Column(ForeignKey("memes.id", ondelete="CASCADE"))
    ups = Column(Integer, nullable=False, server_default=FetchedValue())
    downs = Column(Integer, nullable=False, server_default=FetchedValue())
    ratio = Column(Float(53), nullable=False, server_default=FetchedValue())
    createdAt = Column(DateTime, nullable=False, server_default=FetchedValue())
    updatedAt = Column(DateTime, nullable=False, server_default=FetchedValue())

    meme = relationship(
        "Meme", primaryjoin="Comment.memeId == Meme.id", backref="comments"
    )
    user = relationship(
        "User", primaryjoin="Comment.userId == User.id", backref="comments"
    )


class Emoji(Base):
    __tablename__ = "emojis"

    name = Column(String, primary_key=True)
    url = Column(String, nullable=False)
    createdAt = Column(DateTime, nullable=False, server_default=FetchedValue())


class Follow(Base):
    __tablename__ = "follows"

    followerId = Column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    followingId = Column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    createdAt = Column(DateTime, nullable=False, server_default=FetchedValue())

    user = relationship(
        "User", primaryjoin="Follow.followerId == User.id", backref="user_follows"
    )
    user1 = relationship(
        "User", primaryjoin="Follow.followingId == User.id", backref="user_follows_0"
    )


class MemeVote(Base):
    __tablename__ = "meme_votes"

    userId = Column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    memeId = Column(
        ForeignKey("memes.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    upvote = Column(Boolean, nullable=False)
    createdAt = Column(DateTime, nullable=False, server_default=FetchedValue())

    meme = relationship(
        "Meme", primaryjoin="MemeVote.memeId == Meme.id", backref="meme_votes"
    )
    user = relationship(
        "User", primaryjoin="MemeVote.userId == User.id", backref="meme_votes"
    )


class Meme(Base):
    __tablename__ = "memes"

    id = Column(UUID, primary_key=True)
    isHive = Column(Boolean, nullable=False, server_default=FetchedValue())
    title = Column(String)
    url = Column(String, nullable=False)
    userId = Column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    season = Column(Integer)
    community = Column(String, nullable=False)
    numComments = Column(Integer, nullable=False, server_default=FetchedValue())
    ups = Column(Integer, nullable=False, server_default=FetchedValue())
    downs = Column(Integer, nullable=False, server_default=FetchedValue())
    ratio = Column(Float(53), nullable=False, server_default=FetchedValue())
    createdAt = Column(DateTime, nullable=False, server_default=FetchedValue())
    updatedAt = Column(DateTime, nullable=False, server_default=FetchedValue())
    ocrText = Column(String)
    stonk = Column(String)
    version = Column(String)

    user = relationship("User", primaryjoin="Meme.userId == User.id", backref="memes")


class Migration(Base):
    __tablename__ = "migrations"

    id = Column(Integer, primary_key=True, server_default=FetchedValue())
    timestamp = Column(BigInteger, nullable=False)
    name = Column(String, nullable=False)


class Rank(Base):
    __tablename__ = "rank"

    createdAt = Column(DateTime, primary_key=True, nullable=False)
    userId = Column(String, primary_key=True, nullable=False)
    timeFrame = Column(String, primary_key=True, nullable=False)
    rank = Column(Integer, nullable=False)
    totalPoints = Column(Integer, nullable=False)


class RedditMeme(Base):
    __tablename__ = "reddit_memes"

    id = Column(Integer, primary_key=True, server_default=FetchedValue())
    username = Column(String(20), nullable=False)
    reddit_id = Column(String(20), nullable=False)
    subreddit = Column(String(50), nullable=False)
    title = Column(String(500), nullable=False)
    url = Column(String(1000), nullable=False)
    meme_text = Column(String(1000000))
    timestamp = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False)
    upvote_ratio = Column(Float(53), nullable=False)
    upvotes = Column(Integer, nullable=False)
    downvotes = Column(Integer, nullable=False)
    num_comments = Column(Integer, nullable=False)
    redditor_id = Column(Integer)
    redditorId = Column(ForeignKey("redditors.id"))
    version = Column(String(20))
    meme_clf = Column(String(100))
    meme_clf_correct = Column(Boolean)
    stonk = Column(Boolean)
    stonk_correct = Column(Boolean)
    is_a_template = Column(Boolean)

    redditor = relationship(
        "Redditor",
        primaryjoin="RedditMeme.redditorId == Redditor.id",
        backref="reddit_memes",
    )


class RedditScore(Base):
    __tablename__ = "reddit_scores"

    id = Column(Integer, primary_key=True, server_default=FetchedValue())
    username = Column(String(20), nullable=False)
    subreddit = Column(String(50), nullable=False)
    time_delta = Column(Integer, nullable=False)
    timestamp = Column(Integer, nullable=False)
    datetime = Column(DateTime, nullable=False)
    final_score = Column(Float(53), nullable=False)
    raw_score = Column(Float(53), nullable=False)
    num_in_bottom = Column(Integer, nullable=False)
    num_in_top = Column(Integer, nullable=False)
    shitposter_index = Column(Float(53), nullable=False)
    highest_upvotes = Column(Integer, nullable=False)
    hu_score = Column(Float(53), nullable=False)
    lowest_ratio = Column(Float(53), nullable=False)
    redditor_id = Column(Integer)
    redditorId = Column(ForeignKey("redditors.id"))

    redditor = relationship(
        "Redditor",
        primaryjoin="RedditScore.redditorId == Redditor.id",
        backref="reddit_scores",
    )

    @property
    def serialize(self) -> Dict[str, Any]:
        return {
            "username": self.username,
            "subreddit": self.subreddit,
            "timestamp": self.timestamp,
            "datetime": dump_datetime(self.datetime),
            "final_score": self.final_score,
            "raw_score": self.raw_score,
            "num_in_bottom": self.num_in_bottom,
            "num_in_top": self.num_in_top,
            "shitposter_index": self.shitposter_index,
            "highest_upvotes": self.highest_upvotes,
            "hu_score": self.hu_score,
            "lowest_ratio": self.lowest_ratio,
        }

    @property
    def stats(self):
        return {
            "username": self.username,
            "final_score": self.final_score,
            "num_in_bottom": self.num_in_bottom,
            "num_in_top": self.num_in_top,
            "shitposter_index": self.shitposter_index,
            "highest_upvotes": self.highest_upvotes,
            "hu_score": self.hu_score,
            "lowest_ratio": self.lowest_ratio,
        }


class Redditor(Base):
    __tablename__ = "redditors"

    id = Column(Integer, primary_key=True, server_default=FetchedValue())
    username = Column(String(20), nullable=False, unique=True)


class User(Base):
    __tablename__ = "users"

    id = Column(UUID, primary_key=True)
    isHive = Column(Boolean, nullable=False, server_default=FetchedValue())
    verified = Column(Boolean, nullable=False, server_default=FetchedValue())
    email = Column(String, unique=True)
    username = Column(String, nullable=False, unique=True)
    avatar = Column(String, nullable=False, server_default=FetchedValue())
    createdAt = Column(DateTime, nullable=False, server_default=FetchedValue())
    updatedAt = Column(DateTime, nullable=False, server_default=FetchedValue())
    password = Column(String)
    numFollowing = Column(Integer, nullable=False, server_default=FetchedValue())
    numFollowers = Column(Integer, nullable=False, server_default=FetchedValue())
    numMemeVotesGiven = Column(Integer, nullable=False, server_default=FetchedValue())
    numMemeUpvotesRecieved = Column(
        Integer, nullable=False, server_default=FetchedValue()
    )
    numMemeDownvotesRecieved = Column(
        Integer, nullable=False, server_default=FetchedValue()
    )
    numMemeCommentsRecieved = Column(
        Integer, nullable=False, server_default=FetchedValue()
    )
    numCommentVotesGiven = Column(
        Integer, nullable=False, server_default=FetchedValue()
    )
    numCommentUpvotesRecieved = Column(
        Integer, nullable=False, server_default=FetchedValue()
    )
    numCommentDownvotesRecieved = Column(
        Integer, nullable=False, server_default=FetchedValue()
    )
    totalPoints = Column(Integer, nullable=False, server_default=FetchedValue())
