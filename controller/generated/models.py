# coding: utf-8
from controller.reddit.functions.misc import dump_datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql.base import DOUBLE_PRECISION

db = SQLAlchemy()


class Clan(db.Model):
    __tablename__ = "clans"

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    creatorId = db.Column(db.Integer, nullable=False)
    name = db.Column(db.String, nullable=False)
    size = db.Column(db.Integer, nullable=False, server_default=db.FetchedValue())
    createdAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())
    updatedAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())


class CommentVote(db.Model):
    __tablename__ = "comment_vote"

    userId = db.Column(
        db.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    commentId = db.Column(
        db.ForeignKey("comments.id", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    upvote = db.Column(db.Boolean, nullable=False)
    createdAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())

    comment = db.relationship(
        "Comment",
        primaryjoin="CommentVote.commentId == Comment.id",
        backref="comment_votes",
    )
    user = db.relationship(
        "User", primaryjoin="CommentVote.userId == User.id", backref="comment_votes"
    )


class Comment(db.Model):
    __tablename__ = "comments"

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    text = db.Column(db.String, nullable=False)
    userId = db.Column(db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    memeId = db.Column(db.ForeignKey("memes.id", ondelete="CASCADE"))
    ups = db.Column(db.Integer, nullable=False, server_default=db.FetchedValue())
    downs = db.Column(db.Integer, nullable=False, server_default=db.FetchedValue())
    ratio = db.Column(db.Float(53), nullable=False, server_default=db.FetchedValue())
    createdAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())
    updatedAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())

    meme = db.relationship(
        "Meme", primaryjoin="Comment.memeId == Meme.id", backref="comments"
    )
    user = db.relationship(
        "User", primaryjoin="Comment.userId == User.id", backref="comments"
    )


class Contest(db.Model):
    __tablename__ = "contests"

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    title = db.Column(db.String, nullable=False)
    createdAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())
    updatedAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())


class Follow(db.Model):
    __tablename__ = "follows"

    followerId = db.Column(
        db.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    followingId = db.Column(
        db.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    createdAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())

    user = db.relationship(
        "User", primaryjoin="Follow.followerId == User.id", backref="user_follows"
    )
    user1 = db.relationship(
        "User", primaryjoin="Follow.followingId == User.id", backref="user_follows_0"
    )


class Ito(db.Model):
    __tablename__ = "itos"

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    title = db.Column(db.String, nullable=False)
    createdAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())
    updatedAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())


class MemeVote(db.Model):
    __tablename__ = "meme_vote"

    userId = db.Column(
        db.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    memeId = db.Column(
        db.ForeignKey("memes.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    upvote = db.Column(db.Boolean, nullable=False)
    createdAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())

    meme = db.relationship(
        "Meme", primaryjoin="MemeVote.memeId == Meme.id", backref="meme_votes"
    )
    user = db.relationship(
        "User", primaryjoin="MemeVote.userId == User.id", backref="meme_votes"
    )


class Meme(db.Model):
    __tablename__ = "memes"

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    url = db.Column(db.String, nullable=False, unique=True)
    templateId = db.Column(db.ForeignKey("templates.id", ondelete="CASCADE"))
    baseTemplateId = db.Column(db.Integer)
    contestId = db.Column(db.ForeignKey("contests.id"))
    userId = db.Column(db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    season = db.Column(db.Integer)
    clanId = db.Column(db.ForeignKey("clans.id"))
    community = db.Column(db.String)
    numComments = db.Column(
        db.Integer, nullable=False, server_default=db.FetchedValue()
    )
    ups = db.Column(db.Integer, nullable=False, server_default=db.FetchedValue())
    downs = db.Column(db.Integer, nullable=False, server_default=db.FetchedValue())
    ratio = db.Column(db.Float(53), nullable=False, server_default=db.FetchedValue())
    createdAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())
    updatedAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())
    title = db.Column(db.String)

    clan = db.relationship(
        "Clan", primaryjoin="Meme.clanId == Clan.id", backref="memes"
    )
    contest = db.relationship(
        "Contest", primaryjoin="Meme.contestId == Contest.id", backref="memes"
    )
    template = db.relationship(
        "Template", primaryjoin="Meme.templateId == Template.id", backref="memes"
    )
    user = db.relationship(
        "User", primaryjoin="Meme.userId == User.id", backref="memes"
    )


class Migration(db.Model):
    __tablename__ = "migrations"

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    timestamp = db.Column(db.BigInteger, nullable=False)
    name = db.Column(db.String, nullable=False)


class Rank(db.Model):
    __tablename__ = "rank"

    createdAt = db.Column(db.DateTime, primary_key=True, nullable=False)
    userId = db.Column(db.Integer, primary_key=True, nullable=False)
    rank = db.Column(db.Integer, nullable=False)
    totalPoints = db.Column(db.Integer, nullable=False)


class RedditMeme(db.Model):
    __tablename__ = "reddit_memes"

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    username = db.Column(db.String(20), nullable=False)
    reddit_id = db.Column(db.String(20), nullable=False)
    subreddit = db.Column(db.String(50), nullable=False)
    title = db.Column(db.String(500), nullable=False)
    url = db.Column(db.String(1000), nullable=False)
    meme_text = db.Column(db.String(1000000))
    template = db.Column(db.String(100))
    timestamp = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False)
    upvote_ratio = db.Column(db.Float(53), nullable=False)
    upvotes = db.Column(db.Integer, nullable=False)
    downvotes = db.Column(db.Integer, nullable=False)
    num_comments = db.Column(db.Integer, nullable=False)
    features = db.Column(db.ARRAY(DOUBLE_PRECISION(precision=53)))
    redditorId = db.Column(db.ForeignKey("redditors.id"))
    redditor_id = db.Column(db.Integer)

    redditor = db.relationship(
        "Redditor",
        primaryjoin="RedditMeme.redditorId == Redditor.id",
        backref="reddit_memes",
    )

    @property
    def serialize(self):
        return {
            "reddit_id": self.reddit_id,
            "subreddit": self.subreddit,
            "title": self.title,
            "username": self.username,
            "url": self.url,
            "meme_text": self.meme_text,
            "template": self.template,
            "timestamp": self.timestamp,
            "datetime": dump_datetime(self.datetime),
            "upvote_ratio": self.upvote_ratio,
            "upvotes": self.upvotes,
            "downvotes": self.downvotes,
            "num_comments": self.num_comments,
        }


class RedditScore(db.Model):
    __tablename__ = "reddit_scores"

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    username = db.Column(db.String(20), nullable=False)
    subreddit = db.Column(db.String(50), nullable=False)
    time_delta = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.Integer, nullable=False)
    datetime = db.Column(db.DateTime, nullable=False)
    final_score = db.Column(db.Float(53), nullable=False)
    raw_score = db.Column(db.Float(53), nullable=False)
    num_in_bottom = db.Column(db.Integer, nullable=False)
    num_in_top = db.Column(db.Integer, nullable=False)
    shitposter_index = db.Column(db.Float(53), nullable=False)
    highest_upvotes = db.Column(db.Integer, nullable=False)
    hu_score = db.Column(db.Float(53), nullable=False)
    lowest_ratio = db.Column(db.Float(53), nullable=False)
    redditorId = db.Column(db.ForeignKey("redditors.id"))
    redditor_id = db.Column(db.Integer)

    redditor = db.relationship(
        "Redditor",
        primaryjoin="RedditScore.redditorId == Redditor.id",
        backref="reddit_scores",
    )

    @property
    def serialize(self):
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


class Redditor(db.Model):
    __tablename__ = "redditors"

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    username = db.Column(db.String(20), nullable=False, unique=True)


class Template(db.Model):
    __tablename__ = "templates"

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    name = db.Column(db.String, nullable=False)
    baseMemeId = db.Column(db.Integer, nullable=False)
    itoId = db.Column(db.ForeignKey("itos.id", ondelete="CASCADE"))
    season = db.Column(db.Integer)
    isStonk = db.Column(db.Boolean, nullable=False, server_default=db.FetchedValue())
    createdAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())
    updatedAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())

    ito = db.relationship(
        "Ito", primaryjoin="Template.itoId == Ito.id", backref="templates"
    )


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    isHive = db.Column(db.Boolean, nullable=False, server_default=db.FetchedValue())
    email = db.Column(db.String, unique=True)
    username = db.Column(db.String, nullable=False, unique=True)
    avatar = db.Column(db.String, nullable=False, server_default=db.FetchedValue())
    clanCreatedId = db.Column(db.Integer)
    rankId = db.Column(db.Integer)
    clanId = db.Column(db.ForeignKey("clans.id", ondelete="CASCADE"))
    createdAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())
    updatedAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())
    password = db.Column(db.String)
    numFollowing = db.Column(
        db.Integer, nullable=False, server_default=db.FetchedValue()
    )
    numFollowers = db.Column(
        db.Integer, nullable=False, server_default=db.FetchedValue()
    )
    numMemeVotesGiven = db.Column(
        db.Integer, nullable=False, server_default=db.FetchedValue()
    )
    numMemeUpvotesRecieved = db.Column(
        db.Integer, nullable=False, server_default=db.FetchedValue()
    )
    numMemeDownvotesRecieved = db.Column(
        db.Integer, nullable=False, server_default=db.FetchedValue()
    )
    numMemeCommentsRecieved = db.Column(
        db.Integer, nullable=False, server_default=db.FetchedValue()
    )
    numCommentVotesGiven = db.Column(
        db.Integer, nullable=False, server_default=db.FetchedValue()
    )
    numCommentUpvotesRecieved = db.Column(
        db.Integer, nullable=False, server_default=db.FetchedValue()
    )
    numCommentDownvotesRecieved = db.Column(
        db.Integer, nullable=False, server_default=db.FetchedValue()
    )
    totalPoints = db.Column(
        db.Integer, nullable=False, server_default=db.FetchedValue()
    )

    clan = db.relationship(
        "Clan", primaryjoin="User.clanId == Clan.id", backref="users"
    )


class Wager(db.Model):
    __tablename__ = "wagers"

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    market = db.Column(db.String, nullable=False)
    userId = db.Column(db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    position = db.Column(db.Integer, nullable=False)
    entry = db.Column(db.Float(53), nullable=False)
    closedAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())
    updatedAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())
    exit = db.Column(db.Float(53), nullable=False)

    user = db.relationship(
        "User", primaryjoin="Wager.userId == User.id", backref="wagers"
    )
