# coding: utf-8
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Clan(db.Model):
    __tablename__ = "clans"
    __bind_key__ = "sitedata"

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    creatorId = db.Column(db.Integer, nullable=False)
    name = db.Column(db.String, nullable=False)
    size = db.Column(db.Integer, nullable=False, server_default=db.FetchedValue())
    createdAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())
    updatedAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())


class CommentVote(db.Model):
    __tablename__ = "comment_vote"
    __bind_key__ = "sitedata"

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
    __bind_key__ = "sitedata"

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
    __bind_key__ = "sitedata"

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    title = db.Column(db.String, nullable=False)
    createdAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())
    updatedAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())


class Follow(db.Model):
    __tablename__ = "follows"
    __bind_key__ = "sitedata"

    followerId = db.Column(
        db.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    followingId = db.Column(
        db.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, nullable=False
    )
    createdAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())

    following = db.relationship(
        "User", primaryjoin="Follow.followerId == User.id", backref="following"
    )
    followers = db.relationship(
        "User", primaryjoin="Follow.followingId == User.id", backref="followers"
    )


class Ito(db.Model):
    __tablename__ = "itos"
    __bind_key__ = "sitedata"

    id = db.Column(db.Integer, primary_key=True, server_default=db.FetchedValue())
    title = db.Column(db.String, nullable=False)
    createdAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())
    updatedAt = db.Column(db.DateTime, nullable=False, server_default=db.FetchedValue())


class MemeVote(db.Model):
    __tablename__ = "meme_vote"
    __bind_key__ = "sitedata"

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
    __bind_key__ = "sitedata"

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


class Rank(db.Model):
    __tablename__ = "rank"
    __bind_key__ = "sitedata"

    createdAt = db.Column(db.DateTime, primary_key=True, nullable=False)
    userId = db.Column(db.Integer, primary_key=True, nullable=False)
    rank = db.Column(db.Integer, nullable=False)
    totalPoints = db.Column(db.Integer, nullable=False)


class Template(db.Model):
    __tablename__ = "templates"
    __bind_key__ = "sitedata"

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
    __bind_key__ = "sitedata"

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
    __bind_key__ = "sitedata"

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
