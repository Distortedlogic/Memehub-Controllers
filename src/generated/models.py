# coding: utf-8
from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, ForeignKey, Integer, MetaData, String
from sqlalchemy.dialects.postgresql.base import UUID
from sqlalchemy.schema import FetchedValue
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base


Base = declarative_base()
metadata = Base.metadata



class CommentVote(Base):
    __tablename__ = 'comment_votes'

    userId = Column(ForeignKey('users.id', ondelete='CASCADE'), primary_key=True, nullable=False)
    commentId = Column(ForeignKey('comments.id', ondelete='CASCADE'), primary_key=True, nullable=False)
    upvote = Column(Boolean, nullable=False)
    createdAt = Column(DateTime, nullable=False, server_default=FetchedValue())

    comment = relationship('Comment', primaryjoin='CommentVote.commentId == Comment.id', backref='comment_votes')
    user = relationship('User', primaryjoin='CommentVote.userId == User.id', backref='comment_votes')



class Comment(Base):
    __tablename__ = 'comments'

    id = Column(UUID, primary_key=True)
    text = Column(String, nullable=False)
    isHive = Column(Boolean, nullable=False, server_default=FetchedValue())
    permlink = Column(String)
    userId = Column(ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    memeId = Column(ForeignKey('memes.id', ondelete='CASCADE'))
    ups = Column(Integer, nullable=False, server_default=FetchedValue())
    downs = Column(Integer, nullable=False, server_default=FetchedValue())
    ratio = Column(Float(53), nullable=False, server_default=FetchedValue())
    createdAt = Column(DateTime, nullable=False, server_default=FetchedValue())
    updatedAt = Column(DateTime, nullable=False, server_default=FetchedValue())

    meme = relationship('Meme', primaryjoin='Comment.memeId == Meme.id', backref='comments')
    user = relationship('User', primaryjoin='Comment.userId == User.id', backref='comments')



class Emoji(Base):
    __tablename__ = 'emojis'

    id = Column(UUID, primary_key=True)
    name = Column(String, nullable=False)
    url = Column(String, nullable=False)
    createdAt = Column(DateTime, nullable=False, server_default=FetchedValue())



class Investment(Base):
    __tablename__ = 'investments'

    id = Column(UUID, primary_key=True)
    redditId = Column(String, nullable=False)
    betSize = Column(Integer, nullable=False)
    target = Column(Float(53), nullable=False)
    percentile = Column(Float(53))
    userId = Column(ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    season = Column(Integer)
    createdAt = Column(DateTime, nullable=False, server_default=FetchedValue())
    updatedAt = Column(DateTime, nullable=False, server_default=FetchedValue())
    profitLoss = Column(Integer)

    user = relationship('User', primaryjoin='Investment.userId == User.id', backref='investments')



class Market(Base):
    __tablename__ = 'market'

    id = Column(UUID, primary_key=True)
    name = Column(String, nullable=False)
    templateId = Column(ForeignKey('templates.id'), nullable=False)
    createdAt = Column(DateTime, nullable=False)
    numPosts = Column(Integer, nullable=False)
    numUpvotes = Column(Integer, nullable=False)

    template = relationship('Template', primaryjoin='Market.templateId == Template.id', backref='markets')



class MemeVote(Base):
    __tablename__ = 'meme_votes'

    userId = Column(ForeignKey('users.id', ondelete='CASCADE'), primary_key=True, nullable=False)
    memeId = Column(ForeignKey('memes.id', ondelete='CASCADE'), primary_key=True, nullable=False)
    upvote = Column(Boolean, nullable=False)
    createdAt = Column(DateTime, nullable=False, server_default=FetchedValue())

    meme = relationship('Meme', primaryjoin='MemeVote.memeId == Meme.id', backref='meme_votes')
    user = relationship('User', primaryjoin='MemeVote.userId == User.id', backref='meme_votes')



class Meme(Base):
    __tablename__ = 'memes'

    id = Column(UUID, primary_key=True)
    isHive = Column(Boolean, nullable=False, server_default=FetchedValue())
    title = Column(String)
    ocrText = Column(String)
    url = Column(String, nullable=False)
    userId = Column(ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    meme_clf = Column(String)
    meme_clf_correct = Column(Boolean)
    version = Column(String)
    season = Column(Integer)
    numComments = Column(Integer, nullable=False, server_default=FetchedValue())
    ups = Column(Integer, nullable=False, server_default=FetchedValue())
    downs = Column(Integer, nullable=False, server_default=FetchedValue())
    ratio = Column(Float(53), nullable=False, server_default=FetchedValue())
    createdAt = Column(DateTime, nullable=False, server_default=FetchedValue())
    updatedAt = Column(DateTime, nullable=False, server_default=FetchedValue())

    user = relationship('User', primaryjoin='Meme.userId == User.id', backref='memes')



class Migration(Base):
    __tablename__ = 'migrations'

    id = Column(Integer, primary_key=True, server_default=FetchedValue())
    timestamp = Column(BigInteger, nullable=False)
    name = Column(String, nullable=False)



class Rank(Base):
    __tablename__ = 'rank'

    createdAt = Column(DateTime, primary_key=True, nullable=False)
    userId = Column(String, primary_key=True, nullable=False)
    timeFrame = Column(String, primary_key=True, nullable=False)
    mhpRank = Column(Integer, nullable=False)
    mhp = Column(Integer, nullable=False)
    gbpRank = Column(Integer, nullable=False)
    gbp = Column(Integer, nullable=False)



class RedditMeme(Base):
    __tablename__ = 'reddit_memes'

    id = Column(Integer, primary_key=True, server_default=FetchedValue())
    username = Column(String(20), nullable=False)
    reddit_id = Column(String(20), nullable=False)
    subreddit = Column(String(50), nullable=False)
    title = Column(String(500), nullable=False)
    url = Column(String(1000), nullable=False, unique=True)
    meme_text = Column(String(1000000))
    is_a_template = Column(Boolean)
    meme_clf = Column(String(100))
    meme_clf_correct = Column(Boolean)
    stonk = Column(Boolean)
    stonk_correct = Column(Boolean)
    stonk_official = Column(String)
    is_a_template_official = Column(Boolean)
    version = Column(String(20))
    timestamp = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False)
    upvote_ratio = Column(Float(53), nullable=False)
    upvotes = Column(Integer, nullable=False)
    downvotes = Column(Integer, nullable=False)
    num_comments = Column(Integer, nullable=False)
    redditor_id = Column(Integer)
    redditorId = Column(ForeignKey('redditors.id'))
    percentile = Column(Float(53))

    redditor = relationship('Redditor', primaryjoin='RedditMeme.redditorId == Redditor.id', backref='reddit_memes')



class RedditScore(Base):
    __tablename__ = 'reddit_scores'

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
    redditorId = Column(ForeignKey('redditors.id'))

    redditor = relationship('Redditor', primaryjoin='RedditScore.redditorId == Redditor.id', backref='reddit_scores')



class Redditor(Base):
    __tablename__ = 'redditors'

    id = Column(Integer, primary_key=True, server_default=FetchedValue())
    username = Column(String(20), nullable=False, unique=True)



class Template(Base):
    __tablename__ = 'templates'

    id = Column(UUID, primary_key=True)
    name = Column(String, nullable=False)
    url = Column(String, nullable=False, unique=True)



class Trade(Base):
    __tablename__ = 'trades'

    id = Column(UUID, primary_key=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    price = Column(Integer, nullable=False)
    position = Column(Integer, nullable=False)
    userId = Column(ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    createdAt = Column(DateTime, nullable=False, server_default=FetchedValue())
    templateId = Column(ForeignKey('templates.id'))

    template = relationship('Template', primaryjoin='Trade.templateId == Template.id', backref='trades')
    user = relationship('User', primaryjoin='Trade.userId == User.id', backref='trades')



class UserMemeEmoji(Base):
    __tablename__ = 'user_meme_emojis'

    id = Column(UUID, primary_key=True)
    userId = Column(ForeignKey('users.id'), nullable=False)
    memeId = Column(ForeignKey('memes.id'), nullable=False)
    emojiId = Column(ForeignKey('emojis.id'), nullable=False)
    createdAt = Column(DateTime, nullable=False, server_default=FetchedValue())

    emoji = relationship('Emoji', primaryjoin='UserMemeEmoji.emojiId == Emoji.id', backref='user_meme_emojis')
    meme = relationship('Meme', primaryjoin='UserMemeEmoji.memeId == Meme.id', backref='user_meme_emojis')
    user = relationship('User', primaryjoin='UserMemeEmoji.userId == User.id', backref='user_meme_emojis')



class User(Base):
    __tablename__ = 'users'

    id = Column(UUID, primary_key=True)
    isHive = Column(Boolean, nullable=False, server_default=FetchedValue())
    verified = Column(Boolean, nullable=False, server_default=FetchedValue())
    email = Column(String, unique=True)
    username = Column(String, nullable=False, unique=True)
    avatar = Column(String, nullable=False, server_default=FetchedValue())
    createdAt = Column(DateTime, nullable=False, server_default=FetchedValue())
    updatedAt = Column(DateTime, nullable=False, server_default=FetchedValue())
    password = Column(String)
    numMemeVotesGiven = Column(Integer, nullable=False, server_default=FetchedValue())
    numMemeUpvotesRecieved = Column(Integer, nullable=False, server_default=FetchedValue())
    numMemeDownvotesRecieved = Column(Integer, nullable=False, server_default=FetchedValue())
    numMemeCommentsRecieved = Column(Integer, nullable=False, server_default=FetchedValue())
    numCommentVotesGiven = Column(Integer, nullable=False, server_default=FetchedValue())
    numCommentUpvotesRecieved = Column(Integer, nullable=False, server_default=FetchedValue())
    numCommentDownvotesRecieved = Column(Integer, nullable=False, server_default=FetchedValue())
    mhp = Column(Integer, nullable=False, server_default=FetchedValue())
    gbp = Column(Integer, nullable=False, server_default=FetchedValue())
    lastHivePost = Column(DateTime)
    lastMemehubPost = Column(DateTime)
