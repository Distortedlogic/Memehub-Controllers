from typing import cast

import pandas as pd
from sqlalchemy.sql.elements import ClauseElement, and_
from src.constants import LOAD_MEME_CLF_VERSION
from src.generated.models import RedditMeme
from src.session import site_db
from src.utils.display import display_df


def print_stats(name: str = ""):
    if name:
        correct_stonk = (
            site_db.query(RedditMeme)
            .filter(
                and_(
                    cast(ClauseElement, RedditMeme.version == LOAD_MEME_CLF_VERSION),
                    cast(ClauseElement, RedditMeme.stonk_correct == True),
                    cast(ClauseElement, RedditMeme.meme_clf == name),
                )
            )
            .count()
        )
        wrong_stonk = (
            site_db.query(RedditMeme)
            .filter(
                and_(
                    cast(ClauseElement, RedditMeme.version == LOAD_MEME_CLF_VERSION),
                    cast(ClauseElement, RedditMeme.stonk_correct == False),
                    cast(ClauseElement, RedditMeme.meme_clf == name),
                )
            )
            .count()
        )
        num_posts = (
            site_db.query(RedditMeme)
            .filter(
                and_(
                    cast(ClauseElement, RedditMeme.version == LOAD_MEME_CLF_VERSION),
                    cast(ClauseElement, RedditMeme.meme_clf == name),
                    cast(ClauseElement, RedditMeme.stonk == True),
                )
            )
            .count()
        )
    else:
        correct_stonk = 0
        wrong_stonk = 0
        num_posts = 0
    correct_stonks = (
        site_db.query(RedditMeme)
        .filter(
            and_(
                cast(ClauseElement, RedditMeme.version == LOAD_MEME_CLF_VERSION),
                cast(ClauseElement, RedditMeme.stonk_correct == True),
            )
        )
        .count()
    )
    wrong_stonks = (
        site_db.query(RedditMeme)
        .filter(
            and_(
                cast(ClauseElement, RedditMeme.version == LOAD_MEME_CLF_VERSION),
                cast(ClauseElement, RedditMeme.stonk_correct == False),
            )
        )
        .count()
    )
    correct_meme_clf = (
        site_db.query(RedditMeme)
        .filter(
            and_(
                cast(ClauseElement, RedditMeme.version == LOAD_MEME_CLF_VERSION),
                cast(ClauseElement, RedditMeme.meme_clf_correct == True),
            )
        )
        .count()
    )
    wrong_meme_clf = (
        site_db.query(RedditMeme)
        .filter(
            and_(
                cast(ClauseElement, RedditMeme.version == LOAD_MEME_CLF_VERSION),
                cast(ClauseElement, RedditMeme.meme_clf_correct == False),
            )
        )
        .count()
    )
    memes_classified = (
        site_db.query(RedditMeme)
        .filter(cast(ClauseElement, RedditMeme.version == LOAD_MEME_CLF_VERSION))
        .count()
    )
    memes_unclassified = (
        site_db.query(RedditMeme)
        .filter(cast(ClauseElement, RedditMeme.version == None))
        .count()
    )
    display_df(
        pd.DataFrame.from_records(
            [
                dict(
                    num_posts=num_posts,
                    correct_stonk=correct_stonk,
                    wrong_stonk=wrong_stonk,
                    memes_classified=memes_classified,
                    memes_unclassified=memes_unclassified,
                    correct_stonks=correct_stonks,
                    wrong_stonks=wrong_stonks,
                    correct_meme_clf=correct_meme_clf,
                    wrong_meme_clf=wrong_meme_clf,
                )
            ]
        )
    )
