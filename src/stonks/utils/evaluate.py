from typing import cast

import pandas as pd
from sqlalchemy.sql.elements import ClauseElement, and_
from src.constants import LOAD_MEME_CLF_VERSION
from src.generated.models import RedditMeme
from src.session import site_db
from src.utils.display import display_df


def evaluated():
    num_filled = (
        site_db.query(RedditMeme)
        .filter(
            and_(
                cast(ClauseElement, RedditMeme.version == LOAD_MEME_CLF_VERSION),
                cast(ClauseElement, RedditMeme.is_a_template_official != None),
            )
        )
        .count()
    )
    total_official = (
        site_db.query(RedditMeme)
        .filter(cast(ClauseElement, RedditMeme.is_a_template_official != None))
        .count()
    )
    num_stonks = (
        site_db.query(RedditMeme)
        .filter(cast(ClauseElement, RedditMeme.stonk_official != None))
        .count()
    )
    num_not_template = (
        site_db.query(RedditMeme)
        .filter(cast(ClauseElement, RedditMeme.is_a_template_official == False))
        .count()
    )
    num_not_template_wrong = (
        site_db.query(RedditMeme)
        .filter(
            and_(
                cast(ClauseElement, RedditMeme.version == LOAD_MEME_CLF_VERSION),
                cast(ClauseElement, RedditMeme.stonk == True),
                cast(ClauseElement, RedditMeme.is_a_template_official == False),
            )
        )
        .count()
    )
    tp = (
        site_db.query(RedditMeme)
        .filter(
            and_(
                cast(ClauseElement, RedditMeme.meme_clf == RedditMeme.stonk_official),
                cast(ClauseElement, RedditMeme.stonk == True),
                cast(ClauseElement, RedditMeme.stonk_official != None),
                cast(ClauseElement, RedditMeme.version == LOAD_MEME_CLF_VERSION),
            )
        )
        .count()
    )
    fn = (
        site_db.query(RedditMeme)
        .filter(
            and_(
                cast(ClauseElement, RedditMeme.meme_clf == RedditMeme.stonk_official),
                cast(ClauseElement, RedditMeme.stonk == False),
                cast(ClauseElement, RedditMeme.stonk_official != None),
                cast(ClauseElement, RedditMeme.version == LOAD_MEME_CLF_VERSION),
            )
        )
        .count()
    )
    meme_clf_wrong = (
        site_db.query(RedditMeme)
        .filter(
            and_(
                cast(ClauseElement, RedditMeme.meme_clf != RedditMeme.stonk_official),
                cast(ClauseElement, RedditMeme.stonk_official != None),
                cast(ClauseElement, RedditMeme.version == LOAD_MEME_CLF_VERSION),
            )
        )
        .count()
    )
    meme_clf_wrong_stonk_right = (
        site_db.query(RedditMeme)
        .filter(
            and_(
                cast(ClauseElement, RedditMeme.meme_clf != RedditMeme.stonk_official),
                cast(ClauseElement, RedditMeme.stonk == False),
                cast(ClauseElement, RedditMeme.stonk_official != None),
                cast(ClauseElement, RedditMeme.version == LOAD_MEME_CLF_VERSION),
            )
        )
        .count()
    )
    meme_clf_wrong_stonk_wrong = (
        site_db.query(RedditMeme)
        .filter(
            and_(
                cast(ClauseElement, RedditMeme.meme_clf != RedditMeme.stonk_official),
                cast(ClauseElement, RedditMeme.stonk == True),
                cast(ClauseElement, RedditMeme.stonk_official != None),
                cast(ClauseElement, RedditMeme.version == LOAD_MEME_CLF_VERSION),
            )
        )
        .count()
    )
    display_df(
        pd.DataFrame.from_records(
            [
                dict(
                    num_stonks=num_stonks,
                    num_not_template=num_not_template,
                    total_official=total_official,
                    num_filled=num_filled,
                    percent_done=round(num_filled / total_official, 3),
                )
            ]
        )
    )
    display_df(
        pd.DataFrame.from_records(
            [
                dict(
                    num_not_template_wrong=num_not_template_wrong,
                    num_nt_wrong_percent=round(
                        num_not_template_wrong / num_not_template, 3
                    ),
                    true_positives=tp,
                    tp_percent=round(tp / num_stonks, 3),
                    false_negatives=fn,
                    fn_percent=round(fn / num_stonks, 3),
                )
            ]
        )
    )
    display_df(
        pd.DataFrame.from_records(
            [
                dict(
                    meme_clf_wrong=meme_clf_wrong,
                    wrong_right=meme_clf_wrong_stonk_right,
                    wr_percent=round(fn / meme_clf_wrong, 3),
                    wrong_wrong=meme_clf_wrong_stonk_wrong,
                    ww_percent=round(fn / meme_clf_wrong, 3),
                )
            ]
        )
    )
