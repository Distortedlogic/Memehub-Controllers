from typing import List, cast

import arrow
import pandas as pd
from arrow.arrow import Arrow
from sqlalchemy.sql.elements import ClauseElement
from sqlalchemy.sql.expression import and_
from src.celery_app import CELERY
from src.generated.models import Investment, RedditMeme
from src.session import site_db


@CELERY.task(name="CloseInvestments", unique_on=[], lock_expiry=60 * 60 * 12)
def close_investments():
    df = pd.read_sql(
        cast(
            str,
            site_db.query(RedditMeme.reddit_id, RedditMeme.upvotes)
            .filter(
                and_(
                    cast(
                        ClauseElement,
                        RedditMeme.created_at
                        < cast(Arrow, arrow.utcnow()).shift(days=-1),
                    ),
                    cast(
                        ClauseElement,
                        RedditMeme.created_at
                        > cast(Arrow, arrow.utcnow()).shift(days=-2),
                    ),
                )
            )
            .statement,
        ),
        site_db.bind,
        columns=["id", "upvotes"],
    )
    df["percentile"] = df["upvotes"].rank(pct=True)
    openInvestments = site_db.query(Investment).filter(
        cast(ClauseElement, Investment.redditId.in_(cast(List[str], df["id"].values)))
    )
