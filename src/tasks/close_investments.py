from typing import List, cast

import arrow
import pandas as pd
from arrow.arrow import Arrow
from sqlalchemy.sql.elements import ClauseElement
from sqlalchemy.sql.expression import and_
from src.celery_app import CELERY
from src.generated.models import Investment, RedditMeme, User
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
                        < cast(Arrow, arrow.utcnow()).shift(days=-1).datetime,
                    ),
                    cast(
                        ClauseElement,
                        RedditMeme.created_at
                        > cast(Arrow, arrow.utcnow()).shift(days=-2).datetime,
                    ),
                )
            )
            .statement,
        ),
        site_db.bind,
        columns=["reddit_id", "upvotes"],
    )
    df["percentile"] = df["upvotes"].rank(pct=True)
    openInvestments = site_db.query(Investment).filter(
        and_(
            cast(
                ClauseElement,
                Investment.redditId.in_(cast(List[str], df["reddit_id"].values)),
            ),
            cast(ClauseElement, Investment.percentile == None),
        )
    )
    for investment in openInvestments:
        if user := site_db.query(User).get(cast(str, investment.userId)):
            percentiles = df.loc[df["reddit_id"] == investment.redditId, ["percentile"]]
            percentile = percentiles.values[0]
            investment.percentile = percentile
            if percentile >= investment.target:
                profit = round(investment.betSize / (1 - investment.target))
                investment.profitLoss = profit  # type: ignore
                user.gbp += profit  # type: ignore
            else:
                investment.profitLoss = -investment.betSize  # type: ignore
    site_db.commit()
