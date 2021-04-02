from typing import List, Union, cast

import numpy as np
import pandas as pd
from IPython.core.display import clear_output
from sqlalchemy import and_, func
from sqlalchemy.sql.elements import BooleanClauseList, ClauseElement, or_
from sqlalchemy.sql.expression import ClauseElement, case
from src.constants import LOAD_MEME_CLF_VERSION
from src.generated.models import RedditMeme
from src.session import site_db
from src.utils.auditor_stats import print_stats
from src.utils.display import display_df, display_meme, display_template
from src.utils.model_func import get_static_names


class Auditor:
    def __init__(self):
        self.prev_ids: List[int] = []

    def check_input(self, keypress: str):
        if "r" == keypress:
            return
        if "gb" in keypress:
            if (prev_meme := site_db.query(RedditMeme).get(self.prev_ids.pop())) :
                _ = display_meme(prev_meme)
                _ = self.update_meme(prev_meme)
        elif keypress == "q":
            return False
        return True

    def update_meme(self, meme: RedditMeme) -> Union[bool, None]:
        keypress = input("stonk correct?")
        if self.check_input(keypress) == False:
            return True
        elif self.check_input(keypress) == None:
            _ = self.update_meme(meme)
        elif keypress == "s":
            pass
        else:
            stonk_correct = not bool(keypress)
            meme.stonk_correct = stonk_correct  # type: ignore
            if stonk_correct:
                if meme.stonk:
                    meme.meme_clf_correct = True  # type: ignore
                    meme.is_a_template = True  # type: ignore
                else:
                    meme.meme_clf_correct = False  # type: ignore
                    keypress = input("is_a_template?")
                    if self.check_input(keypress) == False:
                        return False
                    elif self.check_input(keypress) == None:
                        _ = self.update_meme(meme)
                    else:
                        meme.is_a_template = not bool(keypress)  # type: ignore
            else:
                if meme.stonk:
                    meme.meme_clf_correct = False  # type: ignore
                    keypress = input("is_a_template?")
                    if self.check_input(keypress) == False:
                        return False
                    elif self.check_input(keypress) == None:
                        _ = self.update_meme(meme)
                    else:
                        meme.is_a_template = not bool(keypress)  # type: ignore
                else:
                    meme.meme_clf_correct = True  # type: ignore
                    meme.is_a_template = True  # type: ignore

    def audit(self, clause: BooleanClauseList):
        count = site_db.query(RedditMeme).filter(clause).count()
        for idx, meme in enumerate(site_db.query(RedditMeme).filter(clause)):
            clear_output()
            print_stats(meme.meme_clf)
            print("url", meme.url)
            print(f"name - {self.idx+1}/{self.len}")
            print(f"meme - {idx}/{count}")
            if not display_meme(meme):
                continue
            display_template(meme.meme_clf)
            if self.update_meme(meme):
                break
            site_db.commit()
            self.prev_ids.append(meme.id)

    @staticmethod
    def get_stonk_clause(name: str, correct: bool):
        return and_(
            cast(ClauseElement, RedditMeme.meme_clf == name),
            cast(ClauseElement, RedditMeme.stonk == correct),
            cast(ClauseElement, RedditMeme.version == LOAD_MEME_CLF_VERSION),
            cast(ClauseElement, RedditMeme.stonk_correct == None),
        )

    def run(self):
        orderby = input(
            """
        orderby
        i -image count
        p - num posts
        """
        )
        if orderby == "i":
            folder_count_df = pd.DataFrame(
                list(get_static_names(LOAD_MEME_CLF_VERSION)["folder_count"].items()),
                columns=["name", "count"],
            )
            names = cast(List[str], folder_count_df.sort_values("count")["name"])
        elif orderby == "p":
            names = [
                name
                for name, in site_db.query(RedditMeme.meme_clf)
                .group_by(RedditMeme.meme_clf)
                .filter(
                    cast(ClauseElement, RedditMeme.version == LOAD_MEME_CLF_VERSION)
                )
                .order_by(
                    func.count(
                        case([(cast(ClauseElement, RedditMeme.stonk == True), 1)])
                    ).asc()
                )
            ]
        else:
            raise Exception("Invalid")
        keypress = input(
            """
        What do u want to audit?
        c - correct
        i - incorrect
        """
        )
        if keypress == "c":
            correct = True
        elif keypress == "i":
            correct = False
        else:
            raise Exception("bad key press")
        self.len = len(names)
        for self.idx, name in enumerate(names):
            self.audit(self.get_stonk_clause(name, correct=correct))

    @staticmethod
    def set_official():
        for meme in site_db.query(RedditMeme).filter(
            and_(
                cast(ClauseElement, RedditMeme.stonk_correct != None),
                cast(ClauseElement, RedditMeme.is_a_template_official == None),
            )
        ):
            if meme.stonk_correct:
                meme.stonk_official = meme.meme_clf  # type: ignore
                meme.is_a_template_official = True  # type:ignore
            else:
                meme.is_a_template_official = meme.is_a_template  # type:ignore
        site_db.commit()

    @staticmethod
    def clear():
        for meme in site_db.query(RedditMeme).filter(
            cast(ClauseElement, RedditMeme.version != None)
        ):
            meme.version = None  # type: ignore
            meme.meme_clf = None  # type: ignore
            meme.meme_clf_correct = None  # type: ignore
            meme.stonk = None  # type: ignore
            meme.stonk_correct = None  # type: ignore
            meme.is_a_template = None  # type: ignore
        site_db.commit()

    def report(self):
        while keypress := input(
            """
        order by:
        p - num posts
        c - num correct
        w - num wrong
        +a for ASC
        """
        ):
            if keypress == "p":
                if "a" in keypress:
                    clause = func.count(RedditMeme.meme_clf).asc()
                else:
                    clause = func.count(RedditMeme.meme_clf).desc()
            elif keypress == "c":
                if "a" in keypress:
                    clause = func.count(
                        case([(cast(ClauseElement, RedditMeme.stonk_correct), 1)])
                    ).asc()
                else:
                    clause = func.count(
                        case([(cast(ClauseElement, RedditMeme.stonk_correct), 1)])
                    ).desc()
            elif keypress == "w":
                if "a" in keypress:
                    clause = func.count(
                        case(
                            [
                                (
                                    cast(
                                        ClauseElement, RedditMeme.stonk_correct == False
                                    ),
                                    1,
                                )
                            ]
                        )
                    ).asc()
                else:
                    clause = func.count(
                        case(
                            [
                                (
                                    cast(
                                        ClauseElement, RedditMeme.stonk_correct == False
                                    ),
                                    1,
                                )
                            ]
                        )
                    ).desc()
            else:
                raise Exception("nothing selected")
            clear_output()
            df = pd.read_sql(
                cast(
                    str,
                    site_db.query(
                        RedditMeme.meme_clf.label("name"),
                        func.count(RedditMeme.meme_clf).label("num_posts"),
                        func.count(
                            case(
                                [
                                    (
                                        cast(
                                            ClauseElement,
                                            RedditMeme.stonk_correct == True,
                                        ),
                                        1,
                                    )
                                ]
                            )
                        ).label("total_correct"),
                        func.count(
                            case(
                                [
                                    (
                                        cast(
                                            ClauseElement,
                                            RedditMeme.stonk_correct == False,
                                        ),
                                        1,
                                    )
                                ]
                            )
                        ).label("total_wrong"),
                    )
                    .filter(
                        and_(
                            cast(ClauseElement, RedditMeme.stonk == True),
                            cast(
                                ClauseElement,
                                RedditMeme.version == LOAD_MEME_CLF_VERSION,
                            ),
                        )
                    )
                    .group_by(RedditMeme.meme_clf)
                    .order_by(clause)
                    .statement,
                ),
                site_db.bind,
            )
            display_df(df)
