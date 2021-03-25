from typing import List, Union, cast

import pandas as pd
from IPython.core.display import clear_output
from sqlalchemy import and_
from sqlalchemy.sql.elements import BooleanClauseList, ClauseElement
from src.constants import LOAD_MEME_CLF_VERSION
from src.generated.models import RedditMeme
from src.session import site_db
from src.utils.auditor_stats import print_stats
from src.utils.display import display_meme, display_template
from src.utils.model_func import get_static_names


class Auditor:
    def __init__(self):
        self.prev_ids: List[int] = []

    def update_meme(self, meme: RedditMeme) -> Union[bool, None]:
        if "gb" in (keypress := input("stonk correct?")):
            if (prev_meme := site_db.query(RedditMeme).get(self.prev_ids.pop())) :
                _ = display_meme(prev_meme)
                return self.update_meme(prev_meme)
        elif keypress == "q":
            return True
        else:
            stonk_correct = not bool(keypress)
            meme.stonk_correct = stonk_correct  # type: ignore
            if stonk_correct:
                if meme.stonk:
                    meme.meme_clf_correct = True  # type: ignore
                    meme.is_a_template = True  # type: ignore
                else:
                    meme.meme_clf_correct = False  # type: ignore
                    meme.is_a_template = not bool(input("is_a_template?"))  # type: ignore
            else:
                if meme.stonk:
                    meme.meme_clf_correct = False  # type: ignore
                    meme.is_a_template = not bool(input("is_a_template?"))  # type: ignore
                else:
                    meme.meme_clf_correct = True  # type: ignore
                    meme.is_a_template = True  # type: ignore

    def audit(self, clause: BooleanClauseList):
        for meme in site_db.query(RedditMeme).filter(clause):
            clear_output()
            print_stats(meme.meme_clf)
            display_meme(meme)
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
        folder_count_df = pd.DataFrame(
            list(get_static_names(LOAD_MEME_CLF_VERSION)["folder_count"].items()),
            columns=["name", "count"],
        )
        keypress = input(
            """
        What do u want to audit?
        c - correct
        i - incorrect
        """
        )
        if keypress == "c":
            for name in cast(List[str], folder_count_df.sort_values("count")["name"]):
                self.audit(self.get_stonk_clause(name, correct=True))
        elif keypress == "i":
            for name in cast(List[str], folder_count_df.sort_values("count")["name"]):
                self.audit(self.get_stonk_clause(name, correct=False),)

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
