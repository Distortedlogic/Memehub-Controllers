from statistics import stdev
from time import time

import pandas as pd
from numpy import e, log


def score_df(scoring):
    top_1percent = scoring[scoring["upvotes"] > scoring["upvotes"].quantile(0.99)]
    bottom_99 = scoring[scoring["upvotes"] < scoring["upvotes"].quantile(0.99)]
    authors = list(set(list(top_1percent["username"])))
    lowest_upvotes_in_top = top_1percent["upvotes"].min()
    num_memes_in_bottom = (
        bottom_99.groupby("username").apply(len).sort_values(ascending=False)
    )
    num_memes_in_top = (
        top_1percent.groupby("username").apply(len).sort_values(ascending=False)
    )
    highest_upvotes = top_1percent.groupby("username")["upvotes"].max()
    lowest_upvote_ratio = (
        top_1percent.groupby("username")["upvote_ratio"]
        .apply(min)
        .sort_values(ascending=False)
    )

    num_memes_in_bottom_filled = []
    for author in authors:
        try:
            num_memes_in_bottom_filled.append(num_memes_in_bottom[author])
        except:
            num_memes_in_bottom_filled.append(0)

    scores_dict = {}
    for author in authors:
        try:
            inv_spammer_index = num_memes_in_top[author] / (
                num_memes_in_bottom[author] + 1
            )
            spammer_index = num_memes_in_bottom[author] / num_memes_in_top[author]
        except:
            inv_spammer_index = num_memes_in_top[author]
            spammer_index = 0
        highest_upvotes_score = log(
            e + (highest_upvotes[author] / lowest_upvotes_in_top)
        )
        lowest_ratio = lowest_upvote_ratio[author]
        scores_dict[author] = {
            "spammer_index": spammer_index,
            "highest_upvotes_score": highest_upvotes_score,
            "lowest_ratio": lowest_ratio,
            "score": inv_spammer_index * highest_upvotes_score * lowest_ratio,
        }

    raw_scores = [scores_dict[author]["score"] for author in authors]
    final_scores = 50 / (2.2 * stdev(log(raw_scores))) * log(raw_scores) + 50

    return (
        pd.DataFrame(
            {
                "username": authors,
                "final_score": final_scores,
                "raw_score": raw_scores,
                "num_in_bottom": num_memes_in_bottom_filled,
                "num_in_top": [num_memes_in_top[author] for author in authors],
                "shitposter_index": [
                    scores_dict[author]["spammer_index"] for author in authors
                ],
                "highest_upvotes": [highest_upvotes[author] for author in authors],
                "hu_score": [
                    scores_dict[author]["highest_upvotes_score"] for author in authors
                ],
                "lowest_ratio": [
                    scores_dict[author]["lowest_ratio"] for author in authors
                ],
            }
        )
        .sort_values("final_score", ascending=False)
        .reset_index(drop=True)
        .round(decimals=2)
    )


score_columns = [
    "username",
    "final_score",
    "raw_score",
    "num_in_bottom",
    "num_in_top",
    "shitposter_index",
    "highest_upvotes",
    "hu_score",
    "lowest_ratio",
]


def score_kwargs_gen(df):
    for _, row in df.iterrows():
        yield {column: row[column] for column in score_columns}
