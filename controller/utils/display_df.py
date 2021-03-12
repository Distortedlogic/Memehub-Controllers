from IPython.display import display
from pandas import option_context
from pandas.core.frame import DataFrame


def display_df(df: DataFrame):
    with option_context("display.max_rows", None, "display.max_columns", None): # type: ignore
        _=display(df)
