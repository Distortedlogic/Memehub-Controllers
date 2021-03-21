import boto3
import pandas as pd
from decouple import config
from IPython.display import display
from pandas import option_context
from pandas.core.frame import DataFrame
from PIL import Image
from src.generated.models import RedditMeme
from src.session import site_db
from src.utils.image_funcs import load_img_from_url
from torchvision import transforms

s3 = boto3.resource(
    "s3", aws_access_key_id=config("AWS_ID"), aws_secret_access_key=config("AWS_KEY")
)
bucket = s3.Bucket("memehub")


def display_df(df: DataFrame):
    with option_context("display.max_rows", None, "display.max_columns", None):  # type: ignore
        _ = display(df)


def display_template(name: str):
    try:
        object = bucket.Object("memehub/templates/" + name)
        response = object.get()
        file_stream = response["Body"]
        template = Image.open(file_stream)
        _ = display(template)
    except:
        pass


def display_meme(meme: RedditMeme):
    try:
        image = transforms.ToPILImage()(load_img_from_url(meme.url))
    except:
        site_db.delete(meme)
        return
    display_df(pd.DataFrame([("meme_clf", meme.meme_clf), ("stonk", meme.stonk)]))
    _ = display(image)
