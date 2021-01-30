import boto3
import redisai
from decouple import config

rai = redisai.Client(host="redis", port="6379")

s3 = boto3.client(
    "s3", aws_access_key_id=config("AWS_ID"), aws_secret_access_key=config("AWS_KEY"),
)


def load_models_to_redisai(device="CPU"):
    try:
        with open("FILE_NAME", "wb") as f:
            s3.download_fileobj("BUCKET_NAME", "OBJECT_NAME", f)
    except:
        pass
