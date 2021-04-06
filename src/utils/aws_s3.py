import os
from typing import Any, Dict, List

import boto3
import pandas as pd
from botocore.exceptions import ClientError, NoCredentialsError
from decouple import config
from IPython.core.display import clear_output
from src.constants import (
    LOAD_MEME_CLF_REPO,
    LOAD_MEME_CLF_VERSION,
    LOAD_STATIC_PATH,
    LOAD_STONK_REPO,
)
from src.utils.display import display_df

s3: Any = boto3.client(
    "s3", aws_access_key_id=config("AWS_ID"), aws_secret_access_key=config("AWS_KEY"),
)


def upload_to_aws(path: str) -> bool:
    try:
        s3.head_object(Bucket="memehub", Key=path.replace("src", "memehub"))
        return True
    except ClientError:
        try:
            s3.upload_file(path, "memehub", path.replace("src", "memehub"))
            return True
        except FileNotFoundError:
            print("The file was not found")
            return False
        except NoCredentialsError:
            print("Credentials not available")
            return False


def meme_clf_to_aws() -> None:
    success = upload_to_aws(
        LOAD_STATIC_PATH.format(LOAD_MEME_CLF_VERSION) + "static.json"
    )
    print("static " + str(success))
    success = upload_to_aws(LOAD_MEME_CLF_REPO.format("jit") + "features.pt")
    print("features " + str(success))
    success = upload_to_aws(LOAD_MEME_CLF_REPO.format("jit") + "dense.pt")
    print("dense " + str(success))


def stonks_to_aws() -> None:
    names = [
        os.path.splitext(filename)[0]
        for filename in os.listdir(LOAD_STONK_REPO.format("jit"))
    ]
    stats: Dict[str, int] = dict(num_names=len(names), success=0, failed=0)
    for name in names:
        success = upload_to_aws(LOAD_STONK_REPO.format("jit") + f"{name}.pt")
        stats["success" if success else "failed"] += 1
        clear_output()
        display_df(pd.DataFrame.from_records([stats]))
