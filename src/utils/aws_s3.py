from typing import Any, Dict, List

import boto3
import pandas as pd
from botocore.exceptions import ClientError, NoCredentialsError
from decouple import config
from IPython.core.display import clear_output
from src.constants import MODELS_REPO, TRAINING_VERSION
from src.utils.display import display_df

s3: Any = boto3.client(
    "s3", aws_access_key_id=config("AWS_ID"), aws_secret_access_key=config("AWS_KEY"),
)


def upload_to_aws(path: str, key: str, fresh: bool) -> bool:
    if fresh:
        try:
            s3.upload_file(path, "memehub", key)
            return True
        except FileNotFoundError:
            print("The file was not found")
            return False
        except NoCredentialsError:
            print("Credentials not available")
            return False
    try:
        s3.head_object(Bucket="memehub", Key=key)
        return True
    except ClientError:
        try:
            s3.upload_file(path, "memehub", key)
            return True
        except FileNotFoundError:
            print("The file was not found")
            return False
        except NoCredentialsError:
            print("Credentials not available")
            return False


def memeclf_to_aws(fresh: bool) -> None:
    _ = upload_to_aws(
        MODELS_REPO + f"/{TRAINING_VERSION}/static.json",
        f"memehub/models/{TRAINING_VERSION}/static.json",
        fresh,
    )
    _ = upload_to_aws(
        MODELS_REPO + f"/{TRAINING_VERSION}/jit/meme_clf.pt",
        f"memehub/models/{TRAINING_VERSION}/jit/meme_clf.pt",
        fresh,
    )
    _ = upload_to_aws(
        MODELS_REPO + f"/{TRAINING_VERSION}/jit/features.pt",
        f"memehub/models/{TRAINING_VERSION}/jit/features.pt",
        fresh,
    )


def load_all_to_aws(names: List[str], fresh: bool) -> None:
    stats: Dict[str, int] = dict(num_names=len(names), success=0, failed=0)
    for name in names:
        path = MODELS_REPO + f"/{TRAINING_VERSION}/jit/{name}.pt"
        Key = f"memehub/models/{TRAINING_VERSION}/jit/{name}.pt"
        success = upload_to_aws(path, Key, fresh)
        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1
        clear_output()
        display_df(pd.DataFrame.from_records([stats]))
