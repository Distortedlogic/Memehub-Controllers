from typing import cast

from decouple import config
from sqlalchemy.engine import create_engine
from sqlalchemy.orm.session import Session, sessionmaker

from src.generated.models import *

user = cast(str, config("POSTGRES_USER"))
password = cast(str, config("POSTGRES_PASSWORD"))
train_db_name = cast(str, config("TRAINING_DB"))
site_db_name = cast(str, config("POSTGRES_DB"))

TRAINING_URL = f"postgresql://{user}:{password}@127.0.0.1:5432/{train_db_name}"
SITE_URL = f"postgresql://{user}:{password}@sitedata:5432/{site_db_name}"

training_engine = create_engine(TRAINING_URL, echo=False)
TrainingSession = sessionmaker(bind=training_engine)
training_db: Session = TrainingSession()

site_engine = create_engine(SITE_URL, echo=False)
SiteSession = sessionmaker(bind=site_engine)
site_db: Session = SiteSession()
