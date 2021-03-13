from typing import cast

from decouple import config
from sqlalchemy.engine import create_engine
from sqlalchemy.orm.session import Session, sessionmaker

from controller.generated.models import *

user = cast(str, config("POSTGRES_USER"))
password = cast(str, config("POSTGRES_PASSWORD"))
site_db_name = cast(str, config("POSTGRES_DB"))

SITE_URL = f"postgresql://{user}:{password}@sitedata:5432/{site_db_name}"

site_engine = create_engine(SITE_URL, echo=False)
SiteSession = sessionmaker(bind=site_engine)
site_db: Session = SiteSession()
