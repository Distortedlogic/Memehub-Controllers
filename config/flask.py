from typing import cast

from decouple import config

SECRET_KEY = cast(str, config("SECRET_KEY"))
FLASK_ENV = cast(str, config("FLASK_ENV"))
DEBUG = FLASK_ENV == "development"
