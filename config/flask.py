from datetime import timedelta

from celery.schedules import crontab
from decouple import config

# if config('FLASK_ENV') == 'development':
#     DEBUG = True
# else:
#     DEBUG = False

DEBUG = False

SECRET_KEY = config("SECRET_KEY")

CELERY_BROKER_URL = "redis://redis:6379/0"
CELERY_RESULT_BACKEND = "redis://redis:6379/0"
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_REDIS_MAX_CONNECTIONS = 5

CELERYBEAT_SCHEDULE = {
    "reddit_memes_scrapper": {
        "task": "Reddit",
        "schedule": crontab(minute=0, hour="*"),
    },
}

SQLALCHEMY_TRACK_MODIFICATIONS = False
DB_NAME = config("DB_NAME")
DB_USER = config("DB_USER")
DB_PASS = config("POSTGRES_PASSWORD")
DB_PORT = config("DB_PORT")

# MEMEDATA_URI = f'postgresql://{DB_USER}:{DB_PASS}@127.0.0.1:{DB_PORT}/{DB_NAME}'
MEMEDATA_URI = f"postgresql://{DB_USER}:{DB_PASS}@memedata:{DB_PORT}/{DB_NAME}"
SITE_DATA_URI = f"postgresql://{DB_USER}:{DB_PASS}@sitedata:{DB_PORT}/{DB_NAME}"

SQLALCHEMY_DATABASE_URI = MEMEDATA_URI

SQLALCHEMY_BINDS = {"memedata": MEMEDATA_URI, "sitedata": SITE_DATA_URI}
