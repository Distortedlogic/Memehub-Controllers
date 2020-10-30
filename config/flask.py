from celery.schedules import crontab
from decouple import config

SECRET_KEY = config("SECRET_KEY")
CELERY_BROKER_URL = "redis://redis:6379"
CELERY_RESULT_BACKEND = "redis://redis:6379"
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

FLASK_ENV = config("FLASK_ENV")

SQLALCHEMY_TRACK_MODIFICATIONS = False
user = config("POSTGRES_USER")
password = config("POSTGRES_PASSWORD")
db = config("POSTGRES_DB")
SQLALCHEMY_DATABASE_URI = f"postgresql://{user}:{password}@sitedata:5432/{db}"
