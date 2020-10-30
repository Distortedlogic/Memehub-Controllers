import os

from celery.schedules import crontab
from decouple import config
from get_docker_secret import get_docker_secret

secrets_dir = os.path.join("run", "secrets")

SECRET_KEY = get_docker_secret(
    "SECRET_KEY", default=config("SECRET_KEY"), secrets_dir=secrets_dir
)
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

FLASK_ENV = get_docker_secret(
    "FLASK_ENV", default=config("FLASK_ENV"), secrets_dir=secrets_dir
)

SQLALCHEMY_TRACK_MODIFICATIONS = False
user = get_docker_secret(
    "POSTGRES_USER", default=config("POSTGRES_USER"), secrets_dir=secrets_dir
)
password = get_docker_secret(
    "POSTGRES_PASSWORD", default=config("POSTGRES_PASSWORD"), secrets_dir=secrets_dir
)
db = get_docker_secret(
    "POSTGRES_DB", default=config("POSTGRES_DB"), secrets_dir=secrets_dir
)
SQLALCHEMY_DATABASE_URI = f"postgresql://{user}:{password}@sitedata:5432/{db}"
