from celery.schedules import crontab

CELERY_BROKER_URL = "pyamqp://rabbitmq:5672"
CELERY_RESULT_BACKEND = "redis://redis:6379"
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_REDIS_MAX_CONNECTIONS = 5
CELERYBEAT_SCHEDULE = {
    "reddit": {"task": "Reddit", "schedule": crontab(minute=0, hour="*")},
    "reddit_new": {"task": "RedditNew", "schedule": crontab(minute=0, hour="*")},
    "close_investments": {
        "task": "CloseInvestments",
        "schedule": crontab(minute=0, hour="*"),
    },
}
