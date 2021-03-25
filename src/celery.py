from typing import Any, cast

from celery_singleton import Singleton

from celery import Celery

TASK_LIST = ["src.reddit.tasks", "src.stonks.tasks"]


def create_celery_app():
    celery = Celery(broker="redis://redis:6379", include=TASK_LIST,)
    _: Any = celery.config_from_object("config.celery")

    class ContextTask(Singleton):
        abstract = True

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return cast(Any, Singleton.__call__(self, *args, **kwargs))

    celery.Task = ContextTask
    return celery
