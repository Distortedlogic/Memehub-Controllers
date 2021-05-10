from typing import Any, cast

from celery_singleton import Singleton

from celery import Celery

TASK_LIST = [
    "src.tasks.reddit",
    "src.tasks.stonks",
]


def create_celery_app():
    celery = Celery(broker="pyamqp://rabbitmq:5672", include=TASK_LIST,)
    _: Any = celery.config_from_object("config.celery")

    class ContextTask(Singleton):
        abstract = True

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return cast(Any, Singleton.__call__(self, *args, **kwargs))

    celery.Task = ContextTask
    return celery
