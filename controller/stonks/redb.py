from jinja2.environment import Template
from controller.constants import DAY_TD, MONTH_TD, WEEK_TD
import json
from rejson import Client, Path
from sqlalchemy import and_, func
from controller.extensions import db

timeframes = [
    ("Daily", DAY_TD),
    ("Weekly", WEEK_TD),
    ("Monthly", MONTH_TD),
    ("Ever", None),
]


class StonkReDB:
    def __init__(self):
        self.rj = Client(host="redis", port=6379, decode_responses=True)

    def _set(self, name, obj, path=Path.rootPath()):
        self.rj.jsonset(name, path, obj)
        self.names = db.session.query(Template.name)

    def update(self):
        pass
