from typing import Any

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.schema import Column
from sqlalchemy.sql.sqltypes import Integer, String

training_base: Any = declarative_base()
training_metadata = training_base.metadata  # type: ignore


class NotAMemeTrain(training_base):
    __tablename__ = "not_a_meme_train"

    id = Column(Integer, primary_key=True)
    path = Column(String(400), nullable=False)
    name_idx = Column(Integer, nullable=True)


class NotAMemeTest(training_base):
    __tablename__ = "not_a_meme_test"

    id = Column(Integer, primary_key=True)
    path = Column(String(400), nullable=False)
    name_idx = Column(Integer, nullable=True)


class NotATemplateTrain(training_base):
    __tablename__ = "not_a_template_train"

    id = Column(Integer, primary_key=True)
    path = Column(String(400), nullable=False)
    name_idx = Column(Integer, nullable=True)


class NotATemplateTest(training_base):
    __tablename__ = "not_a_template_test"

    id = Column(Integer, primary_key=True)
    path = Column(String(400), nullable=False)
    name_idx = Column(Integer, nullable=True)


class MemeIncorrectTrain(training_base):
    __tablename__ = "meme_incorrect_train"

    id = Column(Integer, primary_key=True)
    name = Column(String(400), nullable=False)
    path = Column(String(400), nullable=False)
    name_idx = Column(Integer, nullable=True)


class MemeIncorrectTest(training_base):
    __tablename__ = "meme_incorrect_test"

    id = Column(Integer, primary_key=True)
    name = Column(String(400), nullable=False)
    path = Column(String(400), nullable=False)
    name_idx = Column(Integer, nullable=True)


class MemeCorrectTrain(training_base):
    __tablename__ = "meme_correct_train"

    id = Column(Integer, primary_key=True)
    name = Column(String(400), nullable=False)
    path = Column(String(400), nullable=False)
    name_idx = Column(Integer, nullable=True)


class MemeCorrectTest(training_base):
    __tablename__ = "meme_correct_test"

    id = Column(Integer, primary_key=True)
    name = Column(String(400), nullable=False)
    path = Column(String(400), nullable=False)
    name_idx = Column(Integer, nullable=True)


class Template(training_base):
    __tablename__ = "templates"

    id = Column(Integer, primary_key=True)
    name = Column(String(400), nullable=False)
    page = Column(String(400), unique=True, nullable=False)
    blank_url = Column(String(400), nullable=False)
