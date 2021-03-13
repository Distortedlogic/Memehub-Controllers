import subprocess

import click


@click.command()
def cli():
    """
    Runs Arbitrary Scripts from Scripts Folder

    Arguments:
        name {[type]} -- filename without ext
    """

    cmd = "sqlacodegen --outfile controller/generated/models.py postgresql://postgres:postgres@sitedata:5432"

    return subprocess.call(cmd, shell=True)
