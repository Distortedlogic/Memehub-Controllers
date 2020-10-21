import subprocess, click, os


@click.command()
def cli():
    """
    Runs Arbitrary Scripts from Scripts Folder

    Arguments:
        name {[type]} -- filename without ext
    """

    cmd = "sqlacodegen --flask --outfile controller/generated/models.py postgresql://postgres:postgres@127.0.0.1:2345"

    return subprocess.call(cmd, shell=True)
