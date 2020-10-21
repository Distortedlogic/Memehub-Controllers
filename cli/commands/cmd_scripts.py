import subprocess, click, os


@click.command()
@click.argument("name", default="")
def cli(name):
    """
    Runs Arbitrary Scripts from Scripts Folder

    Arguments:
        name {[type]} -- filename without ext
    """

    full_path = os.path.join("scripts", name + ".py")

    cmd = f"python {full_path}"

    return subprocess.call(cmd, shell=True)
