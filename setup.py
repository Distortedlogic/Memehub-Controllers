from setuptools import setup

setup(
    name="mh-CLI",
    version="1.0",
    packages=["cli", "cli.commands"],
    include_package_data=True,
    install_requires=["click", "sqlacodegen", "flask-sqlacodegen"],
    entry_points="""
        [console_scripts]
        mh=cli.cli:cli
    """,
)
