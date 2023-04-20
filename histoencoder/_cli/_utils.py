import sys
from typing import NoReturn

import rich_click as click


def info(msg: str) -> None:
    """Display info message."""
    prefix = click.style("INFO: ", bold=True, fg="cyan")
    click.echo(prefix + msg)


def warning(msg: str) -> None:
    """Display warning message."""
    prefix = click.style("WARNING: ", bold=True, fg="yellow")
    click.echo(prefix + msg)


def error(msg: str) -> NoReturn:
    """Display error message and exit."""
    prefix = click.style("ERROR: ", bold=True, fg="red")
    click.echo(prefix + msg)
    sys.exit(1)


def info_exit(msg: str) -> NoReturn:
    info(msg)
    sys.exit(0)
