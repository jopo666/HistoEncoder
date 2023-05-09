import sys
from typing import NoReturn

import rich_click as click

BAR_FORMAT = "  {l_bar}{bar:20}{r_bar}{bar:-20b}"


def warning(msg: str) -> None:
    """Display warning message."""
    prefix = click.style("WARNING: ", bold=True, fg="yellow")
    click.echo(prefix + msg)


def error(msg: str) -> NoReturn:
    """Display error message and exit."""
    prefix = click.style("ERROR: ", bold=True, fg="red")
    click.echo(prefix + msg)
    sys.exit(1)
