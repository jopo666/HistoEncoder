import rich_click as click

from ._cluster import cluster
from ._extract import extract

# Rich-click options.
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.RANGE_STRING = ""
click.rich_click.STYLE_HEADER_TEXT = "dim"
click.rich_click.MAX_WIDTH = 100
click.rich_click.SHOW_METAVARS_COLUMN = False
click.rich_click.APPEND_METAVARS_HELP = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.USE_CLICK_SHORT_HELP = False

click.rich_click.COMMAND_GROUPS = {
    "HistoEncoder": [
        {
            "name": "Commands for slides processed with [bold red]HistoPrep[/bold red]",
            "commands": ["extract", "cluster"],
        },
    ]
}


@click.group()
def run() -> None:
    """Foundation model for digital pathology."""


run.add_command(extract)
run.add_command(cluster)
