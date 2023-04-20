import rich_click as click

from ._extract import extract_group

# Rich-click options.
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.RANGE_STRING = ""
click.rich_click.STYLE_HEADER_TEXT = "dim"
click.rich_click.MAX_WIDTH = 120
click.rich_click.SHOW_METAVARS_COLUMN = False
click.rich_click.APPEND_METAVARS_HELP = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.USE_CLICK_SHORT_HELP = False


cli = click.CommandCollection(sources=[extract_group])
