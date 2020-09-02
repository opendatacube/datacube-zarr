#! /usr/bin/env python
"""Generate basic readme doc for zarrify click application."""

from pathlib import Path
from textwrap import indent

import click

from datacube_zarr.tools.zarrify import main as zarrify

DOC_DIRECTORY = Path(__file__).parent


def generate_zarrify_doc():
    """Get usage/help text from zarrify command and add to a readme."""

    readme_file = DOC_DIRECTORY / "zarrify.md"

    intro = (
        "The `zarrify` tool converts existing raster datasets to "
        "[Zarr format](https://zarr.readthedocs.io/en/stable/spec/v2.html). "
        "It is installed as a command line tool with `datacube-zarr`.\n\n"
        "See usage below for details."
    )

    ctx = click.Context(zarrify)
    usage = "$ zarrify --help\n\n" + zarrify.get_help(ctx)

    readme_text = (
        "# zarrify command line tool\n"
        f"{intro}\n"
        "## Usage\n"
        f"{indent(usage, prefix='    ')}\n"
    )

    with open(readme_file, "w") as f:
        f.write(readme_text)

    return readme_file


if __name__ == "__main__":
    out = generate_zarrify_doc()
    print(f"Generated output: {out.resolve()}")
