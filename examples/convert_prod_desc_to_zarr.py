#! /usr/bin/env python

from pathlib import Path

import click
from ruamel.yaml import YAML


@click.command()
@click.argument("prodfile", type=click.Path(), required=True)
@click.argument("outdir", type=click.Path(), required=False)
def main(prodfile, outdir):
    """Replace format name in product description with 'zarr'."""
    prodfile = Path(prodfile).expanduser().resolve()
    outdir = Path(outdir) if outdir else prodfile.parent
    outpath = outdir / f"{prodfile.stem}_zarr{prodfile.suffix}"

    def _convert(prod):
        if "format" in prod["metadata"]:
            prod["metadata"]["format"]["name"] = "zarr"
            prod["name"] = prod["name"] + "_zarr"
        return prod

    yaml = YAML()

    with open(prodfile, "r") as fin, open(outpath, "w") as fout:
        docs = (_convert(p) for p in yaml.load_all(fin))
        yaml.dump_all(docs, fout)

    print(outpath)


if __name__ == "__main__":
    main()
