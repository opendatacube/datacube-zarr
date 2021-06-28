from pathlib import Path


def copytree(p1: Path, p2: Path) -> None:
    """Copytree for local/s3 paths."""
    for o1 in p1.iterdir():
        o2 = p2 / o1.name
        if o1.is_dir():
            copytree(o1, o2)
        else:
            if o2.as_uri().startswith("file") and not o2.parent.exists():
                o2.parent.mkdir(parents=True)
            o2.write_bytes(o1.read_bytes())
