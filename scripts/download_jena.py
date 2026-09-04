#!/usr/bin/env python3
"""Download and verify the public Jena 2009-2016 climate dataset."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "jena_climate_2009_2016.csv.zip"
    temporary = archive.with_suffix(".download")
    print(f"Downloading {args.url}")
    with urllib.request.urlopen(args.url) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    temporary.replace(archive)
    checksum = sha256(archive)
    with zipfile.ZipFile(archive) as bundle:
        members = [name for name in bundle.namelist() if name.endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV in archive, found {members}")
        bundle.extract(members[0], args.output_dir)
        extracted = args.output_dir / members[0]
    print(f"Archive SHA256: {checksum}")
    print(f"Extracted: {extracted}")


if __name__ == "__main__":
    main()

