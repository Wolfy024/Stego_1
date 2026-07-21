"""Download official COCO and DIV2K image archives without embedded credentials."""

from __future__ import annotations

import argparse
import os
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

DATASETS = {
    "coco-val2017": (
        "http://images.cocodataset.org/zips/val2017.zip",
        "coco/val2017.zip",
    ),
    "div2k-valid": (
        "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
        "div2k/DIV2K_valid_HR.zip",
    ),
}


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"unsafe archive member: {member.filename}")
        handle.extractall(destination)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "StegoGAN/1.0"})
    with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        copied = 0
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            copied += len(chunk)
            if total:
                print(f"\r{destination.name}: {copied / total:6.1%}", end="", flush=True)
    print()
    os.replace(temporary, destination)


def download_dataset(name: str, root: Path, keep_archive: bool) -> Path:
    url, relative_archive = DATASETS[name]
    archive = root / relative_archive
    extraction_root = archive.parent
    if not archive.exists():
        print(f"Downloading {name} from {url}")
        _download(url, archive)
    else:
        print(f"Using existing archive: {archive}")
    _safe_extract(archive, extraction_root)
    if not keep_archive:
        archive.unlink()
    print(f"Ready: {extraction_root.resolve()}")
    return extraction_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets", nargs="+", choices=sorted(DATASETS), help="datasets to download"
    )
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--keep-archives", action="store_true")
    args = parser.parse_args()
    for dataset in args.datasets:
        download_dataset(dataset, args.root, args.keep_archives)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
