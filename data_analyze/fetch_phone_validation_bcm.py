"""
Download the BCMes smartphone battery dataset (JSON) from GitHub.
"""

from __future__ import annotations

import argparse
import io
import ssl
import urllib.request
import zipfile
from pathlib import Path


def download_and_extract(url: str, out_dir: Path, insecure: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    context = ssl._create_unverified_context() if insecure else None
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=context) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(out_dir)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BCMes dataset zip.")
    parser.add_argument(
        "--url",
        default="https://github.com/yuliabardinova/BCMes/archive/refs/heads/master.zip",
        help="ZIP URL for BCMes dataset repository.",
    )
    parser.add_argument(
        "--out-dir",
        default="datasets/phone_validation_real",
        help="Output directory for extracted dataset.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL verification for downloads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    download_and_extract(args.url, out_dir, args.insecure)
    print(f"Extracted to: {out_dir}")


if __name__ == "__main__":
    main()
