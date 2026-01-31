"""
Download a sample WLTP cycle file from the CORA Dataverse dataset.

Dataset: https://doi.org/10.34810/data2395
"""

from __future__ import annotations

import argparse
import re
import ssl
import urllib.request
from pathlib import Path


def _filename_from_headers(headers: dict) -> str | None:
    cd = headers.get("Content-Disposition") or headers.get("content-disposition")
    if not cd:
        return None
    match = re.search(r'filename="?([^";]+)"?', cd)
    return match.group(1) if match else None


def download_file(
    file_id: int,
    out_dir: Path,
    filename: str | None = None,
    insecure: bool = False,
) -> Path:
    url = f"https://dataverse.csuc.cat/api/access/datafile/{file_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    req = urllib.request.Request(url)
    context = ssl._create_unverified_context() if insecure else None
    with urllib.request.urlopen(req, context=context) as resp:
        name = filename or _filename_from_headers(resp.headers)
        if not name:
            name = f"dataverse_file_{file_id}.bin"
        out_path = out_dir / name
        with out_path.open("wb") as handle:
            handle.write(resp.read())
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Dataverse WLTP validation file.")
    parser.add_argument(
        "--file-id",
        type=int,
        default=369256,
        help="Dataverse dataFile id (default: 369256 = Qtzl_Cycle_003_WLTP_partial_data.parquet).",
    )
    parser.add_argument(
        "--out-dir",
        default="datasets/phone_validation",
        help="Output directory for downloaded files.",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Optional filename override.",
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
    out_path = download_file(args.file_id, out_dir, args.filename, args.insecure)
    print(f"Downloaded: {out_path}")


if __name__ == "__main__":
    main()
