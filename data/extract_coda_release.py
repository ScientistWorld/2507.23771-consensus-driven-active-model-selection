"""Extract selected task tensors from the upstream release archive."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="data/downloads/coda_data.zip")
    parser.add_argument("--output-dir", default="data/downloads/coda_data")
    parser.add_argument("--tasks", nargs="+", default=["glue_rte", "glue_sst2", "cifar10_5592"])
    args = parser.parse_args()

    archive = Path(args.archive)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        for task in args.tasks:
            member = f"data_simple/{task}.pt"
            if member not in names:
                raise FileNotFoundError(f"{member} not found in {archive}")
            target = out_dir / f"{task}.pt"
            if target.exists():
                continue
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"extracted {target}")


if __name__ == "__main__":
    main()
