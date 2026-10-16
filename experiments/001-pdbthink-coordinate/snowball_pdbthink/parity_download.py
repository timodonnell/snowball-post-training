"""Download a completed parity diagnostic without rewriting remote artifacts."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from finestore.reader import ReadView
from rigging.filesystem.storage_path import StoragePath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = StoragePath(args.root)
    with (root / "complete.json").open("r") as stream:
        complete = json.load(stream)
    if complete["optimizer_updates"] != 0:
        raise ValueError("The diagnostic unexpectedly performed an optimizer update")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("complete.json", "started.json", "weight-verification.json", "router-weight-verification.json"):
        with (root / name).open("rb") as stream:
            (args.output / name).write_bytes(stream.read())
    for suffix in ("", "decode-control"):
        destination = args.output / suffix
        destination.mkdir(exist_ok=True)
        view = ReadView(str(root / suffix) if suffix else str(root))
        for table in ("probe", "scores", "manifest"):
            data = view.scan(table)
            pq.write_table(data, destination / f"{table}.parquet", compression="zstd")
            print(json.dumps({"table": f"{suffix}/{table}", "rows": data.num_rows}), flush=True)
    print(json.dumps({"complete": complete["global_step"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
