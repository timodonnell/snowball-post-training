"""Download diagnostic scores and their explicit completion scope."""

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
    parser.add_argument(
        "--vllm-only", action="store_true", help="Recover only a failed attempt's committed vLLM replay"
    )
    parser.add_argument(
        "--fixed-panel-only",
        action="store_true",
        help="Recover a completed primary panel without the auxiliary control",
    )
    args = parser.parse_args()
    if args.vllm_only and args.fixed_panel_only:
        parser.error("Choose at most one recovery scope")
    root = StoragePath(args.root)
    if args.vllm_only:
        view = ReadView(str(root))
        probes, scores = view.scan("probe"), view.scan("scores")
        rows = scores.to_pylist()
        if not rows or any(row["scorer"] != "vllm.rescore" or row["mode"] != "" for row in rows):
            raise ValueError("Partial recovery requires exactly the initial vLLM replay")
        if len(rows) != probes.num_rows or {row["sample_id"] for row in rows} != set(probes["sample_id"].to_pylist()):
            raise ValueError("Partial vLLM replay has incomplete sample coverage")
        args.output.mkdir(parents=True, exist_ok=True)
        pq.write_table(probes, args.output / "probe.parquet", compression="zstd")
        pq.write_table(scores, args.output / "scores.parquet", compression="zstd")
        for name in ("started.json", "weight-verification.json", "router-weight-verification.json"):
            with (root / name).open("rb") as stream:
                (args.output / name).write_bytes(stream.read())
        status = {"complete": False, "recovered": "vllm.rescore only", "samples": len(rows), "source": args.root}
        (args.output / "partial.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status))
        return
    marker = "fixed-panel-complete.json" if args.fixed_panel_only else "complete.json"
    with (root / marker).open("r") as stream:
        complete = json.load(stream)
    if complete["optimizer_updates"] != 0:
        raise ValueError("The diagnostic unexpectedly performed an optimizer update")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in (marker, "started.json", "weight-verification.json", "router-weight-verification.json"):
        with (root / name).open("rb") as stream:
            (args.output / name).write_bytes(stream.read())
    for suffix in ("",) if args.fixed_panel_only else ("", "decode-control"):
        destination = args.output / suffix
        destination.mkdir(exist_ok=True)
        view = ReadView(str(root / suffix) if suffix else str(root))
        for table in ("probe", "scores", "manifest"):
            data = view.scan(table)
            pq.write_table(data, destination / f"{table}.parquet", compression="zstd")
            print(json.dumps({"table": f"{suffix}/{table}", "rows": data.num_rows}), flush=True)
    print(json.dumps({"complete_step": complete["global_step"], "scope": marker, "output": str(args.output)}))


if __name__ == "__main__":
    main()
