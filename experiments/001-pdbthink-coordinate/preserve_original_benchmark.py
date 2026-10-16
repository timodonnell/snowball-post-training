"""Preserve completed benchmark evidence with an explicit, credential-free allowlist."""

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

import boto3
from botocore.config import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/001/original-benchmark"))
    parser.add_argument("--results", type=Path, default=Path("experiments/001-pdbthink-coordinate/results"))
    args = parser.parse_args()
    plan = json.loads((args.run / "plan.json").read_text())
    comparison = json.loads((args.run / "comparison.json").read_text())
    assert comparison["n"] == plan["selected"] == 40
    names = [
        "plan.json",
        "selection.json",
        "comparison.json",
        "native-checkpoint-verification.json",
        "export-verification.json",
        "runtime-provenance.json",
        "historical-score-replay.json",
        "benchmark-source-snapshot.tar.gz",
    ]
    for model in ("baseline", "trained"):
        assert json.loads((args.run / model / "complete.json").read_text())["completed"] == 40
        names.extend(f"{model}/{name}.json" for name in ("contract", "server", "complete"))
        names.extend(f"{model}/{hashlib.sha256(j['render_id'].encode()).hexdigest()[:20]}.json" for j in plan["jobs"])
    hashes = {name: hashlib.sha256((args.run / name).read_bytes()).hexdigest() for name in names}
    manifest_path = args.run / "evidence-files.json"
    manifest_path.write_text(json.dumps(hashes, indent=2) + "\n")
    archive = args.run / "original-benchmark-evidence.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in [*names, "evidence-files.json"]:
            tar.add(args.run / name, arcname=name)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    bucket = "marin-us-east-02a"
    key = (
        "marin/checkpoints/bizon/snowball-pdbthink-epoch/2026.10.15.1/exports/analysis/"
        f"original-benchmark-step384-{sha[:16]}.tar.gz"
    )
    client = boto3.client(
        "s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"], config=Config(s3={"addressing_style": "virtual"})
    )
    client.upload_file(str(archive), bucket, key, ExtraArgs={"Metadata": {"sha256": sha}})
    head = client.head_object(Bucket=bucket, Key=key)
    assert head["ContentLength"] == archive.stat().st_size and head["Metadata"]["sha256"] == sha
    manifest = {
        "uri": f"s3://{bucket}/{key}",
        "bytes": archive.stat().st_size,
        "sha256": sha,
        "files": hashes,
        "dataset_fingerprint": plan["dataset_fingerprint"],
        "benchmark_version": plan["benchmark_version"],
        "eligible": plan["selected"],
        "excluded": plan["excluded"],
        "selected_by_family": plan["selected_by_family"],
        "selection": comparison["selection"],
        "runtime": json.loads((args.run / "runtime-provenance.json").read_text()),
        "historical_score_replay": json.loads((args.run / "historical-score-replay.json").read_text()),
        "verification": "Remote size and SHA256 metadata verified after upload; per-file SHA256 manifest inside archive",
    }
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "original-benchmark-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copyfile(args.run / "comparison.json", args.results / "original-benchmark-comparison.json")
    print(json.dumps({k: manifest[k] for k in ("uri", "bytes", "sha256", "eligible", "excluded")}))


if __name__ == "__main__":
    main()
