"""Publish the immutable overlay and start the durable Marin launcher."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import boto3
import yaml
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT / "experiments/001-pdbthink-coordinate"
OUT = ROOT / "runs/002-replay"
sys.path.insert(0, str(PREVIOUS))
from snowball_pdbthink.coreweave import credential_environment


def main():
    cfg = yaml.safe_load((HERE / "launch.yaml").read_text())
    settings = cfg["runtime"]["task_env"]
    uri = urlparse(settings["SNOWBALL_REPLAY_CODE_URI"])
    payload = (OUT / Path(uri.path).name).read_bytes()
    if hashlib.sha256(payload).hexdigest() != settings["SNOWBALL_REPLAY_CODE_SHA256"]:
        raise ValueError("Overlay identity mismatch")
    env = credential_environment("/home/bizon/.kube/coreweave-iris-rno2a")
    storage = boto3.client(
        "s3",
        config=Config(s3={"addressing_style": "virtual"}),
        endpoint_url=env["AWS_ENDPOINT_URL"],
        aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
        region_name=env["AWS_DEFAULT_REGION"],
    )
    storage.put_object(Bucket=uri.netloc, Key=uri.path.lstrip("/"), Body=payload)
    stored = storage.get_object(Bucket=uri.netloc, Key=uri.path.lstrip("/"))["Body"].read()
    if stored != payload:
        raise ValueError("Uploaded overlay differs")
    parent = str(Path(uri.path.lstrip("/")).parent)
    for name in ("launch.yaml", "design.json", "overlay-manifest.json"):
        storage.put_object(Bucket=uri.netloc, Key=parent + "/" + name, Body=(HERE / name).read_bytes())
    env["PYTHONPATH"] = str(PREVIOUS)
    command = [
        str(ROOT / ".venv/bin/python"),
        str(PREVIOUS / "run_marin.py"),
        "--marin",
        str(ROOT / "data/marin"),
        "--python",
        "/home/bizon/git/marin/.venv/bin/python",
        "--module",
        "experiments.snowball_pdbthink.replay_launch",
        "--config",
        str(HERE / "launch.yaml"),
    ]
    log = OUT / "launch.log"
    with log.open("wb") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
        )
    record = {
        "pid": process.pid,
        "command": command,
        "log": str(log),
        "job": cfg["iris"]["job_name"],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    (OUT / "process.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
