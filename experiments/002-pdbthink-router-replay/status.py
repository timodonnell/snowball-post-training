"""Read the retry's durable audit and summarize training and fixed validation."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import boto3
from botocore.config import Config

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREVIOUS = ROOT / "experiments/001-pdbthink-coordinate"
sys.path.insert(0, str(PREVIOUS))
from snowball_pdbthink.coreweave import credential_environment
from snowball_pdbthink.monitor import summarize_history


def main():
    design = json.loads((HERE / "design.json").read_text())
    uri = urlparse(design["remote_root"] + "/audit/")
    env = credential_environment("/home/bizon/.kube/coreweave-iris-rno2a")
    client = boto3.client(
        "s3",
        endpoint_url=env["AWS_ENDPOINT_URL"],
        aws_access_key_id=env["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"],
        region_name=env["AWS_DEFAULT_REGION"],
        config=Config(s3={"addressing_style": "virtual"}),
    )
    cache = ROOT / "runs/002-replay/audit" / design["run"]
    prefix = uri.path.lstrip("/")
    training, validation, startup = [], [], []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=uri.netloc, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if not key.endswith(".json"):
                continue
            relative = key.removeprefix(prefix)
            path = cache / relative
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(client.get_object(Bucket=uri.netloc, Key=key)["Body"].read())
            record = json.loads(path.read_text())
            if relative.startswith("steps/"):
                training.append(record)
            elif relative.startswith("validation/"):
                validation.append({"trainer/global_step": record["global_step"], **record["metrics"]})
            elif "/" not in relative:
                startup.append(record)
    training.sort(key=lambda row: row["global_step"])
    validation.sort(key=lambda row: row["trainer/global_step"])
    counts = json.loads((PREVIOUS / "cohort_manifest.json").read_text())["family_counts"]["validation"]
    result = {
        "checked_at": datetime.now(UTC).isoformat(),
        "job": design["run"],
        "hardware": "48 CoreWeave H100",
        "startup": startup,
        "completed_optimizer_step": training[-1]["global_step"] if training else 0,
        "target_step": design["optimizer_updates"],
        "training": training,
        "validation": summarize_history(validation, counts),
    }
    (ROOT / "runs/002-replay/progress.json").write_text(json.dumps(result, indent=2) + "\n")
    latest = training[-1]["metrics"] if training else {}
    keys = [
        "reward/avg_raw_reward",
        "policy/mismatch/pooled/log_ratio_abs_mean",
        "policy/mismatch/pooled/log_ratio_abs_p999",
        "policy/mismatch/pooled/kl_k3",
        "policy/mismatch/pooled/ess_fraction",
        "policy/mismatch/pooled/frac_outside_0_5_2",
        "policy/mismatch/pooled/pos_first256/log_ratio_abs_mean",
        "policy/mismatch/pooled/pos_last256/log_ratio_abs_mean",
        "policy/correction/truncated_fraction",
        "policy/router_replay/hit_fraction",
        "policy/frozen_router_bias_unchanged",
    ]
    print(
        json.dumps(
            {
                "job": design["run"],
                "completed_optimizer_step": result["completed_optimizer_step"],
                "startup_checks_completed": bool(startup),
                "latest_metrics": {key: latest.get(key) for key in keys},
                "latest_validation": result["validation"][-1] if validation else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
