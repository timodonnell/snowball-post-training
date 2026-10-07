"""Stage this experiment in a pinned Marin checkout and invoke its artifact main."""

import argparse
import netrc
import os
import shutil
import subprocess
from pathlib import Path

from snowball_pdbthink.pins import MARIN_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marin", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--module", default="experiments.snowball_pdbthink.launch")
    args, options = parser.parse_known_args()
    checkout = args.marin.resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if revision != MARIN_REVISION:
        raise ValueError(f"Expected Marin {MARIN_REVISION}, got {revision}")
    source = Path(__file__).parent / "snowball_pdbthink"
    target = checkout / "experiments/snowball_pdbthink"
    shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(checkout), *(str(p) for p in sorted((checkout / "lib").glob("*/src")))])
    env["MARIN_PREFIX"] = "s3://marin-us-east-02a/marin"
    env["MARIN_CLUSTER"] = "coreweave"
    if "WANDB_API_KEY" not in env:
        auth = netrc.netrc().authenticators("api.wandb.ai")
        if auth:
            env["WANDB_API_KEY"] = auth[2]
    command = [str(args.python.absolute()), "-m", args.module, *options]
    raise SystemExit(subprocess.call(command, cwd=checkout, env=env))


if __name__ == "__main__":
    main()
