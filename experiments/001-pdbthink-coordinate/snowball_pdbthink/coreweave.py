"""Run a local command with existing CoreWeave task credentials, without printing them."""

import argparse
import base64
import json
import os
import subprocess


def credential_environment(kubeconfig):
    response = subprocess.run(
        ["kubectl", "--kubeconfig", kubeconfig, "-n", "iris", "get", "secret", "iris-task-env", "-o", "json"],
        check=True,
        text=True,
        capture_output=True,
    )
    secret = json.loads(response.stdout)["data"]
    env = dict(os.environ)
    for name in (
        "CW_KEY_ID",
        "CW_KEY_SECRET",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
        "AWS_REGION",
    ):
        env[name] = base64.b64decode(secret[name]).decode()
    env["CW_S3_ENDPOINT"] = "https://cwobject.com"
    env["AWS_ENDPOINT_URL"] = "https://cwobject.com"
    config = json.loads(base64.b64decode(secret["FSSPEC_S3"]).decode())
    config["endpoint_url"] = "https://cwobject.com"
    config.setdefault("client_kwargs", {}).pop("endpoint_url", None)
    env["FSSPEC_S3"] = json.dumps(config)
    env["MARIN_CLUSTER"] = "coreweave"
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kubeconfig", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[0] == "--" else args.command
    raise SystemExit(subprocess.call(command, env=credential_environment(args.kubeconfig)))


if __name__ == "__main__":
    main()
