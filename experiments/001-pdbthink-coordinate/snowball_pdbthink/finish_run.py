"""Wait for a Marin export, evaluate it, and preserve the paired comparison."""

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from .compare import compare, load_evaluation
from .pins import MODEL_REVISION, SKYRL_REVISION


def exported_model(manifest):
    result = manifest["result"]
    if result["state"] != "succeeded" or result["launcher_commit"] != SKYRL_REVISION:
        raise ValueError("Expected a successful export from the pinned MarinSkyRL runtime")
    model = result["model"]
    if model is None or model["global_step"] <= 0 or model["tokenizer_revision"] != MODEL_REVISION:
        raise ValueError("Export is missing a trained checkpoint or uses a different tokenizer")
    return model


def wait_for_export(iris, terminal_uri, coordinator, timeout):
    from iris.cluster.types import JobName
    from iris.resources.state import JobState, is_job_finished
    from rigging.filesystem.storage_path import StoragePath

    terminal = StoragePath(terminal_uri)
    started = time.monotonic()
    previous = None
    while True:
        if terminal.exists():
            with terminal.open("r") as source:
                manifest = json.load(source)
            exported_model(manifest)
            return manifest
        state = iris.job_state(JobName.from_wire(coordinator))
        if state != previous:
            print(f"Training/export coordinator: {state}", flush=True)
            previous = state
        if is_job_finished(state) and state != JobState.SUCCEEDED:
            raise RuntimeError(f"Training/export coordinator ended in {state}")
        if time.monotonic() - started > timeout:
            raise TimeoutError("Timed out waiting for a verified terminal export")
        time.sleep(30)


def main():
    from iris.cli.connect import open_iris_client

    from .evaluate_iris import evaluate_endpoint
    from .serve_evaluation import submit_server
    from .stage import upload_directory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terminal-uri", required=True)
    parser.add_argument("--coordinator", required=True)
    parser.add_argument("--server-name", required=True, help="A fresh Iris job name for this evaluation attempt")
    parser.add_argument("--cluster-config", type=Path, default=Path("lib/iris/config/marin.yaml"))
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results-uri", required=True)
    parser.add_argument("--wandb-run", help="Optional entity/project/run_id to receive held-out summary metrics")
    parser.add_argument("--wait-hours", type=float, default=24)
    args = parser.parse_args()
    before = load_evaluation(args.baseline)
    if hashlib.sha256(args.tasks.read_bytes()).hexdigest() != before[0]["tasks_sha256"]:
        raise ValueError("Follow-up tasks do not match the complete baseline cohort")
    args.output.mkdir(parents=True, exist_ok=True)

    def status(state, **details):
        payload = {
            "state": state,
            "updated_at": datetime.now(UTC).isoformat(),
            "terminal_uri": args.terminal_uri,
            "coordinator": args.coordinator,
            **details,
        }
        # Mutable local progress stays outside the immutable results directory.
        args.output.with_name(args.output.name + "-status.json").write_text(json.dumps(payload, indent=2) + "\n")

    status("waiting_for_export")
    try:
        with open_iris_client(config_file=args.cluster_config, workspace=Path.cwd()) as iris:
            manifest = wait_for_export(iris, args.terminal_uri, args.coordinator, args.wait_hours * 3600)
            model = exported_model(manifest)
            (args.output / "training-terminal.json").write_text(json.dumps(manifest, indent=2) + "\n")
            evaluation = args.output / "validation"
            complete = False
            if (evaluation / "summary.json").exists():
                summary = json.loads((evaluation / "summary.json").read_text())
                complete = summary["completed"] == summary["expected"]
            if not complete:
                server = submit_server(iris, model["policy_export_uri"], args.server_name)
                try:
                    status("evaluating", server=str(server), model=model)
                    evaluate_endpoint(
                        argparse.Namespace(
                            endpoint="/serve/" + args.server_name,
                            cluster_config=args.cluster_config,
                            tasks=args.tasks,
                            verifier=args.verifier,
                            model_identity=model["policy_export_uri"],
                            output=evaluation,
                            concurrency=16,
                            ready_timeout=1800,
                        )
                    )
                finally:
                    server.cancel()
                    print("Released evaluation server", str(server), flush=True)
            after = load_evaluation(evaluation)
            if after[0]["model_identity"] != model["policy_export_uri"]:
                raise ValueError("Existing evaluation belongs to a different checkpoint")
            comparison = compare(*before, *after)
            (args.output / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
            status("preserving_results", model=model, results_uri=args.results_uri)
            upload_directory(args.output, args.results_uri)
            if args.wandb_run:
                import wandb

                run = wandb.Api().run(args.wandb_run)
                prefix = "heldout/validation/"
                run.summary[prefix + "accuracy"] = comparison["trained"]["task_weighted_accuracy"]
                run.summary[prefix + "accuracy_delta"] = comparison["task_weighted_accuracy_delta"]
                run.summary[prefix + "family_macro_accuracy"] = comparison["trained"]["family_macro_accuracy"]
                run.summary[prefix + "results_uri"] = args.results_uri
                for family, row in comparison["families"].items():
                    run.summary[prefix + family + "/accuracy"] = row["trained_accuracy"]
                    run.summary[prefix + family + "/accuracy_delta"] = row["accuracy_delta"]
                run.summary.update()
            status("complete", model=model, results_uri=args.results_uri)
            print(json.dumps({"comparison": str(args.output / "comparison.json"), "results_uri": args.results_uri}))
    except Exception as error:  # noqa: BLE001 - sanitize all provider exception messages before logging
        # API exceptions can contain scoped endpoint URLs. Keep status/logs credential-free.
        status("failed", error_type=type(error).__name__)
        raise RuntimeError(f"Follow-up failed: {type(error).__name__}; inspect the saved run status") from None


if __name__ == "__main__":
    main()
