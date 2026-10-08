"""Select a banked checkpoint using complete validation, then export it natively."""

import hashlib
import json
import math
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from .pins import SKYRL_REVISION
from .recover_export import recover_export


def validation_point(step, metrics, counts):
    scores = {}
    for family, count in counts.items():
        if metrics.get(f"eval/{family}/sequences") != count:
            raise ValueError("Checkpoint selection requires the complete validation cohort")
        score = metrics[f"eval/{family}/avg_score"]
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid validation accuracy")
        scores[family] = score
    return {
        "step": step,
        "tasks": sum(counts.values()),
        "task_weighted_accuracy": sum(scores[f] * counts[f] for f in counts) / sum(counts.values()),
        "family_macro_accuracy": sum(scores.values()) / len(scores),
        "family_accuracy": scores,
    }


def best_checkpoint(points):
    return max(points, key=lambda p: (p["task_weighted_accuracy"], p["family_macro_accuracy"], -p["step"]))


def preserve_verification(path, record):
    if path.exists():
        previous = json.loads(path.read_text())
        for key in ("model", "training_job", "resolved_config_sha256", "export_manifest_sha256", "checks"):
            if previous[key] != record[key]:
                raise ValueError("Cannot change a recorded export verification")
    else:
        path.write_text(json.dumps(record, indent=2) + "\n")


def select_and_export(iris, args, status):
    from connectrpc.code import Code
    from connectrpc.errors import ConnectError
    from iris.cluster.types import JobName
    from iris.resources.state import JobState, is_job_finished
    from marin.rl.skyrl import _launcher_command
    from rigging.filesystem.storage_path import StoragePath

    prepared = json.loads((args.tasks.parent / "manifest.json").read_text())
    counts = prepared["family_counts"]["validation"]
    run_root = StoragePath(args.terminal_uri).parent
    resolved = run_root / "resolved-launch.yaml"
    deadline = time.monotonic() + args.wait_hours * 3600

    def read_json(path):
        with path.open("r") as source:
            return json.load(source)

    def wait_tick():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for the scale-up run")
        time.sleep(60)

    while not resolved.exists():
        if is_job_finished(iris.job_state(JobName.from_wire(args.coordinator))):
            raise RuntimeError("Coordinator ended before writing the resolved configuration")
        wait_tick()
    resolved_record = read_json(resolved)
    config = resolved_record["config"]
    if config["runtime"]["launcher_commit"] != SKYRL_REVISION:
        raise ValueError("Unexpected scale-up runtime")
    if config["inputs"]["validation_data"][0]["relative_path"] != "validation.parquet":
        raise ValueError("Selection requires full validation during training")
    data_uri = config["inputs"]["validation_data"][0]["uri"]
    remote_prepared = read_json(StoragePath(data_uri) / "manifest.json")
    if remote_prepared != prepared:
        raise ValueError("Training validation artifact differs from the local frozen cohort")
    trainer = config["skyrl"]["trainer"]
    final_step, interval = trainer["max_steps"], trainer["eval_interval"]
    initial_step = int(Path(trainer["resume_path"]).name.removeprefix("global_step_"))
    expected = sorted(
        {initial_step, final_step, *range((initial_step // interval + 1) * interval, final_step, interval)}
    )
    training_job = args.coordinator + "/" + config["iris"]["job_name"]
    points = {}
    summary_step = None
    while True:
        try:
            state = iris.job_state(JobName.from_wire(training_job))
        except ConnectError as error:
            if error.code != Code.NOT_FOUND or is_job_finished(iris.job_state(JobName.from_wire(args.coordinator))):
                raise
            wait_tick()
            continue
        for step in expected:
            path = run_root / "exports" / "dumped_evals" / f"global_step_{step}_evals" / "aggregated_results.jsonl"
            if step not in points and path.exists():
                with path.open("r") as source:
                    rows = [json.loads(line) for line in source if line.strip()]
                points[step] = validation_point(step, rows[-1], counts)
        curve = [points[step] for step in sorted(points)]
        if args.wandb_run and curve and curve[-1]["step"] != summary_step:
            import wandb

            try:
                run = wandb.Api().run(args.wandb_run)
                for name, point in (("latest", curve[-1]), ("best", best_checkpoint(curve))):
                    prefix = "development/full_validation/" + name + "/"
                    for key in ("step", "tasks", "task_weighted_accuracy", "family_macro_accuracy"):
                        run.summary[prefix + key] = point[key]
                run.summary.update()
                summary_step = curve[-1]["step"]
            except wandb.errors.CommError:
                print("W&B summary update unavailable; native validation scores remain saved", flush=True)
        status(
            "training",
            training_job=training_job,
            training_state=str(state),
            validation=curve,
            best_validation=best_checkpoint(curve) if curve else None,
        )
        if state == JobState.SUCCEEDED:
            break
        if is_job_finished(state):
            raise RuntimeError("Scale-up training ended without success; banked checkpoints remain available")
        wait_tick()
    if set(points) != set(expected):
        raise ValueError("Completed training is missing a scheduled full validation evaluation")
    selected = best_checkpoint(curve)
    selection = {
        "rule": "Highest full-validation task-weighted accuracy; ties by family macro accuracy, then earlier step",
        "selected": selected,
        "curve": curve,
        "tasks_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "training_job": training_job,
    }
    selection_path = args.output / "validation-selection.json"
    if selection_path.exists() and read_json(selection_path) != selection:
        raise ValueError("Cannot change recorded checkpoint selection")
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")

    def wait_publication(terminal_uri, coordinator, step, *, intermediate=False, process=None):
        model_root = StoragePath(terminal_uri).parent / "exports" / f"global_step_{step}" / "policy"
        export_deadline = time.monotonic() + 7200
        while not (model_root / ".marinskyrl-model-manifest.json").exists():
            if process is not None and process.poll() is not None:
                raise RuntimeError("Native checkpoint export exited before publishing a complete model")
            if time.monotonic() > export_deadline:
                raise TimeoutError("Native checkpoint export did not publish within two hours")
            time.sleep(30)
        return recover_export(iris, terminal_uri, coordinator, step, prepared, selected_intermediate=intermediate)

    # Let the canonical terminal export complete. If job teardown stalls again,
    # verify the publication before releasing this run's stale coordinator tree.
    status("waiting_for_terminal_export", selected=selected)
    terminal_model, terminal_record = wait_publication(args.terminal_uri, args.coordinator, final_step)
    settle_deadline = time.monotonic() + 180
    while (
        not is_job_finished(iris.job_state(JobName.from_wire(args.coordinator))) and time.monotonic() < settle_deadline
    ):
        time.sleep(30)
    if not is_job_finished(iris.job_state(JobName.from_wire(args.coordinator))):
        iris.cancel_job(JobName.from_wire(args.coordinator))
        terminal_record["cleanup"] = "Cancelled stale coordinator after verifying complete terminal publication"
    preserve_verification(args.output / "terminal-export-verification.json", terminal_record)

    step = selected["step"]
    if step == final_step:
        return terminal_model, selection
    if step == initial_step:
        initial = json.loads(args.initial_export_record.read_text())
        if initial["model"]["checkpoint_root"] + f"/global_step_{step}" != trainer["resume_path"]:
            raise ValueError("Starting export does not match the resumed checkpoint")
        model, record = recover_export(
            iris, initial["model"]["terminal_manifest_uri"], initial["coordinator"], step, prepared
        )
    else:
        source = args.output / "resolved-training-config.json"
        source.write_text(json.dumps(config, indent=2) + "\n")
        model_root = run_root / "exports" / f"global_step_{step}" / "policy"
        process = None
        if not (model_root / ".marinskyrl-model-manifest.json").exists():
            job_name = f"snowball-pdbthink-selected-step{step}-" + datetime.now(UTC).strftime("%Y%m%d%H%M%S")
            command = _launcher_command(
                f"marinskyrl @ git+https://github.com/marin-community/MarinSkyRL.git@{SKYRL_REVISION}", "unused"
            )
            command = command[: command.index("marinskyrl")] + [
                "python",
                "-m",
                "cloud.iris.export_hf_checkpoint",
                "--request",
                config["artifacts"]["checkpoint_root"] + f"/global_step_{step}",
                "--launch-config",
                str(source.absolute()),
                "--gpu-variant",
                "H100",
                "--cluster",
                "cw-rno2a",
                "--cluster-config",
                str(Path("lib/iris/config/cw-rno2a.yaml").absolute()),
                "--target-cluster",
                "cw-rno2a",
                "--parent-cluster-config",
                str(args.cluster_config.absolute()),
                "--cpu",
                "64",
                "--memory",
                "1800GB",
                "--disk",
                "1000GB",
                "--job-name",
                job_name,
            ]
            status("exporting_selected_checkpoint", selected=selected, export_job="/bizon/" + job_name)
            log_path = args.output.with_name(args.output.name + "-selected-export.log")
            with log_path.open("ab") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            model, record = wait_publication(
                args.terminal_uri, args.coordinator, step, intermediate=True, process=process
            )
        finally:
            if process is not None:
                try:
                    process.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    iris.cancel_job(JobName.from_wire("/bizon/" + job_name))
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=30)
    preserve_verification(args.output / "selected-export-verification.json", record)
    return model, selection
