"""Freeze and score the original PDBThink benchmark using its existing verifier.

Run with the PDBThink environment. No files in that repository are modified.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(args):
    from pdbthink.evaluation.runner import dataset_fingerprint
    from pdbthink.evaluation.score import _scoring_parameters
    from scripts.clean_coordinate import frozen_inputs
    from transformers import AutoTokenizer

    instances, renders = frozen_inputs()
    instances = {r.semantic_instance_id: r for r in instances}
    freeze = json.loads((args.repo / "configs/clean_coordinate_v4.json").read_text())
    old = json.loads((args.repo / "runs/clean_coordinate_v4/snowball/plan.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    jobs, excluded = [], []
    for render in sorted(renders, key=lambda r: r.render_id):
        messages = [
            {"role": "system", "content": render.system_prompt},
            {"role": "user", "content": render.user_prompt},
        ]
        tokens = len(
            tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, enable_thinking=True, return_dict=True
            )["input_ids"]
        )
        row = {
            "render_id": render.render_id,
            "family": render.question_family,
            "protein_group_id": render.protein_group_id,
            "prompt_tokens": tokens,
            "system_sha256": digest(render.system_prompt.encode()),
            "user_sha256": digest(render.user_prompt.encode()),
            "gold_sha256": render.gold_sha256,
            "gold": render.gold_answer,
            "answer_schema": render.answer_schema,
            "parameters": _scoring_parameters(instances[render.semantic_instance_id]),
        }
        if tokens >= 32768:
            excluded.append({**row, "reason": "native_context_exceeded"})
        else:
            row["body"] = {
                "messages": messages,
                "max_tokens": 32768 - tokens,
                "temperature": 0.6,
                "top_p": 0.95,
                "seed": 20260929,
                "top_k": 20,
                "stop_token_ids": [128001, 128009],
                "chat_template_kwargs": {"enable_thinking": True},
                "tools": None,
                "tool_choice": "none",
            }
            jobs.append(row)
    assert dataset_fingerprint(renders) == freeze["dataset_fingerprint"]
    assert {j["render_id"] for j in jobs} == {j["render_id"] for j in old["jobs"]}
    previous = {j["render_id"]: j for j in old["jobs"]}
    for job in jobs:
        body = dict(previous[job["render_id"]]["body"])
        body.pop("model")
        assert body == job["body"], job["render_id"]
    source_hashes = {
        str(p.relative_to(args.repo)): digest(p.read_bytes())
        for folder in ("src/pdbthink", "scripts")
        for p in sorted((args.repo / folder).rglob("*.py"))
    }
    plan = {
        "version": "snowball-rl-original-benchmark-v1",
        "benchmark_version": freeze["version"],
        "dataset_fingerprint": freeze["dataset_fingerprint"],
        "total": len(renders),
        "protocol": "Exactly the existing clean-coordinate-v4 Snowball requests, with only model identity changed; one completion per question",
        "headline": "Strict original benchmark score; terminal-box normalization reported separately for both models",
        "context": 32768,
        "selected": len(jobs),
        "excluded": len(excluded),
        "selected_by_family": dict(sorted(Counter(j["family"] for j in jobs).items())),
        "chat_template_sha256": digest(tokenizer.chat_template.encode()),
        "source_git_head": subprocess.check_output(
            ["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_file_sha256": source_hashes,
        "jobs": jobs,
        "exclusions": excluded,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "plan.json"
    if target.exists() and json.loads(target.read_text()) != plan:
        raise ValueError("Cannot replace frozen benchmark plan")
    target.write_text(json.dumps(plan, indent=2) + "\n")
    print(
        json.dumps({k: plan[k] for k in ("total", "selected", "excluded", "selected_by_family", "dataset_fingerprint")})
    )


def score(args):
    import numpy as np
    from pdbthink.evaluation.tool_policy import response_tool_events
    from pdbthink.scoring import score_response
    from pdbthink.scoring.boxed import normalise_terminal_box

    plan = json.loads((args.output / "plan.json").read_text())
    contracts = {m: json.loads((args.output / m / "contract.json").read_text()) for m in ("baseline", "trained")}
    for model, contract in contracts.items():
        assert contract["plan_sha256"] == digest((args.output / "plan.json").read_bytes())
        assert json.loads((args.output / model / "complete.json").read_text())["completed"] == plan["selected"]
    for key in ("concurrency", "server", "gpus", "sampling"):
        assert contracts["baseline"][key] == contracts["trained"][key]
    for name, expected in plan["source_file_sha256"].items():
        if digest((args.repo / name).read_bytes()) != expected:
            raise ValueError("Original scorer source changed after freezing: " + name)
    results = []
    for job in plan["jobs"]:
        name = digest(job["render_id"].encode())[:20] + ".json"
        scores = {}
        for model in ("baseline", "trained"):
            record = json.loads((args.output / model / name).read_text())
            assert record["render_id"] == job["render_id"]
            assert {k: v for k, v in record["request"].items() if k != "model"} == job["body"]
            body = record["response"]
            assert len(body["choices"]) == 1
            assert body["usage"]["prompt_tokens"] == job["prompt_tokens"]
            assert not response_tool_events(body)
            choice = body["choices"][0]
            message = choice["message"]
            text = message.get("content") or ""
            truncated = choice["finish_reason"] == "length"
            common = {
                "parameters": job["parameters"],
                "truncated": truncated,
                "provider_refusal": bool(message.get("refusal")),
            }
            strict = score_response(text, job["answer_schema"], job["gold"], **common)
            normalized = normalise_terminal_box(text)
            boxed = score_response(normalized["scoring_text"], job["answer_schema"], job["gold"], **common)
            for outcome in (strict, boxed):
                outcome["correct"] = bool(outcome["score"].get("correct")) and not any(
                    outcome[k] for k in ("format_error", "refusal", "truncated")
                )
            scores[model] = {
                "strict": strict,
                "boxed": boxed,
                "box_normalized": normalized["changed"],
                "usage": body["usage"],
                "finish_reason": choice["finish_reason"],
                "response_sha256": digest(json.dumps(body, sort_keys=True).encode()),
            }
        results.append({k: job[k] for k in ("render_id", "family", "protein_group_id", "prompt_tokens")} | scores)
    summary = {
        "n": len(results),
        "total_benchmark": plan["total"],
        "excluded": plan["excluded"],
        "dataset_fingerprint": plan["dataset_fingerprint"],
        "selection": json.loads((args.output / "selection.json").read_text()),
        "contracts": contracts,
        "source_groups": len({r["protein_group_id"] for r in results}),
        "results": results,
    }
    summary["response_statistics"] = {
        model: {
            "total_output_tokens": sum(r[model]["usage"]["completion_tokens"] for r in results),
            "mean_output_tokens": sum(r[model]["usage"]["completion_tokens"] for r in results) / len(results),
            "finish_reasons": dict(Counter(r[model]["finish_reason"] for r in results)),
        }
        for model in ("baseline", "trained")
    }
    for policy in ("strict", "boxed"):
        groups = defaultdict(list)
        for r in results:
            groups[r["family"]].append(r)
        families = {
            f: {"n": len(rows), **{m: sum(r[m][policy]["correct"] for r in rows) for m in ("baseline", "trained")}}
            for f, rows in sorted(groups.items())
        }
        metric = {"families": families}
        for m in ("baseline", "trained"):
            metric[m] = {
                "correct": sum(r[m][policy]["correct"] for r in results),
                "accuracy": sum(r[m][policy]["correct"] for r in results) / len(results),
                "family_macro": sum(v[m] / v["n"] for v in families.values()) / len(families),
                **{k: sum(r[m][policy][k] for r in results) for k in ("format_error", "refusal", "truncated")},
            }
        metric["gains"] = sum(not r["baseline"][policy]["correct"] and r["trained"][policy]["correct"] for r in results)
        metric["losses"] = sum(
            r["baseline"][policy]["correct"] and not r["trained"][policy]["correct"] for r in results
        )
        metric["difference_pp"] = 100 * (metric["trained"]["accuracy"] - metric["baseline"]["accuracy"])
        metric["family_macro_difference_pp"] = 100 * (
            metric["trained"]["family_macro"] - metric["baseline"]["family_macro"]
        )
        protein_groups = defaultdict(list)
        for row in results:
            protein_groups[row["protein_group_id"]].append(row)
        units = list(protein_groups.values())
        sizes = np.array([len(rows) for rows in units])
        differences = np.array(
            [
                sum(int(r["trained"][policy]["correct"]) - int(r["baseline"][policy]["correct"]) for r in rows)
                for rows in units
            ]
        )
        draws = np.random.default_rng(17).integers(0, len(units), (50000, len(units)))
        samples = differences[draws].sum(axis=1) / sizes[draws].sum(axis=1)
        metric["group_bootstrap_95_delta_pp"] = (100 * np.quantile(samples, [0.025, 0.975])).tolist()
        metric["bootstrap_method"] = (
            "50000 paired protein_group_id resamples, seed 17; source-composition uncertainty only, not decoding or training-seed uncertainty"
        )
        summary[policy] = metric
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps({p: {k: v for k, v in summary[p].items() if k != "families"} for p in ("strict", "boxed")}, indent=2)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "score"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    args = parser.parse_args()
    sys.path[:0] = [str(args.repo), str(args.repo / "src")]
    {"prepare": prepare, "score": score}[args.action](args)


if __name__ == "__main__":
    main()
