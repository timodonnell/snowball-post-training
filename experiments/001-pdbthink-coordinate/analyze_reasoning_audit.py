"""Independent geometry checks and full-cohort controls for the frozen reasoning audit.

Reads displayed PDB coordinates, never model-generated coordinates. The torsion
implementation is cross-checked against the projected-vector definition.
Requires the project's PyArrow environment plus NumPy.
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from snowball_pdbthink.scoring import load_scorer, score

AUDIT = Path("runs/001/reasoning-audit")
PREPARED = Path("data/pdbthink-001-v3")
RESULTS = Path(__file__).resolve().parent / "results" / "reasoning-audit"


def atoms_for(prompt):
    out = {}
    for message in prompt:
        for line in message["content"].splitlines():
            if line[:6] in ("ATOM  ", "HETATM"):
                key = (line[21], int(line[22:26]), line[12:16].strip())
                assert key not in out
                out[key] = [float(line[a : a + 8]) for a in (30, 38, 46)]
    return out


def residue_key(name):
    chain, residue = name.split(":")
    return chain, int(residue[1:])


def torsion_bin(angle):
    return "g+" if 0 <= angle < 120 else "g-" if -120 <= angle < 0 else "t"


def torsion(points):
    a, b, c, d = np.asarray(points, float)
    u, v, w = b - a, c - b, d - c
    n0, n1 = np.cross(u, v), np.cross(v, w)
    assert np.linalg.norm(v) > 0 and np.linalg.norm(n0) > 0 and np.linalg.norm(n1) > 0
    x = float(n0 @ n1)
    y = float(np.linalg.norm(v) * (n0 @ w))
    angle = (math.degrees(math.atan2(y, x)) + 180) % 360 - 180
    vunit = v / np.linalg.norm(v)
    p = -u - (-u @ vunit) * vunit
    q = w - (w @ vunit) * vunit
    alternate = math.degrees(math.atan2(np.cross(vunit, p) @ q, p @ q))
    assert abs((angle - alternate + 180) % 360 - 180) < 1e-7
    wrong = math.degrees(math.atan2(n0 @ w, x))
    return {
        "angle": angle,
        "missing_bond_norm_angle": wrong,
        "missing_bond_norm_bin": torsion_bin(wrong),
        "bin": torsion_bin(angle),
        "points": points,
        "n0": n0.tolist(),
        "n1": n1.tolist(),
        "x": x,
        "y": y,
    }


def main():
    tasks = pq.read_table(PREPARED / "validation.parquet").to_pylist()
    sample = json.loads((AUDIT / "sample.json").read_text())
    notes = json.loads((RESULTS / "review-notes.json").read_text())
    assert set(map(int, notes)) == set(range(100))
    selected = {r["id"] for r in sample}
    native = {
        r["path"]: r for r in json.loads(Path("runs/001/regression-audit/evals/384/native-responses.json").read_text())
    }
    scorer = load_scorer(PREPARED / "native_verifier")
    geometry, rotamers, nearest = {}, [], []
    populations = defaultdict(
        lambda: {
            "n": 0,
            "correct0": 0,
            "correct384": 0,
            "gold": Counter(),
            "parsed0": Counter(),
            "parsed384": Counter(),
        }
    )
    for task in tasks:
        gold = json.loads(task["reward_model"]["ground_truth"])
        p, family, expected = gold["parameters"], task["family"], gold["gold_answer"]["value"]
        after = native[task["path"]]
        text = "".join(m.get("content") or "" for m in after["messages"])
        result = score(scorer, text, gold, truncated=after["stop_reason"] == "length")
        assert result["reward"] == after["reward"]
        base = json.loads((Path("runs/001/baseline-validation-v2") / (task["path"] + ".json")).read_text())["result"]
        pop = populations[family]
        pop["n"] += 1
        pop["correct0"] += int(base["reward"])
        pop["correct384"] += int(result["reward"])
        pop["gold"][json.dumps(expected, sort_keys=True)] += 1
        pop["parsed0"][json.dumps(base["outcome"]["parsed"]["value"], sort_keys=True)] += 1
        pop["parsed384"][json.dumps(result["outcome"]["parsed"]["value"], sort_keys=True)] += 1
        if family not in ("S09", "G03") and task["path"] not in selected:
            continue
        ats = atoms_for(task["prompt"])
        if family == "S09":
            points = [ats[(*residue_key(p["residue"]), n)] for n in ("N", "CA", "CB", p["atom4"])]
            g = torsion(points)
            assert g["bin"] == expected
            rotamers.append({"id": task["path"], **g})
        elif family == "G01":
            points = []
            for key in ("atom1", "atom2"):
                residue, atom = p[key].rsplit(":", 1)
                points.append(ats[(*residue_key(residue), atom)])
            dist = math.dist(*points)
            assert abs(dist - expected) < 0.001
            g = {"points": points, "distance": dist}
        elif family == "P03":
            xyz = ats[(*residue_key(p["residue"]), p["atom"])]
            assert xyz == expected
            g = {"xyz": xyz}
        elif family == "G03":
            residues = defaultdict(list)
            for key, xyz in ats.items():
                residues[key[:2]].append(xyz)
            distances = {
                c: min(
                    (math.dist(a, b), a, b)
                    for a in residues[residue_key(p["target"])]
                    for b in residues[residue_key(c)]
                )
                for c in p["candidates"]
            }
            ca = {
                c: math.dist(ats[(*residue_key(p["target"]), "CA")], ats[(*residue_key(c), "CA")])
                for c in p["candidates"]
            }
            assert min(distances, key=distances.get) == expected
            proxy = min(ca, key=ca.get)
            predicted = result["outcome"]["parsed"]["value"]
            g = {"min_heavy_atom_distances": distances, "ca_distances": ca, "ca_nearest": proxy}
            nearest.append(
                {
                    "id": task["path"],
                    "gold": expected,
                    "ca_proxy": proxy,
                    "prediction384": predicted,
                    "proxy_correct": proxy == expected,
                    "matches_proxy": predicted == proxy,
                    "scored_correct": bool(result["reward"]),
                }
            )
        elif family == "S01":
            # Verify the positive witness, not merely a CA or sidechain-centroid proxy.
            target, partner = p["residue"], expected
            charged = {"K": ("NZ",), "R": ("NE", "NH1", "NH2"), "D": ("OD1", "OD2"), "E": ("OE1", "OE2")}
            pairs = [
                (math.dist(ats[(*residue_key(target), a)], ats[(*residue_key(partner), b)]), a, b)
                for a in charged[target.split(":")[1][0]]
                for b in charged[partner.split(":")[1][0]]
            ]
            best = min(pairs)
            assert best[0] <= 4
            g = {
                "charged_atom_pairs": sorted(pairs),
                "minimum": best,
                "target": target,
                "partner": partner,
                "target_points": {a: ats[(*residue_key(target), a)] for a in charged[target.split(":")[1][0]]},
                "partner_points": {a: ats[(*residue_key(partner), a)] for a in charged[partner.split(":")[1][0]]},
            }
        else:
            continue
        if task["path"] in selected:
            geometry[task["path"]] = g
    out = {"sample": geometry, "all_s09": rotamers, "all_g03": nearest}
    controls = {
        "s09_missing_norm_correct": sum(r["bin"] == r["missing_bond_norm_bin"] for r in rotamers),
        "s09_total": len(rotamers),
        "g03_proxy_correct": sum(r["proxy_correct"] for r in nearest),
        "g03_matches_proxy": sum(r["matches_proxy"] for r in nearest),
        "g03_total": len(nearest),
        "g03_when_proxy_wrong": {
            "n": sum(not r["proxy_correct"] for r in nearest),
            "model_correct": sum(r["scored_correct"] for r in nearest if not r["proxy_correct"]),
            "model_matches_wrong_proxy": sum(r["matches_proxy"] for r in nearest if not r["proxy_correct"]),
        },
        "g03_sample_proxy_matches": sum(r["matches_proxy"] for r in nearest if r["id"] in selected),
        "selected_source_groups": len({r["group"] for r in sample}),
        "reviewed_questions": len(sample),
        "reviewed_responses": 2 * len(sample),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "geometry.json").write_text(json.dumps(out, indent=2) + "\n")
    (RESULTS / "statistics.json").write_text(
        json.dumps({"population": dict(populations), "controls": controls}, indent=2) + "\n"
    )
    annotations = [
        {
            "index": i,
            "id": r["id"],
            "family": r["family"],
            "group": r["group"],
            "transition": r["transition"],
            "gold": r["gold"],
            "review": notes[str(i)],
            "steps": {
                s: {
                    "text_sha256": a["text_sha256"],
                    "reward": a["result"]["reward"],
                    "parsed": a["result"]["outcome"]["parsed"],
                    "truncated": a["result"]["truncated"],
                    "refusal": a["result"]["refusal"],
                }
                for s, a in r["answers"].items()
            },
        }
        for i, r in enumerate(sample)
    ]
    (RESULTS / "annotations.json").write_text(json.dumps(annotations, indent=2) + "\n")
    (RESULTS / "sample-manifest.json").write_bytes((AUDIT / "sample-manifest.json").read_bytes())
    print(json.dumps(controls, indent=2))
    for f in ("S03", "S04", "S05", "S09"):
        print(f, json.dumps(populations[f]))


if __name__ == "__main__":
    main()
