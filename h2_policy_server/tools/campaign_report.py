#!/usr/bin/env python3
"""Aggregate every campaign phase into one results folder.

Pure stdlib on purpose: it runs on the host, where the simulators and numpy
deliberately are not installed. It never recomputes physics -- it only joins the
CSVs the batches already wrote and derives success rates, sensitivity rankings
and disagreement tables from them.

    python3 campaign_report.py --out <campaign dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import statistics

WS = Path("/home/yoonki/workspace")
MJ = WS / "Mujoco" / "results"
IL = WS / "IsaacLab" / "sim2sim_motion_retargeting" / "results"
CATEGORIES = json.load(open(WS / "eval_datasets" / "tools" / "motion_categories.json"))

#: phase -> the run directories that belong to it
PHASES = {
    "nominal": [MJ / "c1_nominal", IL / "c1_physx", IL / "c1_mjwarp"],
    "train_dist_ofat": [MJ / "c2_ofat", IL / "c2_physx_ofat"],
    "ood_sim2real": [MJ / "c3_ood", IL / "c3_physx_ood"],
    "rough_terrain": [IL / "c5_rough"],
}
MC_DIRS = [WS / "h2_policy_server" / "results" / "c4_mc", IL / "c4_mc"]


def load_rows() -> list[dict]:
    """Read every phase's summary.csv into one table."""
    rows = []
    for phase, dirs in PHASES.items():
        for d in dirs:
            f = d / "summary.csv"
            if not f.exists():
                continue
            for r in csv.DictReader(open(f)):
                r["phase"] = phase
                r["run_dir"] = d.name
                r["category"] = CATEGORIES.get(r["motion"], "?")
                r["ok"] = r["termination_reason"] == "completed"
                rows.append(r)
    return rows


def f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def pct(n, d):
    return f"{n}/{d} ({100.0 * n / d:.0f}%)" if d else "—"


def table(headers, body):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return out


def nominal_tables(rows):
    nom = [r for r in rows if r["phase"] == "nominal"]
    keys = sorted({(r["backend"], r["profile"]) for r in nom})
    lines = ["## 1. Nominal baseline — success by backend", ""]
    body = []
    for b, p in keys:
        sel = [r for r in nom if r["backend"] == b and r["profile"] == p]
        ok = sum(r["ok"] for r in sel)
        pmins = [f(r["pelvis_height_min"]) for r in sel if f(r["pelvis_height_min"]) is not None]
        sats = [f(r["torque_saturation_rate"]) for r in sel if f(r["torque_saturation_rate"]) is not None]
        body.append([b, p, pct(ok, len(sel)),
                     f"{min(pmins):.3f} / {statistics.median(pmins):.3f}" if pmins else "—",
                     f"{statistics.median(sats):.2f}" if sats else "—"])
    lines += table(["backend", "profile", "completed", "pelvis min (worst/median) [m]",
                    "sat (median)"], body)

    lines += ["", "## 2. Nominal success by motion category", ""]
    cats = sorted({r["category"] for r in nom})
    body = []
    for c in cats:
        row = [c]
        for b, p in keys:
            sel = [r for r in nom if r["category"] == c and r["backend"] == b and r["profile"] == p]
            row.append(pct(sum(r["ok"] for r in sel), len(sel)))
        body.append(row)
    lines += table(["category"] + [f"{b}<br>{p}" for b, p in keys], body)

    # disagreements: same motion, strict-native, different verdict across backends
    lines += ["", "## 3. Cross-simulator disagreements (nominal, strict-native)", ""]
    strict = [r for r in nom if r["profile"] == "strict-native"]
    by_motion = {}
    for r in strict:
        by_motion.setdefault(r["motion"], {})[r["backend"]] = r
    body = []
    for m, per in sorted(by_motion.items()):
        verdicts = {b: per[b]["termination_reason"] for b in per}
        if len({v == "completed" for v in verdicts.values()}) > 1:
            body.append([m, CATEGORIES.get(m, "?")]
                        + [verdicts.get(b, "—") for b in sorted(by_motion[m])])
    if body:
        backends = sorted({b for per in by_motion.values() for b in per})
        body = [[m, c] + [ (by_motion[m].get(b, {}) or {}).get("termination_reason", "—")
                           for b in backends] for m, c, *_ in body]
        lines += table(["motion", "category"] + backends, body)
    else:
        lines += ["Every motion got the same verdict on every backend."]
    return lines, nom


def sensitivity(rows, phase, title):
    """Per-axis damage ranking: falls, and worst pelvis-min drop vs same-motion nominal."""
    sel = [r for r in rows if r["phase"] == phase]
    if not sel:
        return [f"## {title}", "", "_no data yet_"]
    nominal = {(r["motion"], r["backend"]): f(r["pelvis_height_min"])
               for r in sel if r["condition"] == "nominal"}
    if not nominal:  # phase without its own nominals: fall back to the baseline phase
        nominal = {(r["motion"], r["backend"]): f(r["pelvis_height_min"])
                   for r in rows if r["phase"] == "nominal" and r["profile"] == "strict-native"}
    axes = {}
    for r in sel:
        if r["condition"] in ("nominal", "push"):
            axis = r["condition"]
        else:
            axis = r["condition"].rstrip("0123456789.-")
        if axis == "nominal":
            continue
        a = axes.setdefault(axis, {"n": 0, "fall": 0, "drops": [], "backends": set()})
        a["n"] += 1
        a["fall"] += not r["ok"]
        a["backends"].add(r["backend"])
        base = nominal.get((r["motion"], r["backend"]))
        pm = f(r["pelvis_height_min"])
        if base is not None and pm is not None:
            a["drops"].append(base - pm)
    lines = [f"## {title}", ""]
    body = []
    for axis, a in sorted(axes.items(), key=lambda kv: (-kv[1]["fall"],
                          -(max(kv[1]["drops"]) if kv[1]["drops"] else 0))):
        body.append([axis, a["n"], a["fall"],
                     f"{max(a['drops']):+.3f}" if a["drops"] else "—",
                     f"{statistics.median(a['drops']):+.3f}" if a["drops"] else "—",
                     ",".join(sorted(a["backends"]))])
    lines += table(["axis / condition", "runs", "falls", "worst Δpelvis_min [m]",
                    "median Δ [m]", "backends"], body)
    return lines


def fragile_motions(rows):
    """Motions that pass nominal but fail under some perturbation, per backend."""
    nominal_ok = {(r["motion"], r["backend"]) for r in rows
                  if r["phase"] == "nominal" and r["profile"] == "strict-native" and r["ok"]}
    out = {}
    for r in rows:
        if r["phase"] in ("train_dist_ofat", "ood_sim2real", "rough_terrain") \
                and not r["ok"] and (r["motion"], r["backend"]) in nominal_ok:
            out.setdefault((r["motion"], r["backend"]), []).append(
                (r["phase"], r["condition"], r["termination_reason"]))
    lines = ["## Nominal-pass but perturbation-fail", ""]
    body = [[m, b, len(conds),
             "; ".join(f"{c}[{ph[:4]}:{why}]" for ph, c, why in conds[:4])
             + (" …" if len(conds) > 4 else "")]
            for (m, b), conds in sorted(out.items(), key=lambda kv: -len(kv[1]))]
    lines += table(["motion", "backend", "failing conds", "examples"], body) if body \
        else ["_none observed_"]
    return lines


def mc_table():
    lines = ["## Monte Carlo (TRAIN_DIST, multi-axis, seed 0)", ""]
    body = []
    for root in MC_DIRS:
        for summ in sorted(root.glob("*/summary_seed*.json")):
            d = json.load(open(summ))
            body.append([d.get("backend", "mujoco"), summ.parent.name,
                         d["rollouts"], f"{d['success_rate']:.2f}",
                         ", ".join(d["axes"].keys())])
    lines += table(["backend", "motion", "rollouts", "success", "axes"], body) if body \
        else ["_no data yet_"]
    return lines


def failures_md(rows, out_dir):
    lines = ["# Every non-completed rollout", "",
             "A fall, a divergence or a NaN is a result, not an error. Videos exist "
             "where the backend can record.", ""]
    for phase in PHASES:
        sel = [r for r in rows if r["phase"] == phase and not r["ok"]]
        if not sel:
            continue
        lines += [f"## {phase} — {len(sel)}", ""]
        lines += table(["motion", "category", "backend", "profile", "condition",
                        "end", "at [s]", "pelvis min", "video"],
                       [[r["motion"], r["category"], r["backend"], r["profile"],
                         r["condition"], r["termination_reason"], r.get("fell_at_s", ""),
                         r.get("pelvis_height_min", ""),
                         "yes" if r.get("video") else "—"] for r in sel])
        lines += [""]
    (out_dir / "failures.md").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(IL / "campaign_sim2real"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    print(f"  {len(rows)} rollouts across {len({r['run_dir'] for r in rows})} run dirs")

    cols = list(rows[0].keys()) if rows else []
    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    lines = ["# Campaign summary — strict sim2sim, SONIC H2 EDU", ""]
    nom_lines, _ = nominal_tables(rows)
    lines += nom_lines + [""]
    lines += sensitivity(rows, "train_dist_ofat",
                         "4. TRAIN_DIST sensitivity (OFAT, representative clips)") + [""]
    lines += sensitivity(rows, "ood_sim2real",
                         "5. OOD_SIM2REAL sensitivity (operator-chosen ranges)") + [""]
    lines += sensitivity(rows, "rough_terrain", "6. Rough terrain") + [""]
    lines += fragile_motions(rows) + [""]
    lines += mc_table() + [""]
    (out / "summary.md").write_text("\n".join(lines))

    failures_md(rows, out)

    # provenance: take one baseline run's, they all print the same commit/sha
    for d in PHASES["nominal"]:
        p = d / "provenance.json"
        if p.exists():
            shutil.copy(p, out / "provenance.json")
            break
    # One symlink per phase run keeps every rollout JSON and video reachable
    # from the campaign folder without duplicating gigabytes.
    for name in ("raw", "videos"):
        base = out / name
        base.mkdir(exist_ok=True)
        for dirs in PHASES.values():
            for d in dirs:
                src = d / name
                if not src.exists():
                    continue
                link = base / d.name
                if link.is_symlink() or link.exists():
                    continue
                link.symlink_to(os.path.relpath(src, base))
    print(f"  -> {out}/summary.csv, summary.md, failures.md, provenance.json")


if __name__ == "__main__":
    main()
