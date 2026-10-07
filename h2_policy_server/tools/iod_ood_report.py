#!/usr/bin/env python3
"""Aggregate the IOD-vs-OOD ladder campaign into its own results folder.

Separate from campaign_report.py on purpose (that one is phase-keyed to the
sim2real campaign and, per the preservation rules, is not edited). Stdlib only.

    python3 iod_ood_report.py --name campaign_iod_ood_20260930
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
MAPPING = json.load(open(WS / "eval_datasets" / "tools" / "iod_ood_mapping.json"))
LEVEL_NAMES = {1: "idle", 2: "in-place arms", 3: "straight walk", 4: "turn/side-step",
               5: "approach+reach", 6: "lift", 7: "carry", 8: "put down",
               9: "bimanual", 10: "whole-body tool"}


def load(name: str) -> tuple[list[dict], list[Path]]:
    rows, dirs = [], []
    for d in (WS / "Mujoco" / "results" / name,
              WS / "IsaacLab" / "sim2sim_motion_retargeting" / "results" / name):
        f = d / "summary.csv"
        if not f.exists():
            continue
        dirs.append(d)
        track = {}
        tf = d / "tracking.csv"
        if tf.exists():
            track = {r["motion"]: r for r in csv.DictReader(open(tf))}
        for r in csv.DictReader(open(f)):
            meta = MAPPING.get(r["motion"], {})
            r["source"] = meta.get("source", "?")
            r["level"] = meta.get("level", 0)
            r["ok"] = r["termination_reason"] == "completed"
            r["joint_mae_rad"] = track.get(r["motion"], {}).get("joint_mae_rad", "")
            rows.append(r)
    return rows, dirs


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def table(headers, body):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="campaign_iod_ood_20260930")
    args = ap.parse_args()
    rows, dirs = load(args.name)
    out = WS / "IsaacLab" / "sim2sim_motion_retargeting" / "results" / args.name / "report"
    out.mkdir(parents=True, exist_ok=True)
    backends = sorted({r["backend"] for r in rows})
    print(f"  {len(rows)} rollouts from {len(dirs)} simulators: {backends}")

    with open(out / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), extrasaction="ignore")
        w.writeheader(); w.writerows(rows)

    lines = [f"# IOD vs OOD ladder — {args.name}", "",
             "IOD = clips from the SONIC training set (BONES-SEED `smpl_filtered`); "
             "OOD = OMOMO test split, provenance-independent of BONES-SEED. Both sides "
             "went through the byte-identical wbc_h12 retargeting pipeline "
             "(shared canonicalisation: neutral betas, GMR height handling).", ""]

    lines += ["## Success by source", ""]
    body = []
    for b in backends:
        for src in ("IOD", "OOD"):
            sel = [r for r in rows if r["backend"] == b and r["source"] == src]
            ok = sum(r["ok"] for r in sel)
            maes = [f(r["joint_mae_rad"]) for r in sel if f(r["joint_mae_rad"])]
            surv = [f(r["survival_s"]) for r in sel if f(r["survival_s"]) is not None]
            sats = [f(r["torque_saturation_rate"]) for r in sel if f(r["torque_saturation_rate"]) is not None]
            body.append([b, src, f"{ok}/{len(sel)}",
                         f"{statistics.median(maes):.3f}" if maes else "—",
                         f"{statistics.median(surv):.1f}" if surv else "—",
                         f"{statistics.median(sats):.2f}" if sats else "—"])
    lines += table(["backend", "source", "completed", "jMAE med [rad]",
                    "survival med [s]", "sat med"], body) + [""]

    lines += ["## Success by ladder level", ""]
    hdr = ["L", "level"] + [f"{b}<br>{s}" for b in backends for s in ("IOD", "OOD")]
    body = []
    for lv in range(1, 11):
        row = [lv, LEVEL_NAMES[lv]]
        for b in backends:
            for src in ("IOD", "OOD"):
                sel = [r for r in rows if r["backend"] == b and r["source"] == src
                       and r["level"] == lv]
                row.append(f"{sum(r['ok'] for r in sel)}/{len(sel)}" if sel else "—")
        body.append(row)
    lines += table(hdr, body) + [""]

    lines += ["## Per-motion detail", ""]
    body = [[r["level"], r["source"], r["motion"][:44], r["backend"],
             r["termination_reason"], r.get("fell_at_s") or "",
             r.get("pelvis_height_min"), r.get("joint_mae_rad"),
             r.get("torque_saturation_rate"), "yes" if r.get("video") else "—"]
            for r in sorted(rows, key=lambda r: (r["level"], r["source"], r["motion"], r["backend"]))]
    lines += table(["L", "src", "motion", "backend", "end", "at[s]",
                    "pelvis_min", "jMAE", "sat", "video"], body)
    (out / "summary.md").write_text("\n".join(lines))

    fails = [r for r in rows if not r["ok"]]
    fl = ["# IOD/OOD failures", ""]
    fl += table(["L", "src", "motion", "backend", "end", "at[s]", "pelvis_min", "video"],
                [[r["level"], r["source"], r["motion"][:44], r["backend"],
                  r["termination_reason"], r.get("fell_at_s") or "",
                  r.get("pelvis_height_min"), r.get("video") or "—"] for r in
                 sorted(fails, key=lambda r: (r["level"], r["source"]))]) if fails else ["none"]
    (out / "failures.md").write_text("\n".join(fl))

    for d in dirs:
        for sub in ("raw", "videos"):
            label = "mujoco" if "Mujoco" in str(d) else "isaacsim_physx"
            link = out / sub / label
            link.parent.mkdir(exist_ok=True)
            if not link.exists():
                link.symlink_to(os.path.relpath(d / sub, link.parent))
        p = d / "provenance.json"
        if p.exists() and not (out / "provenance.json").exists():
            shutil.copy(p, out / "provenance.json")
    print(f"  -> {out}/summary.md, failures.md, summary.csv")


if __name__ == "__main__":
    main()
