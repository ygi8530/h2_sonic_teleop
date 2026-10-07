#!/usr/bin/env python3
"""Sample several TRAIN_DIST axes at once and measure where the policy fails.

OFAT answers "how sensitive is the policy to one parameter"; this answers "what
happens when several move together", which is the situation a real robot is
actually in. Each rollout draws every requested axis independently from the range
SONIC training randomised it over, so the sampled population is the training
distribution rather than a guess.

Not tuning. The strict nominal baseline is fixed first and never revisited; these
results live in their own directory and every rollout records the exact parameter
vector that produced it, so a failure can be traced back to a combination.

    python tools/monte_carlo.py --rollouts 20 --seed 0 \\
        --axes robot_friction body_mass_scale torso_com_y \\
        --motion walk_forward_loop_001__A029 --seconds 15 --out-dir results/mc
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys

_EVAL = Path(__file__).resolve().parents[1] / "h2_policy_if" / "evaluation.py"
_spec = importlib.util.spec_from_file_location("h2_eval_standalone", _EVAL)
ev = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ev
_spec.loader.exec_module(ev)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backends  # noqa: E402

# Axis -> the TRAIN_RANDOMIZATION key that bounds it. Kept in step with the
# runners' PERTURB_AXES. Axes a backend cannot express exactly are rejected by
# that runner, not silently approximated here.
AXES = {
    "robot_friction": "robot_material.static_friction",
    "robot_dynamic_friction": "robot_material.dynamic_friction",
    "robot_restitution": "robot_material.restitution",
    "body_mass_scale": "body_mass_scale",
    "torso_com_x": "torso_com_offset_m",
    "torso_com_y": "torso_com_offset_m",
    "torso_com_z": "torso_com_offset_m",
    "joint_default_offset": "joint_default_pos_offset_rad",
}


def bounds(axis: str) -> tuple[float, float]:
    """Return the training range for an axis."""
    value = ev.TRAIN_RANDOMIZATION[AXES[axis]].value
    return value[axis[-1]] if isinstance(value, dict) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="seeds both the sampling and each rollout's pushes")
    parser.add_argument("--axes", nargs="+", default=["robot_friction", "body_mass_scale", "torso_com_y"],
                        choices=sorted(AXES))
    parser.add_argument("--motion", default="walk_forward_loop_001__A029")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--push", action="store_true", help="also replay training's push event")
    parser.add_argument("--out-dir", default="results/mc")
    parser.add_argument("--profile", default="strict-native",
                        choices=("strict-native", "backend-recommended"),
                        help="strict-native uses the backend's own numerical defaults; "
                             "backend-recommended adds the solver settings that backend's "
                             "humanoid examples use, which are recorded in every result")
    parser.add_argument("--backend", default="mujoco", choices=sorted(backends.BACKENDS),
                        help="simulator to sample on; each is driven from its own folder")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    refused = {a: backends.support(ev, a, args.backend) for a in args.axes}
    refused = {a: h for a, h in refused.items() if h == "unsupported"}
    if refused:
        raise SystemExit(
            f"{args.backend} cannot express {', '.join(refused)} exactly. Drop the axis "
            f"or move the study to a backend that supports it -- see evaluation.AXIS_SUPPORT."
        )

    rng = random.Random(args.seed)
    print(f"backend   {args.backend}   profile {args.profile}")
    print(f"axes      {', '.join(args.axes)}   (all TRAIN_DIST)")
    for axis in args.axes:
        lo, hi = bounds(axis)
        print(f"  {axis:24s} [{lo}, {hi}]   <- {ev.TRAIN_RANDOMIZATION[AXES[axis]].source}")
    print(f"rollouts  {args.rollouts}   motion {args.motion}   {args.seconds} s   seed {args.seed}")
    print(f"out-dir   {args.out_dir}")
    print()
    if args.dry_run:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    # The rollout runs as the container's own user; the directory this host-side
    # driver just created must be writable by it too.
    os.chmod(args.out_dir, 0o777)
    rows = []
    for i in range(args.rollouts):
        sample = {axis: round(rng.uniform(*bounds(axis)), 6) for axis in args.axes}
        out = os.path.join(args.out_dir, f"mc_{args.seed:03d}_{i:03d}.json")
        cmd, cwd = backends.command(
            args.backend, motion=args.motion, seconds=args.seconds, out=out,
            perturbations=sample, profile=args.profile,
            seed=args.seed * 1000 + i, push=args.push)
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        out = backends.host_path(args.backend, out)
        if not os.path.exists(out):
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-2:]
            print(f"  [{i:3d}] FAILED TO RUN: {' | '.join(tail)}")
            continue
        with open(out) as fh:
            met = json.load(fh)["metrics"]
        rows.append({"i": i, "sample": sample, "met": met})
        flag = "FELL" if met["terminated"] else "ok  "
        print(f"  [{i:3d}] {flag} surv {met['survival_s']:>6}s  pelvis min {met['pelvis_height_min']}"
              f"  sat {met['torque_saturation_rate']}   "
              + "  ".join(f"{a}={v:g}" for a, v in sample.items()))

    if not rows:
        raise SystemExit("no rollout produced a result")

    fell = [r for r in rows if r["met"]["terminated"]]
    mins = [r["met"]["pelvis_height_min"] for r in rows if r["met"]["pelvis_height_min"] is not None]
    sats = [r["met"]["torque_saturation_rate"] for r in rows if r["met"]["torque_saturation_rate"] is not None]

    print()
    print("== summary ==")
    print(f"  rollouts            {len(rows)}")
    print(f"  success rate        {1 - len(fell) / len(rows):.3f}   ({len(rows) - len(fell)}/{len(rows)})")
    if mins:
        print(f"  pelvis min          min {min(mins):.4f}  median {statistics.median(mins):.4f}  max {max(mins):.4f}")
    if sats:
        print(f"  torque saturation   min {min(sats):.3f}  median {statistics.median(sats):.3f}  max {max(sats):.3f}")
    worst = min(rows, key=lambda r: r["met"]["pelvis_height_min"] or 1e9)
    print(f"  worst rollout       #{worst['i']}  pelvis min {worst['met']['pelvis_height_min']}")
    print(f"                      {worst['sample']}")
    if fell:
        print("  failures:")
        for r in fell:
            print(f"    #{r['i']:3d} at {r['met']['fell_at_s']}s   {r['sample']}")
    else:
        print("  failures            none inside the training distribution")

    summary = os.path.join(args.out_dir, f"summary_seed{args.seed:03d}.json")
    with open(summary, "w") as fh:
        json.dump({
            "axes": {a: list(bounds(a)) for a in args.axes},
            "grade": "TRAIN_DIST", "backend": args.backend, "profile": args.profile,
            "motion": args.motion, "seconds": args.seconds, "seed": args.seed,
            "rollouts": len(rows),
            "success_rate": 1 - len(fell) / len(rows),
            "results": rows,
        }, fh, indent=2)
    print(f"\n  summary -> {summary}")
    print("  TRAIN_DIST results. Do not merge them with the strict nominal baseline.")


if __name__ == "__main__":
    main()
