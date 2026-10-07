#!/usr/bin/env python3
"""One-factor-at-a-time sweep over the ranges SONIC training actually randomised.

This is not tuning. The nominal strict baseline is fixed first and never
revisited; this walks one training-distribution axis at a time and records where
the untouched policy starts to fail. Results land in their own directory and
carry the perturbation in their metadata, so they can never be confused with a
strict baseline row.

Only axes with a ``TRAIN_DIST`` entry in ``h2_policy_if.evaluation`` are offered,
and the runner itself refuses a value outside the training range. An axis
training never touched is ``OOD_SIM2REAL`` and belongs in a separate study with
its own justified ranges.

    python tools/ofat_sweep.py --axis robot_friction --points 5 \\
        --motion walk_forward_loop_001__A029 --seconds 10 --out-dir results/ofat
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

# evaluation.py depends only on the standard library, but the package __init__
# pulls in h2_joints, which needs numpy. This driver runs on the host, where the
# simulator dependencies deliberately are not installed, so load the one module
# directly by path instead of importing the package.
import importlib.util  # noqa: E402

_EVAL = Path(__file__).resolve().parents[1] / "h2_policy_if" / "evaluation.py"
_spec = importlib.util.spec_from_file_location("h2_eval_standalone", _EVAL)
ev = importlib.util.module_from_spec(_spec)
# dataclasses resolves field types through sys.modules, so the module has to be
# registered before it executes or every @dataclass in it raises.
sys.modules[_spec.name] = ev
_spec.loader.exec_module(ev)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backends  # noqa: E402

# Sweep axis -> the TRAIN_RANDOMIZATION key that bounds it. Kept in step with
# PERTURB_AXES in the two runners. Whether a given backend can express an axis
# is not restated here: it comes from evaluation.AXIS_SUPPORT via backends.py,
# so an axis is refused rather than approximated.
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


def grid(lo: float, hi: float, points: int) -> list[float]:
    """Evenly spaced sweep values including both endpoints."""
    if points < 2:
        return [(lo + hi) / 2.0]
    step = (hi - lo) / (points - 1)
    return [round(lo + i * step, 6) for i in range(points)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--axis", required=True, choices=sorted(AXES))
    parser.add_argument("--points", type=int, default=5, help="sweep points across the training range")
    parser.add_argument("--motion", default="walk_forward_loop_001__A029")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--out-dir", default="results/ofat")
    parser.add_argument("--profile", default="strict-native",
                        choices=("strict-native", "backend-recommended"),
                        help="strict-native uses the backend's own numerical defaults; "
                             "backend-recommended adds the solver settings that backend's "
                             "humanoid examples use, which are recorded in every result")
    parser.add_argument("--backend", default="mujoco", choices=sorted(backends.BACKENDS),
                        help="simulator to sweep on; each is driven from its own folder")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    how = backends.support(ev, args.axis, args.backend)
    if how == "unsupported":
        raise SystemExit(
            f"{args.backend} cannot express {args.axis} exactly "
            f"({ev.AXIS_SUPPORT[backends._support_name(args.axis)]['note']}). "
            f"Run this axis on a backend that can; approximating it would make the "
            f"number mean something other than the training parameter."
        )
    key = AXES[args.axis]
    sourced = ev.TRAIN_RANDOMIZATION[key]
    # torso_com_offset_m is per-component; the rest are a plain (lo, hi) pair.
    lo, hi = sourced.value[args.axis[-1]] if isinstance(sourced.value, dict) else sourced.value
    values = grid(lo, hi, args.points)

    print(f"axis        {args.axis}  ({sourced.grade})")
    print(f"backend     {args.backend}  [{how}]   profile {args.profile}")
    print(f"range       [{lo}, {hi}]   <- {sourced.source}")
    print(f"points      {values}")
    print(f"motion      {args.motion}   {args.seconds} s")
    print(f"out-dir     {args.out_dir}")
    print()
    if args.dry_run:
        return

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for value in values:
        out = os.path.join(args.out_dir, f"{args.axis}_{value:g}.json")
        cmd, cwd = backends.command(args.backend, motion=args.motion, seconds=args.seconds,
                                    out=out, perturbations={args.axis: value},
                                    profile=args.profile)
        print(f"-> {args.axis}={value:g}")
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        out = backends.host_path(args.backend, out)
        if not os.path.exists(out):
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
            print(f"   FAILED: {' | '.join(tail)}")
            continue
        with open(out) as fh:
            met = json.load(fh)["metrics"]
        rows.append((value, met))
        print(f"   survival {met['survival_s']:>6} s  end {met['termination_reason']:<10} "
              f"pelvis min {met['pelvis_height_min']}  sat {met['torque_saturation_rate']}")

    if not rows:
        raise SystemExit("no runs produced a result")

    print()
    print(f"{'value':>10}  {'survival':>9}  {'end':<11}  {'pelvis min':>11}  {'sat rate':>9}  {'|tau|max':>9}")
    print("-" * 70)
    for value, met in rows:
        print(f"{value:>10g}  {met['survival_s']:>9}  {met['termination_reason']:<11}  "
              f"{met['pelvis_height_min']:>11}  {met['torque_saturation_rate']:>9}  "
              f"{met['torque_abs_max']:>9}")
    failures = [v for v, m in rows if m["terminated"]]
    print()
    print(f"failures inside the training range: "
          f"{failures if failures else 'none'}")
    print("These are TRAIN_DIST results. Do not merge them with the strict nominal baseline.")


if __name__ == "__main__":
    main()
