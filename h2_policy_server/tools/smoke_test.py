#!/usr/bin/env python3
"""End-to-end check of the H2 EDU policy server. Needs no simulator.

Two phases:

1. **Transport** - every op round-trips, shapes and ranges are as declared.
2. **Open-loop sweep** - the reference clip is swept while the robot is held at
   its default pose and ``update_history=False``, so consecutive calls are
   independent. This is the server-side equivalent of
   ``wbc_h12/h2_tools/tests/test_models_offline.py`` and carries the same
   thresholds, so a failure here means what a failure there means.

What this cannot check is tracking: holding the robot still while the reference
moves is not a control loop. Use ``Mujoco/sim/h2_edu_sim.py`` for that.

Run from the host once the server is up::

    python tools/smoke_test.py
    python tools/smoke_test.py --ab-order
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h2_policy_if import H2PolicyClient, h2_joints, protocol  # noqa: E402

# Thresholds copied from wbc_h12/h2_tools/tests/test_models_offline.py:12-13.
ACTION_ABS_MAX = 10.0
ACTION_JUMP_MAX = 3.0
FSQ_ABS_MAX = 1.0001  # tokens are FSQ-quantised into [-1, 1]


def sweep(client: H2PolicyClient, motion: str, ref_order: str, steps: int) -> dict:
    """Sweep the clip open-loop with the robot held at the default pose.

    Args:
        client: Connected client.
        motion: Reference motion name.
        ref_order: Joint order the reference command is fed in.
        steps: Number of 50 Hz samples to take.

    Returns:
        Statistics over the sampled actions and tokens, plus per-request latency.
    """
    info = client.reset(motion=motion, ref_order=ref_order)
    q = h2_joints.DEFAULT_ANGLES_MUJOCO.copy()
    qd = np.zeros(protocol.NUM_DOF)
    base_quat = np.array([1.0, 0.0, 0.0, 0.0])  # upright
    ang_vel = np.zeros(3)

    actions, tokens, latencies = [], [], []
    for step in range(steps):
        t0 = time.perf_counter()
        out = client.act(
            t=step * client.control_dt,
            base_quat=base_quat,
            joint_pos=q,
            joint_vel=qd,
            base_ang_vel=ang_vel,
            update_history=False,
        )
        latencies.append((time.perf_counter() - t0) * 1e3)
        for name in ("joint_pos_target", "action", "tokens"):
            if not np.all(np.isfinite(out[name])):
                raise AssertionError(f"step {step}: non-finite {name}")
        actions.append(out["action"].copy())
        tokens.append(out["tokens"].copy())

    a = np.asarray(actions)
    jumps = np.abs(np.diff(a, axis=0))
    return {
        "duration": info["duration"],
        "action_abs_max": float(np.abs(a).max()),
        "jump_max": float(jumps.max()) if jumps.size else 0.0,
        "jump_p99": float(np.percentile(jumps, 99)) if jumps.size else 0.0,
        "token_abs_max": float(np.abs(np.asarray(tokens)).max()),
        "rtt_median_ms": float(np.median(latencies)),
        "rtt_p99_ms": float(np.percentile(latencies, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--endpoint", default=protocol.DEFAULT_ENDPOINT)
    parser.add_argument("--motion", default="idle_loop_003__A041")
    parser.add_argument("--steps", type=int, default=150, help="samples at 50 Hz")
    parser.add_argument(
        "--ab-order",
        action="store_true",
        help="sweep both candidate reference joint orders side by side",
    )
    args = parser.parse_args()

    client = H2PolicyClient(endpoint=args.endpoint, session="smoke")
    print("-- ping --")
    for key in ("encoder", "decoder", "wbc_root", "num_dof", "control_dt", "providers"):
        print(f"  {key:12s} {client.info.get(key)}")

    print("-- motions --")
    motions = client.motions()
    for name in motions:
        print(f"  {name}")
    if args.motion not in motions:
        raise SystemExit(f"motion {args.motion!r} not served; available: {motions}")

    print("-- raw passthrough --")
    tokens = client.encode(np.zeros(protocol.TOKENIZER_DIM))
    action = client.decode(np.zeros(protocol.DECODER_INPUT_DIM))
    assert tokens.shape == (protocol.TOKEN_DIM,), tokens.shape
    assert action.shape == (protocol.NUM_DOF,), action.shape
    print(f"  encode: tokens {tokens.shape} |max| {np.abs(tokens).max():.3f}")
    print(f"  decode: action {action.shape} |max| {np.abs(action).max():.3f}")

    print("-- error handling --")
    try:
        client.act(
            t=0.0,
            base_quat=np.zeros(4),  # not a unit quaternion
            joint_pos=np.zeros(protocol.NUM_DOF),
            joint_vel=np.zeros(protocol.NUM_DOF),
            base_ang_vel=np.zeros(3),
        )
    except Exception as exc:
        print(f"  rejected as expected: {type(exc).__name__}")
    else:
        raise SystemExit("server accepted a zero quaternion; validation is not working")

    orders = ("isaaclab", "mujoco") if args.ab_order else ("isaaclab",)
    print(f"-- open-loop sweep: {args.motion}, {args.steps} samples @ 20 ms --")
    failures = []
    for order in orders:
        r = sweep(client, args.motion, order, args.steps)
        ok = (
            r["action_abs_max"] < ACTION_ABS_MAX
            and r["jump_max"] < ACTION_JUMP_MAX
            and r["token_abs_max"] <= FSQ_ABS_MAX
        )
        print(
            f"  ref_order={order:9s} |action|max {r['action_abs_max']:6.3f}  "
            f"jump max {r['jump_max']:6.3f} p99 {r['jump_p99']:6.3f}  "
            f"|token|max {r['token_abs_max']:.3f}  "
            f"rtt {r['rtt_median_ms']:.2f}/{r['rtt_p99_ms']:.2f} ms  "
            f"{'PASS' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append(order)

    print()
    if failures:
        raise SystemExit(f"FAIL: thresholds exceeded for ref_order={failures}")
    print("RESULT: PASS")
    if args.ab_order:
        print(
            "NOTE  A held-still sweep cannot tell the two reference orders apart.\n"
            "      Settle it in closed loop: Mujoco/sim/h2_edu_sim.py --ref-order {isaaclab,mujoco}"
        )


if __name__ == "__main__":
    main()
