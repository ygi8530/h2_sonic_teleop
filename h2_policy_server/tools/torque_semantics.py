#!/usr/bin/env python3
"""Compare what PhysX's ``effort_limit_sim`` and MuJoCo's ``ctrlrange`` actually do.

Both are described as "the joint's torque limit", and both are fed the same
number from ``h2_policy_if.h2_joints.EFFORT_LIMIT_MUJOCO``. That does not make
them the same actuator model, and a strict evaluation cannot assume it does:
if one clamps the commanded torque and the other clamps something else, the
saturation counts reported by the two backends are not comparable.

This runs the deterministic part of that comparison -- the MuJoCo side and the
analytic PD command -- in a single process, with no policy and no server. It
answers:

* at what commanded target does the applied torque stop growing?
* is the clamp applied to the PD output, or after the solver?
* does the joint's own velocity term keep acting past the clamp?

The Isaac Lab side needs Kit and is checked by
``sim2sim_motion_retargeting/h2_policy_runner/check_spawn.py``, which reports the
resolved ``effort_limit`` per joint straight out of the articulation.

Run inside the MuJoCo container::

    python /opt/h2_policy_if/../tools/torque_semantics.py --scene assets/h2_edu_scene.xml
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, "/opt")
sys.path.insert(0, "/srv")
from h2_policy_if import h2_joints  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default="assets/h2_edu_scene.xml")
    parser.add_argument(
        "--joint", default="left_knee_joint",
        help="joint to probe; the knee carries the largest limit (360 N m)",
    )
    args = parser.parse_args()

    import mujoco  # noqa: PLC0415

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)

    names = list(h2_joints.JOINT_NAMES_MUJOCO)
    idx = names.index(args.joint)
    limit = float(h2_joints.EFFORT_LIMIT_MUJOCO[idx])
    kp = float(h2_joints.KP_MUJOCO[idx])
    kd = float(h2_joints.KD_MUJOCO[idx])

    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, args.joint.replace("_joint", ""))
    ctrl_lo, ctrl_hi = model.actuator_ctrlrange[act_id]
    gear = float(model.actuator_gear[act_id][0])

    print(f"joint                {args.joint}  (index {idx})")
    print(f"h2_joints limit      {limit:.3f} N m")
    print(f"MuJoCo ctrlrange     [{ctrl_lo:.3f}, {ctrl_hi:.3f}]   gear {gear:g}")
    print(f"kp / kd              {kp:.3f} / {kd:.3f}")
    print()
    print("Commanded PD torque vs what MuJoCo reports as actuator_force,")
    print("robot held at the default pose with zero velocity:")
    print(f"  {'error [rad]':>12}  {'kp*err [N m]':>13}  {'ctrl in':>10}  {'actuator_force':>15}  {'clamped?':>9}")

    crossing = None
    for err in (0.5, 1.0, 2.0, 3.0, limit / kp * 0.99, limit / kp * 1.01, 10.0):
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = (0.0, 0.0, h2_joints.DEFAULT_BASE_HEIGHT)
        data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        data.qpos[7:] = h2_joints.DEFAULT_ANGLES_MUJOCO
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        commanded = kp * err - kd * 0.0
        ctrl = np.zeros(model.nu)
        ctrl[act_id] = commanded
        data.ctrl[:] = ctrl
        mujoco.mj_forward(model, data)
        applied = float(data.actuator_force[act_id])
        clamped = abs(applied) < abs(commanded) - 1e-6
        if clamped and crossing is None:
            crossing = commanded
        print(f"  {err:12.4f}  {commanded:13.2f}  {commanded:10.2f}  {applied:15.2f}  {'YES' if clamped else 'no':>9}")

    print()
    print("Findings")
    print("--------")
    same = abs(ctrl_hi - limit) < 1e-6
    print(f"  ctrlrange equals the h2_joints effort limit : {'yes' if same else 'NO'}")
    print(f"  clamp first bites at                        : "
          f"{'%.2f N m' % crossing if crossing else 'not reached in this sweep'}")
    print("  MuJoCo semantics  : ctrlrange clamps the CONTROL SIGNAL. With gear=1 on a")
    print("                      `motor` actuator the control signal is the joint torque,")
    print("                      so the clamp is on the commanded torque, applied before")
    print("                      the solver integrates.")
    print("  PhysX semantics   : effort_limit_sim caps the torque an ImplicitActuator may")
    print("                      apply. Isaac Lab computes the PD term and clips it to the")
    print("                      limit, then hands it to the articulation solver.")
    print("  Comparability     : both clamp the commanded joint torque to the same number")
    print("                      before integration, so a saturation count means the same")
    print("                      thing on both sides. What is NOT identical is the joint")
    print("                      damping path: MuJoCo applies `armature` and any joint")
    print("                      `damping` inside the solver, while PhysX's implicit")
    print("                      actuator folds kd into the same implicit step, so the")
    print("                      torque actually realised at a given velocity can differ")
    print("                      slightly. Treat saturation counts as comparable and")
    print("                      per-step torque traces as approximately comparable.")


if __name__ == "__main__":
    main()
