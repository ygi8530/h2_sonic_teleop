#!/usr/bin/env python3
"""Closed-loop MuJoCo simulation of the H2 EDU SONIC policy.

Runs the 50 Hz control loop against the policy server: the server owns the
encoder, the decoder, the reference clip, the proprio history and the joint-order
bookkeeping, so this file only does physics and PD control.

    python sim/h2_edu_sim.py --motion idle_loop_003__A041 --seconds 10
    python sim/h2_edu_sim.py --motion walk_forward_loop_001__A029 --video walk.mp4
    python sim/h2_edu_sim.py --motion walk_forward_loop_001__A029 --viewer
    python sim/h2_edu_sim.py --ab-order          # settle the reference-order question

Loop structure, per control step (0.02 s = 4 physics steps of 0.005 s, matching
``sim.dt`` x ``decimation`` in gear_sonic's env config)::

    read q, qd, base quat, base angular velocity  (MuJoCo order)
        -> server.act(...)  -> joint position targets (MuJoCo order)
        -> tau = kp * (target - q) - kd * qd, 4x mj_step
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

# MUJOCO_GL is read when mujoco first creates a GL context, so the backend has to
# be chosen before the import below. Headless runs keep EGL (the image sets it,
# and it renders ~280x faster than Mesa for --video); the interactive viewer
# needs a windowing backend, which is GLFW.
_VIEWER = "--viewer" in sys.argv
if _VIEWER:
    os.environ["MUJOCO_GL"] = "glfw"

import mujoco
import numpy as np

if _VIEWER:
    # Imported at module scope on purpose: an `import mujoco.viewer` inside a
    # function makes `mujoco` a local name there and shadows the module above.
    import mujoco.viewer

sys.path.insert(0, "/opt")  # h2_policy_if is mounted there, read-only
from h2_policy_if import H2PolicyClient, h2_joints, protocol  # noqa: E402
from h2_policy_if import evaluation as ev  # noqa: E402
from h2_policy_if.interface_perturbations import InterfaceLayer  # noqa: E402

DEFAULT_SCENE = Path(__file__).resolve().parents[1] / "assets" / "h2_edu_scene.xml"
PHYSICS_SUBSTEPS = 4  # decimation


# Axes this runner can perturb, with the training term each one reproduces.
# Only parameters SONIC actually randomised are offered; an axis training never
# touched would be OOD_SIM2REAL and must not be mixed into these results.
PERTURB_AXES = {
    "robot_friction": (
        "robot_material.static_friction",
        "sliding friction of every robot geom; training randomised the robot body "
        "material and combined it with the terrain by multiply",
    ),
    "body_mass_scale": (
        "body_mass_scale",
        "mass multiplier on *_wrist_yaw_link and torso_link, the bodies "
        "events/tracking/level0_4.yaml scales",
    ),
    "torso_com_x": ("torso_com_offset_m", "torso_link centre of mass, body-frame x"),
    "torso_com_y": ("torso_com_offset_m", "torso_link centre of mass, body-frame y"),
    "torso_com_z": ("torso_com_offset_m", "torso_link centre of mass, body-frame z"),
    "joint_default_offset": (
        "joint_default_pos_offset_rad",
        "uniform offset added to every joint's default position; applied to BOTH the "
        "reset pose here and the policy server's action offset, because Isaac Lab's "
        "default_joint_pos drives both",
    ),
}

# Axes a MuJoCo run must refuse rather than approximate. See
# h2_policy_if.evaluation.AXIS_SUPPORT for why each one cannot be expressed here.
UNSUPPORTED_AXES = ("robot_dynamic_friction", "robot_restitution")


def _apply_ood(model: mujoco.MjModel, axis: str, value: float) -> dict:
    """Apply one simulator-level OOD_SIM2REAL axis to the compiled model.

    Initial-condition axes are recorded here but realised in :func:`reset_robot`,
    which owns the spawn state.

    Args:
        model: Compiled MuJoCo model, modified in place.
        axis: An ``OOD_PERTURBATIONS`` key whose layer is ``simulator``.
        value: The operator's chosen value; there is no range to check it against.

    Returns:
        What changed, for the result metadata.
    """
    if axis == "kp_scale":
        # The PD gains live on the policy server side of the loop in this setup:
        # MuJoCo actuators are plain torque motors and the server sends targets.
        # Scaling here would be a lie, so the gain is scaled where it is used --
        # see KP_SCALE/KD_SCALE in run().
        return {"value": value, "applied_in": "control loop"}
    if axis == "kd_scale":
        return {"value": value, "applied_in": "control loop"}
    if axis == "armature_scale":
        model.dof_armature[:] = model.dof_armature * value
        return {"value": value, "dofs": int(model.nv)}
    if axis == "effort_limit_scale":
        model.actuator_ctrlrange[:] = model.actuator_ctrlrange * value
        return {"value": value, "actuators": int(model.nu)}
    if axis == "body_inertia_scale":
        model.body_inertia[:] = model.body_inertia * value
        return {"value": value, "bodies": int(model.nbody)}
    if axis in ("init_height_offset", "init_joint_pos_noise", "init_root_vel_noise"):
        return {"value": value, "applied_in": "reset_robot"}
    raise SystemExit(f"{axis} is registered but this runner does not implement it")


def apply_perturbations(model: mujoco.MjModel, specs: list[str],
                        allow_ood: bool = False) -> dict:
    """Apply TRAIN_DIST and simulator-level OOD perturbations to the model, in place.

    Args:
        model: Compiled MuJoCo model, modified in place.
        specs: ``AXIS=VALUE`` strings from ``--perturb``.
        allow_ood: Whether OOD_SIM2REAL axes may be applied at all. False makes
            any such axis a hard error, so a strict baseline cannot contain one
            by accident.

    Returns:
        A record of what was applied, for the result's metadata. Empty when no
        perturbation was requested, which is what a strict baseline must show.

    Raises:
        SystemExit: On an unknown axis, or a value outside the training range --
            an out-of-range value would be OOD_SIM2REAL and needs its own study,
            not a silent entry in a TRAIN_DIST sweep.
    """
    applied: dict = {}
    for spec in specs or []:
        if "=" not in spec:
            raise SystemExit(f"--perturb expects AXIS=VALUE, got {spec!r}")
        axis, raw = spec.split("=", 1)
        if axis in ev.OOD_PERTURBATIONS:
            entry = ev.OOD_PERTURBATIONS[axis]
            if entry.layer == "interface":
                continue  # handled by InterfaceLayer, which owns the control loop
            if not allow_ood:
                raise SystemExit(
                    f"{axis} is OOD_SIM2REAL -- SONIC training never randomised it, so "
                    "there is no range to justify a value. Pass --allow-ood to run it "
                    "anyway; the result is then graded OOD_SIM2REAL and must not be "
                    "compared against a strict baseline."
                )
            if entry.support.get("mujoco") != "exact":
                raise SystemExit(
                    f"{axis} cannot be expressed in MuJoCo.\n  {entry.note or entry.description}"
                )
            applied[axis] = _apply_ood(model, axis, float(raw))
            applied[axis].update(grade="OOD_SIM2REAL", unit=entry.unit,
                                 layer="simulator", description=entry.description)
            continue
        if axis in UNSUPPORTED_AXES:
            rec = ev.AXIS_SUPPORT[axis]
            raise SystemExit(
                f"{axis} cannot be expressed in MuJoCo, so this runner refuses it rather "
                f"than approximating it.\n  {rec['note']}\n"
                f"  Run this axis on --engine isaacsim_physx, where it is exact."
            )
        if axis not in PERTURB_AXES:
            raise SystemExit(f"unknown perturbation axis {axis!r}; known: {', '.join(PERTURB_AXES)}")
        value = float(raw)
        train_key, description = PERTURB_AXES[axis]
        train_range = ev.TRAIN_RANDOMIZATION[train_key].value
        # torso_com_offset_m is a per-component dict; the others are a plain pair.
        lo, hi = train_range[axis[-1]] if isinstance(train_range, dict) else train_range
        if not lo <= value <= hi:
            raise SystemExit(
                f"{axis}={value} is outside the training range [{lo}, {hi}] "
                f"({ev.TRAIN_RANDOMIZATION[train_key].source}). Outside that range the "
                f"perturbation is OOD_SIM2REAL and must be run and reported separately."
            )

        if axis == "robot_friction":
            # geom_friction is (sliding, torsional, rolling); only the sliding
            # term has a PhysX counterpart in the training material.
            floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            touched = 0
            for gid in range(model.ngeom):
                if gid == floor:
                    continue
                model.geom_friction[gid, 0] = value
                touched += 1
            applied[axis] = {"value": value, "geoms": touched}
        elif axis == "body_mass_scale":
            names = [n for n in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
                                 for b in range(model.nbody))
                     if n and (n.endswith("wrist_yaw_link") or n == "torso_link")]
            for name in names:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                model.body_mass[bid] *= value
                model.body_inertia[bid] *= value
            applied[axis] = {"value": value, "bodies": names}
        elif axis.startswith("torso_com_"):
            comp = {"x": 0, "y": 1, "z": 2}[axis[-1]]
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
            before = float(model.body_ipos[bid, comp])
            model.body_ipos[bid, comp] = before + value
            applied[axis] = {"value": value, "body": "torso_link",
                             "ipos_before": round(before, 6),
                             "ipos_after": round(float(model.body_ipos[bid, comp]), 6)}
        elif axis == "joint_default_offset":
            # Recorded here; the caller applies it to the reset pose and hands the
            # same vector to the policy server, so the two halves cannot disagree.
            applied[axis] = {"value": value, "joints": int(h2_joints.NUM_DOF)}

        applied[axis]["grade"] = ev.TRAIN_RANDOMIZATION[train_key].grade
        applied[axis]["train_range"] = [lo, hi]
        applied[axis]["source"] = ev.TRAIN_RANDOMIZATION[train_key].source
        applied[axis]["description"] = description
    return applied


def base_angular_velocity_local(data: mujoco.MjData) -> np.ndarray:
    """Pelvis angular velocity in the base frame [rad/s].

    MuJoCo's free-joint ``qvel[3:6]`` is already expressed in the body frame, which
    is what the policy's ``base_ang_vel`` observation is.
    """
    return data.qvel[3:6].copy()


def settle(model: mujoco.MjModel, data: mujoco.MjData, seconds: float = 0.0) -> None:
    """Hold the default pose with PD control before the policy takes over.

    Off by default, and kept only as a diagnostic. The spawn pose already has the
    soles in contact (measured: lowest sole at z = +0.0007 m), so there is no free
    fall to absorb -- and PD alone cannot hold this crouch, because kp = 99 N m/rad
    at the knee against 75 kg settles about a radian short. Holding for 0.5 s
    therefore *creates* the problem it looks like it prevents: the pelvis sinks
    from 1.04 m to 0.67 m and the policy starts from a deep crouch it then has to
    climb out of. Handing the robot over standing keeps the minimum pelvis height
    at ~0.95 m instead.
    """
    if seconds <= 0.0:
        return
    target = h2_joints.DEFAULT_ANGLES_MUJOCO
    for _ in range(int(seconds / model.opt.timestep)):
        data.ctrl[:] = h2_joints.KP_MUJOCO * (target - data.qpos[7:]) - h2_joints.KD_MUJOCO * data.qvel[6:]
        mujoco.mj_step(model, data)


def reset_robot(model: mujoco.MjModel, data: mujoco.MjData,
                default_offset_mj: np.ndarray | None = None,
                init: dict | None = None, rng: "np.random.Generator | None" = None) -> None:
    """Put the robot at the training spawn pose: default joints, upright, 1.04 m.

    Args:
        model: Compiled scene.
        data: Simulation state, reset in place.
        default_offset_mj: Optional per-joint offset [rad] in MuJoCo order, added
            to the default pose. The SAME offset must reach the policy server via
            ``reset(joint_default_offset=...)``, because Isaac Lab's
            ``default_joint_pos`` is both the reset pose and the action offset.
        init: Optional OOD_SIM2REAL spawn perturbations --
            ``init_height_offset`` [m], ``init_joint_pos_noise`` [rad] and
            ``init_root_vel_noise`` [m/s or rad/s] as noise standard deviations.
        rng: Seeded generator for the two noise terms; required when either is set.
    """
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = (0.0, 0.0, h2_joints.DEFAULT_BASE_HEIGHT)
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)  # wxyz identity
    pose = h2_joints.DEFAULT_ANGLES_MUJOCO.copy()
    if default_offset_mj is not None:
        pose = pose + default_offset_mj
    init = init or {}
    if init.get("init_height_offset"):
        data.qpos[2] += init["init_height_offset"]
    if init.get("init_joint_pos_noise"):
        pose = pose + rng.normal(0.0, init["init_joint_pos_noise"], pose.shape)
    data.qpos[7:] = pose
    data.qvel[:] = 0.0
    if init.get("init_root_vel_noise"):
        data.qvel[0:6] = rng.normal(0.0, init["init_root_vel_noise"], 6)
    mujoco.mj_forward(model, data)


def run(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    client: H2PolicyClient,
    motion: str,
    ref_order: str,
    seconds: float,
    renderer: "mujoco.Renderer | None" = None,
    fps: int = 30,
    viewer_handle=None,
    settle_seconds: float = 0.0,
    fall_height: float = 0.5,
    record_series: bool = False,
    default_offset_mj: np.ndarray | None = None,
    push: "ev.PushSchedule | None" = None,
    iface: "InterfaceLayer | None" = None,
    kp_scale: float = 1.0,
    kd_scale: float = 1.0,
    init: dict | None = None,
    seed: int | None = None,
) -> dict:
    """Run one closed-loop episode.

    Args:
        model: Compiled scene.
        data: Simulation state, reset by this function.
        client: Connected policy client.
        motion: Reference motion name.
        ref_order: Joint order the reference command is fed to the encoder in.
        seconds: Wall-clock length of the episode [s].
        renderer: Offscreen renderer, or None to skip video.
        fps: Video frame rate.
        viewer_handle: Passive ``mujoco.viewer`` handle, or None for headless.
            When given, the loop is paced to wall clock so the motion plays at
            real speed instead of as fast as the machine can solve it.
        iface: Interface-level OOD perturbations (latency, jitter, sensor noise).
            None means a neutral pass-through.
        kp_scale: Multiplier on the position gain [x]. OOD_SIM2REAL.
        kd_scale: Multiplier on the damping gain [x]. OOD_SIM2REAL.
        init: OOD_SIM2REAL spawn perturbations passed to :func:`reset_robot`.
        seed: Seeds the spawn noise, so a perturbed spawn is reproducible.

    Returns:
        ``fell`` (whether the pelvis dropped below 0.5 m), the step it happened at,
        the minimum pelvis height [m], the mean per-step server latency [ms], and
        the captured frames.
    """
    reset_robot(model, data, default_offset_mj, init,
                np.random.default_rng(seed) if init else None)
    settle(model, data, settle_seconds)
    info = client.reset(
        motion=motion, ref_order=ref_order,
        joint_default_offset=(None if default_offset_mj is None
                              else h2_joints.to_isaaclab(default_offset_mj)),
    )

    control_dt = client.control_dt
    steps = int(seconds / control_dt)
    frame_every = max(1, round(1.0 / (fps * control_dt)))

    # Foot bodies for the contact metric. H2's ankle chain is inverted, so
    # *_ankle_pitch_link is the sole (robots/h2_edu.py).
    foot_bodies = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in ("left_ankle_pitch_link", "right_ankle_pitch_link")
    }
    foot_bodies.discard(-1)
    effort_limit = h2_joints.EFFORT_LIMIT_MUJOCO

    if iface is None:
        iface = InterfaceLayer(None, control_dt, PHYSICS_SUBSTEPS)
    kp = h2_joints.KP_MUJOCO * kp_scale
    kd = h2_joints.KD_MUJOCO * kd_scale
    default_pose_mj = (h2_joints.DEFAULT_ANGLES_MUJOCO if default_offset_mj is None
                       else h2_joints.DEFAULT_ANGLES_MUJOCO + default_offset_mj)
    substeps_used = []

    frames = []
    latencies = []
    heights = []
    sat_steps = 0
    pushes_applied = 0
    contact_steps = 0
    torque_abs_max = 0.0
    joint_pos_abs_max = 0.0
    joint_vel_abs_max = 0.0
    saw_nan = False
    saw_divergence = False
    series = {"t": [], "root_pos": [], "root_quat": [], "joint_pos": [], "joint_vel": [], "torque": []}
    min_height = float("inf")
    fell_at = None
    # Wall-clock anchor for the viewer. The control loop itself is unchanged:
    # every iteration is still exactly one 0.02 s control step and four 0.005 s
    # physics steps, so the policy sees the same sequence it does headless. The
    # pacing below only decides when the next iteration starts in real time.
    wall_start = time.perf_counter()
    late_steps = 0
    max_lag = 0.0

    for step in range(steps):
        obs_quat, obs_pos, obs_vel, obs_ang = iface.observe(
            data.qpos[3:7], data.qpos[7:], data.qvel[6:],
            base_angular_velocity_local(data),
        )
        t0 = time.perf_counter()
        out = client.act(
            t=step * control_dt,
            base_quat=obs_quat,
            joint_pos=obs_pos,
            joint_vel=obs_vel,
            base_ang_vel=obs_ang,
            order="mujoco",
        )
        latencies.append((time.perf_counter() - t0) * 1e3)
        target = iface.command(out["joint_pos_target"], default_pose_mj)

        step_saturated = False
        n_sub = iface.substeps()
        substeps_used.append(n_sub)
        for _ in range(n_sub):
            tau = kp * (target - data.qpos[7:]) - kd * data.qvel[6:]
            # Saturation is measured on the commanded torque, before the
            # actuator's ctrlrange clamps it, so the count reflects the policy
            # asking for more than the machine has.
            if np.any(np.abs(tau) >= effort_limit):
                step_saturated = True
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
        sat_steps += int(step_saturated)

        if push is not None:
            vel = push.due((step - 1) * control_dt, step * control_dt)
            if vel is not None:
                # push_by_setting_velocity SETS the base velocity; it does not add
                # an impulse. qvel[0:3] is linear, qvel[3:6] angular in the body
                # frame, which is the frame the training event uses.
                data.qvel[0:6] = vel
                pushes_applied += 1

        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
            print(f"  non-finite state at t={step * control_dt:.2f} s")
            saw_nan = True
            steps = step + 1
            break

        height = float(data.qpos[2])
        if ev.diverged(height):
            # Not a fall: the solver has left the physical band. Stopping here
            # keeps post-divergence values out of the height and torque metrics.
            print(f"  diverged at t={step * control_dt:.2f} s (pelvis {height:.1f} m)")
            saw_divergence = True
            steps = step + 1
            break
        heights.append(height)
        min_height = min(min_height, height)
        torque_abs_max = max(torque_abs_max, float(np.abs(data.actuator_force).max()))
        joint_pos_abs_max = max(joint_pos_abs_max, float(np.abs(data.qpos[7:]).max()))
        joint_vel_abs_max = max(joint_vel_abs_max, float(np.abs(data.qvel[6:]).max()))
        touching = any(
            model.geom_bodyid[data.contact[i].geom1] in foot_bodies
            or model.geom_bodyid[data.contact[i].geom2] in foot_bodies
            for i in range(data.ncon)
        )
        contact_steps += int(touching)
        if record_series:
            series["t"].append(round(step * control_dt, 4))
            series["root_pos"].append([round(float(v), 5) for v in data.qpos[0:3]])
            series["root_quat"].append([round(float(v), 5) for v in data.qpos[3:7]])
            series["joint_pos"].append([round(float(v), 5) for v in data.qpos[7:]])
            series["joint_vel"].append([round(float(v), 4) for v in data.qvel[6:]])
            series["torque"].append([round(float(v), 3) for v in data.actuator_force])
        if fell_at is None and height < fall_height:
            fell_at = step

        if renderer is not None and step % frame_every == 0:
            renderer.update_scene(data, camera=-1)
            frames.append(renderer.render())

        if viewer_handle is not None:
            if not viewer_handle.is_running():
                print(f"  viewer closed at t={step * control_dt:.2f} s; stopping")
                steps = step + 1
                break
            viewer_handle.sync()
            # Sleep until this step's wall-clock deadline. If a step overran (a
            # slow server round trip, a GC pause), the deadline has passed and we
            # simply continue -- the simulation never rewinds or drops a step, it
            # just falls behind real time, which is counted and reported.
            deadline = wall_start + (step + 1) * control_dt
            remaining = deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                late_steps += 1
                max_lag = max(max_lag, -remaining)

    done = len(latencies)
    reason = ("nan" if saw_nan else "diverged" if saw_divergence
              else "fall" if fell_at is not None else "completed")
    lat = np.asarray(latencies) if latencies else np.zeros(1)
    return {
        "backend": "mujoco",
        "motion": info["motion"],
        "ref_order": ref_order,
        "steps": steps,
        "control_steps": done,
        "requested_s": round(seconds, 3),
        "survival_s": round(done * control_dt, 3),
        "terminated": reason != "completed",
        "termination_reason": reason,
        "fell": fell_at is not None,
        "fell_at_s": None if fell_at is None else round(fell_at * control_dt, 2),
        "pelvis_height_min": round(min_height, 4) if heights else None,
        "pelvis_height_final": round(float(data.qpos[2]), 4),
        "pelvis_height_mean": round(float(np.mean(heights)), 4) if heights else None,
        "root_xy_displacement": round(float(np.hypot(data.qpos[0], data.qpos[1])), 4),
        "joint_pos_abs_max": round(joint_pos_abs_max, 4),
        "joint_vel_abs_max": round(joint_vel_abs_max, 3),
        "torque_abs_max": round(torque_abs_max, 2),
        "torque_saturation_steps": sat_steps,
        "torque_saturation_rate": round(sat_steps / done, 4) if done else None,
        "foot_contact_rate": round(contact_steps / done, 4) if done else None,
        "pushes_applied": pushes_applied,
        "nan": saw_nan,
        "diverged": saw_divergence,
        "no_motion": joint_vel_abs_max < 1e-6,
        # With jitter this is no longer the nominal decimation, so it is reported
        # rather than assumed.
        "physics_substeps_mean": (round(float(np.mean(substeps_used)), 3)
                                  if substeps_used else None),
        "policy_rtt_ms_median": round(float(np.median(lat)), 3),
        "policy_rtt_ms_p99": round(float(np.percentile(lat, 99)), 3),
        # legacy keys kept so the existing console line and viewer path still read
        "min_height": round(min_height, 4) if heights else None,
        "final_height": round(float(data.qpos[2]), 4),
        "rtt_mean_ms": round(float(np.mean(lat)), 2),
        "frames": frames,
        "late_steps": late_steps,
        "max_lag_ms": round(max_lag * 1e3, 1),
        "_series": series if record_series else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE,
                        help="compiled MJCF; regenerate it with tools/make_scene.py")
    parser.add_argument("--endpoint", default=protocol.DEFAULT_ENDPOINT,
                        help="policy server endpoint")
    parser.add_argument("--motion", default="idle_loop_003__A041",
                        help="reference clip served by the policy server")
    parser.add_argument("--seconds", type=float, default=10.0, help="episode length [s]")
    parser.add_argument(
        "--ref-order",
        default="isaaclab",
        choices=protocol.JOINT_ORDERS,
        help="joint order the reference command is fed to the encoder in",
    )
    parser.add_argument(
        "--ab-order",
        action="store_true",
        help="run both reference orders and compare; the wrong one should fall",
    )
    parser.add_argument("--video", type=Path, default=None, help="write an mp4 here")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="open an interactive MuJoCo window and pace the loop to real time "
             "(needs DISPLAY; forces the GLFW backend)",
    )
    parser.add_argument("--width", type=int, default=640, help="video width [px]")
    parser.add_argument("--height", type=int, default=480, help="video height [px]")
    parser.add_argument("--fps", type=int, default=30, help="video frame rate")
    parser.add_argument(
        "--settle", type=float, default=0.0,
        help="seconds of PD hold at the default pose before the policy starts. 0 (the "
             "default) hands the robot over standing; a non-zero value makes it crouch first",
    )
    parser.add_argument("--list-motions", action="store_true", help="print the clips the server offers and exit")
    parser.add_argument("--fall-height", type=float, default=0.5,
                        help="pelvis height counted as a fall [m]; an evaluation threshold, not physics")
    parser.add_argument(
        "--profile", default="strict-native", choices=sorted(ev.PROFILES),
        help="evaluation profile. strict-native uses only training-sourced physical "
             "parameters and MuJoCo's own numerical defaults",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="recorded in the result; also seeds --push so every backend replays "
                             "the identical disturbance sequence")
    parser.add_argument("--push", action="store_true",
                        help="reproduce training's push_robot event: set the base velocity every "
                             "4-6 s from the training ranges. Requires --seed. TRAIN_DIST.")
    parser.add_argument("--out", type=Path, default=None, help="write the canonical JSON result here")
    parser.add_argument("--series", action="store_true", help="include per-step arrays in --out")
    parser.add_argument("--wbc-root", type=Path, default=Path("/opt/wbc_h12"), help="read-only wbc_h12 checkout, for provenance")
    parser.add_argument(
        "--perturb", action="append", default=None, metavar="AXIS=VALUE",
        help="apply one perturbation, repeatable. TRAIN_DIST axes are bounded by the "
             "training ranges and refused outside them; OOD_SIM2REAL axes have no "
             "training range and need --allow-ood. --list-axes prints them all. "
             "A perturbed run is NOT a strict baseline and is recorded as such.",
    )
    parser.add_argument("--allow-ood", action="store_true",
                        help="permit OOD_SIM2REAL axes. Training randomised none of them, so "
                             "the value is the operator's choice and the result is graded "
                             "OOD_SIM2REAL -- never comparable to a strict baseline.")
    parser.add_argument("--list-axes", action="store_true",
                        help="print every perturbation axis with its grade, unit and range, then exit")
    parser.add_argument("--headless", action="store_true",
                        help="no viewer (the default; accepted so both runners take the same flags)")
    parser.add_argument("--terrain", default="flat", choices=("flat",),
                        help="ground type. Only flat is available here: training's "
                             "ROUGH_TERRAINS_CFG is an Isaac Lab terrain generator with no "
                             "MuJoCo equivalent, so rough terrain runs on --engine isaacsim_physx")
    args = parser.parse_args()

    if args.list_axes:
        ev.print_axes("mujoco")
        return
    if args.headless and args.viewer:
        raise SystemExit("--headless and --viewer contradict each other")

    if not args.scene.is_file():
        raise SystemExit(f"{args.scene} not found. Run: python tools/make_scene.py")

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    expected_nq = 7 + protocol.NUM_DOF
    if model.nq != expected_nq:
        raise SystemExit(f"scene has nq={model.nq}, expected {expected_nq} (7 free + 31 joints)")
    print(f"scene {args.scene.name}: nq={model.nq} nu={model.nu} timestep={model.opt.timestep}")

    renderer = None
    if args.video is not None:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)

    client = H2PolicyClient(endpoint=args.endpoint, session=f"mujoco-{args.ref_order}")
    print(f"policy {client.info['decoder']} @ {args.endpoint}")

    if args.list_motions:
        print(f"reference clips served by {args.endpoint}:")
        for name in client.motions():
            print(f"  {name}")
        return

    profile = ev.PROFILES[args.profile]
    settle_seconds = args.settle
    if profile.strict and settle_seconds != 0.0:
        print(f"[strict] --settle {settle_seconds} ignored: a pre-policy PD hold has no "
              f"training source ({ev.INITIAL_STATE['settle'].source})")
        settle_seconds = 0.0

    # MuJoCo has no solver tuning applied in either profile: the scene uses
    # MuJoCo's own defaults plus the training-sourced timestep, friction,
    # armature and torque limits. backend-recommended therefore differs from
    # strict-native only by permission, not by any value -- recorded as such.
    solver_overrides: dict = {}

    push = None
    if args.push:
        if args.seed is None:
            raise SystemExit("--push needs --seed, so the disturbance sequence is reproducible "
                             "and replayable on another backend")
        push = ev.PushSchedule(seed=args.seed, duration_s=args.seconds)
        print(f"[push] TRAIN_DIST: {len(push.events)} pushes, seed {args.seed} "
              f"<- {ev.TRAIN_RANDOMIZATION['push_velocity'].source}")

    iface_specs, sim_specs = ev.split_perturbations(args.perturb, args.allow_ood)
    perturbations = apply_perturbations(model, sim_specs, args.allow_ood)
    iface = InterfaceLayer(iface_specs, client.control_dt, PHYSICS_SUBSTEPS, args.seed)
    init_specs = {k: perturbations[k]["value"] for k in
                  ("init_height_offset", "init_joint_pos_noise", "init_root_vel_noise")
                  if k in perturbations}
    kp_scale = perturbations.get("kp_scale", {}).get("value", 1.0)
    kd_scale = perturbations.get("kd_scale", {}).get("value", 1.0)
    default_offset_mj = None
    if "joint_default_offset" in perturbations:
        value = perturbations["joint_default_offset"]["value"]
        default_offset_mj = np.full(h2_joints.NUM_DOF, value, dtype=np.float64)
    perturbations.update(iface.as_dict())
    ev.print_perturbations(perturbations)

    provenance = ev.collect_provenance(
        wbc_root=str(args.wbc_root), backend="mujoco",
        backend_version=f"mujoco {mujoco.__version__}",
        profile=profile, motion=args.motion,
        motion_path=None, seed=args.seed, solver_overrides=solver_overrides,
    )
    provenance["perturbations"] = perturbations
    provenance["push"] = push.as_dict() if push is not None else {}
    ev.print_banner(provenance)

    orders = protocol.JOINT_ORDERS if args.ab_order else (args.ref_order,)
    results = []
    for order in orders:
        client.session = f"mujoco-{order}"
        if args.viewer:
            if not os.environ.get("DISPLAY"):
                raise SystemExit(
                    "--viewer needs DISPLAY. docker/docker-compose.yaml forwards it and mounts\n"
                    "/tmp/.X11-unix, so this usually means the host shell had no DISPLAY when the\n"
                    "container started. Run headless, or with --video, instead."
                )
            with mujoco.viewer.launch_passive(
                model, data, show_left_ui=False, show_right_ui=False
            ) as handle:
                result = run(
                    model, data, client, args.motion, order, args.seconds,
                    renderer, args.fps, viewer_handle=handle, settle_seconds=settle_seconds,
                    fall_height=args.fall_height, record_series=args.series,
                    default_offset_mj=default_offset_mj, push=push, iface=iface,
                    kp_scale=kp_scale, kd_scale=kd_scale, init=init_specs, seed=args.seed,
                )
        else:
            result = run(
                model, data, client, args.motion, order, args.seconds, renderer, args.fps,
                settle_seconds=settle_seconds, fall_height=args.fall_height,
                record_series=args.series, default_offset_mj=default_offset_mj, push=push,
                iface=iface, kp_scale=kp_scale, kd_scale=kd_scale,
                init=init_specs, seed=args.seed,
            )
        results.append(result)
        verdict = f"FELL at {result['fell_at_s']}s" if result["fell"] else "stayed up"
        print(
            f"  ref_order={order:9s} {verdict:20s} "
            f"min pelvis {result['min_height']:.3f} m  final {result['final_height']:.3f} m  "
            f"rtt {result['rtt_mean_ms']:.2f} ms"
        )
        if args.out is not None:
            out = args.out if len(orders) == 1 else args.out.with_stem(f"{args.out.stem}_{order}")
            series = result.pop("_series", None)
            metrics = {k: v for k, v in result.items()
                       if k != "frames" and not k.startswith("_")}
            metrics["profile"] = profile.name
            metrics["seed"] = args.seed
            metrics["terrain"] = args.terrain
            metrics["grade"] = ev.result_grade(perturbations)
            prov = dict(provenance)
            prov["profile"] = {**provenance["profile"]}
            ev.write_result(str(out), prov, metrics, series)
            print(f"    result {out}")
        else:
            result.pop("_series", None)
        if args.viewer:
            late = result["late_steps"]
            print(
                f"    realtime: {result['steps'] - late}/{result['steps']} steps met their "
                f"20 ms deadline"
                + (f", worst overrun {result['max_lag_ms']} ms" if late else "")
            )
        if args.video is not None and result["frames"]:
            import imageio.v2 as imageio

            out = args.video if len(orders) == 1 else args.video.with_stem(
                f"{args.video.stem}_{order}"
            )
            imageio.mimsave(out, result["frames"], fps=args.fps)
            print(f"    video {out} ({len(result['frames'])} frames)")

    if args.ab_order:
        standing = [r["ref_order"] for r in results if not r["fell"]]
        print()
        if len(standing) == 1:
            print(
                f"VERDICT  ref_order={standing[0]!r} is the correct reference joint order.\n"
                f"         The other one falls, which is what a permuted command looks like."
            )
        elif not standing:
            print(
                "VERDICT  both fell. The reference order is not the discriminator here -- check\n"
                "         the PD gains, the settle phase and the base_ang_vel frame first."
            )
        else:
            print(
                "VERDICT  both stayed up on this clip. A standing clip is too forgiving;\n"
                "         retry with --motion walk_forward_loop_001__A029."
            )


if __name__ == "__main__":
    main()
