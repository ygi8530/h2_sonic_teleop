#!/usr/bin/env python3
"""Display PICO VR 3-point targets as markers in the real H2 MuJoCo scene.

Minimal bridge for the H2 teleoperation stage 1: subscribes to pelvis-relative
targets published by GR00T-WholeBodyControl's ``h2_vr3pt_mujoco_viewer.py
--pub-port`` and draws three markers (left wrist / right wrist / torso) on the
H2 of this workspace's ``assets/h2_edu_scene.xml``. The robot is held static at
its spawn pose (mj_forward only); nothing is simulated or controlled, and no
existing simulator file is touched.

Run inside the sim container (meshes at /opt/wbc_h12):
    ./run.sh shell
    python tools/h2_vr3pt_marker_bridge.py

Or directly on the host with any python that has mujoco+zmq (e.g. the GR00T
teleop venv) — the mesh directory is remapped to ~/workspace/wbc_h12
automatically when /opt/wbc_h12 does not exist:
    ~/workspace/GR00T-WholeBodyControl/.venv_teleop/bin/python \
        tools/h2_vr3pt_marker_bridge.py

Headless self-test (no display; saves PNG frames + received-count check):
    ... tools/h2_vr3pt_marker_bridge.py --headless-frames 60
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mujoco
import numpy as np
import zmq
from scipy.spatial.transform import Rotation as sRot

_HERE = Path(__file__).resolve().parent.parent
_SCENE = _HERE / "assets" / "h2_edu_scene.xml"
_CONTAINER_WBC = "/opt/wbc_h12"
_HOST_WBC = str(Path.home() / "workspace" / "wbc_h12")

_MARKER_RGBA = {
    "left": np.array([0.2, 0.75, 0.75, 0.9]),
    "right": np.array([0.95, 0.65, 0.2, 0.9]),
    "torso": np.array([0.95, 0.95, 0.95, 0.9]),
}
_AXIS_RGBA = [
    np.array([1.0, 0.2, 0.2, 0.9]),
    np.array([0.2, 1.0, 0.2, 0.9]),
    np.array([0.3, 0.4, 1.0, 0.9]),
]
_AXIS_LEN = 0.09
_BALL_R = 0.02


def load_scene() -> mujoco.MjModel:
    """Load the H2 scene; remap the container meshdir to the host checkout if needed."""
    xml = _SCENE.read_text()
    if _CONTAINER_WBC in xml and not os.path.isdir(_CONTAINER_WBC):
        if not os.path.isdir(_HOST_WBC):
            raise FileNotFoundError(
                f"Neither {_CONTAINER_WBC} nor {_HOST_WBC} exists - run inside the "
                "sim container (./run.sh shell) or on a host with the wbc_h12 checkout."
            )
        xml = xml.replace(_CONTAINER_WBC, _HOST_WBC)
        print(f"[Bridge] meshdir remapped: {_CONTAINER_WBC} -> {_HOST_WBC}")
    return mujoco.MjModel.from_xml_string(xml)


def add_marker(scn, pos_w, quat_wxyz, rgba):
    rot = sRot.from_quat(quat_wxyz, scalar_first=True).as_matrix()
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([_BALL_R, 0, 0]), np.asarray(pos_w),
        np.eye(3).flatten(), rgba,
    )
    scn.ngeom += 1
    for ax in range(3):
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
            np.eye(3).flatten(), _AXIS_RGBA[ax],
        )
        mujoco.mjv_connector(
            g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.004, np.asarray(pos_w),
            np.asarray(pos_w) + _AXIS_LEN * rot[:, ax],
        )
        scn.ngeom += 1


def targets_to_world(vr_3pt, pelvis_pos, pelvis_quat_wxyz):
    rot_p = sRot.from_quat(pelvis_quat_wxyz, scalar_first=True)
    out = []
    for i in range(3):
        pos_w = pelvis_pos + rot_p.apply(vr_3pt[i, :3])
        quat_w = (rot_p * sRot.from_quat(vr_3pt[i, 3:], scalar_first=True)).as_quat(
            scalar_first=True
        )
        out.append((pos_w, quat_w))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1", help="Publisher host (GR00T teleop PC)")
    ap.add_argument("--port", type=int, default=5559, help="Publisher port (--pub-port value)")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--headless-frames", type=int, default=0,
                    help="Render N offscreen frames then exit (self-test, no display)")
    ap.add_argument("--out-dir", default="/tmp/h2_vr3pt_bridge")
    ap.add_argument("--pelvis-z", type=float, default=1.04,
                    help="Spawn pelvis height [m] (h2_joints.DEFAULT_BASE_HEIGHT)")
    args = ap.parse_args()

    model = load_scene()
    data = mujoco.MjData(model)
    # The scene's pelvis body carries no XML offset; the simulator sets the spawn
    # height at reset (sim/h2_edu_sim.py:272, h2_joints.DEFAULT_BASE_HEIGHT). Mirror it.
    data.qpos[0:3] = (0.0, 0.0, args.pelvis_z)
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    pelvis_pos = data.qpos[0:3].copy()
    pelvis_quat = data.qpos[3:7].copy()
    joint_qposadr = {}  # filled lazily from q_upper payloads (stage 2-B IK preview)

    def apply_q_upper(payload):
        qu = payload.get("q_upper")
        if not qu:
            return
        for n, val in qu.items():
            adr = joint_qposadr.get(n)
            if adr is None:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                if jid < 0:
                    continue
                adr = model.jnt_qposadr[jid]
                joint_qposadr[n] = adr
            data.qpos[adr] = val
        mujoco.mj_forward(model, data)
    print(f"[Bridge] H2 pelvis: pos={np.round(pelvis_pos, 3)} quat(wxyz)={np.round(pelvis_quat, 3)}")

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.CONFLATE, 1)  # latest frame only
    sub.connect(f"tcp://{args.host}:{args.port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    print(f"[Bridge] Subscribed to tcp://{args.host}:{args.port}")

    def poll_latest():
        try:
            msg = sub.recv(flags=zmq.NOBLOCK)  # single-part (CONFLATE-compatible)
        except zmq.Again:
            return None
        payload = json.loads(msg.decode())
        return np.asarray(payload["vr_3pt"], dtype=np.float64), payload

    period = 1.0 / args.hz

    if args.headless_frames > 0:
        os.makedirs(args.out_dir, exist_ok=True)
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = (pelvis_pos[0] + 0.2, pelvis_pos[1], 0.9)
        cam.distance, cam.elevation, cam.azimuth = 2.4, -15.0, 160.0
        received = 0
        last = None
        for k in range(args.headless_frames):
            res = poll_latest()
            if res is not None:
                last = res
                received += 1
            if last is not None:
                vr_3pt, payload = last
                apply_q_upper(payload)
                renderer.update_scene(data, camera=cam)
                for (pos_w, quat_w), key in zip(
                    targets_to_world(vr_3pt, pelvis_pos, pelvis_quat), ["left", "right", "torso"]
                ):
                    add_marker(renderer.scene, pos_w, quat_w, _MARKER_RGBA[key])
                if k % 20 == 0 or k == args.headless_frames - 1:
                    frame = renderer.render()
                    try:
                        from PIL import Image

                        Image.fromarray(frame).save(
                            os.path.join(args.out_dir, f"bridge_{k:04d}.png")
                        )
                    except ImportError:
                        np.save(os.path.join(args.out_dir, f"bridge_{k:04d}.npy"), frame)
            time.sleep(period)
        print(f"[Bridge headless] frames={args.headless_frames} messages received={received}")
        if last is not None:
            print(f"[Bridge headless] last rel targets [m]:\n{np.round(last[0][:, :3], 3)}")
        return

    from mujoco import viewer as mj_viewer

    with mj_viewer.launch_passive(model, data) as viewer:
        print("[Bridge] Viewer running - markers follow the published targets.")
        last = None
        last_print = 0.0
        while viewer.is_running():
            t0 = time.time()
            res = poll_latest()
            if res is not None:
                last = res
            if last is not None:
                vr_3pt, payload = last
                apply_q_upper(payload)
                viewer.user_scn.ngeom = 0
                for (pos_w, quat_w), key in zip(
                    targets_to_world(vr_3pt, pelvis_pos, pelvis_quat), ["left", "right", "torso"]
                ):
                    add_marker(viewer.user_scn, pos_w, quat_w, _MARKER_RGBA[key])
                now = time.time()
                if now - last_print > 2.0:
                    calib = "calibrated" if payload.get("calibrated") else "UNCALIBRATED"
                    qmsg = "q_upper OK (robot follows)" if payload.get("q_upper") else                         "no q_upper (publisher not calibrated or started without --ik)"
                    print(f"[Bridge] {calib} | {qmsg} | rel targets [m]: "
                          f"{np.round(vr_3pt[:, :3], 3).tolist()}")
                    last_print = now
            viewer.sync()
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)


if __name__ == "__main__":
    main()
