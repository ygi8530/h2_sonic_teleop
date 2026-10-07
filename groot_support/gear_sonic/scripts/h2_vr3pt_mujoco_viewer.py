"""PICO VR 3-point targets rendered as live markers on the H2 MuJoCo model.

Stage-1 teleop milestone: NO IK, NO control, NO policy. The H2 stays at its
spawn pose; three target markers (left wrist / right wrist / torso) move with
the operator, expressed in the H2 pelvis frame using the same calibration
pipeline as the G1 manager path — only the FK reference is swapped to H2.

Pipeline:
    PICO -> XRoboToolkit -> PicoReader -> _process_3pt_pose (unchanged)
         -> H2ThreePointPose (calibration vs H2 FK, training frame/offsets)
         -> pelvis-relative targets [m, Z-up, wxyz]
         -> world = H2 pelvis pose o target  ->  MuJoCo user_scn markers

Usage:
    # Live (PICO required):
    python gear_sonic/scripts/h2_vr3pt_mujoco_viewer.py

    # Headless self-test without PICO (synthetic circular wrist motion):
    python gear_sonic/scripts/h2_vr3pt_mujoco_viewer.py --fake-input --headless-frames 90

Calibration: stand in the H2 default (zero) posture, then press 'C' in the
viewer window or hold A+B+X+Y on the controllers.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as sRot

from gear_sonic.data.robot_model.instantiation.h2 import instantiate_h2_robot_model
from gear_sonic.scripts.pico_manager_thread_server import (
    PicoReader,
    ThreePointPose,
    _process_3pt_pose,
    get_abxy_buttons,
)
from gear_sonic.utils.teleop.h2_key_frames import (
    H2_KEY_FRAME_OFFSETS,  # noqa: F401 - re-exported for callers
    get_h2_key_frame_poses,  # noqa: F401
)
from gear_sonic.utils.teleop.h2_three_point import H2ThreePointPose

_H2_MJCF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "assets",
    "robot_description",
    "mjcf",
    "h2.xml",
)

# Marker colors (RGBA): left wrist teal, right wrist orange, torso white
_MARKER_RGBA = {
    "left": np.array([0.2, 0.75, 0.75, 0.9]),
    "right": np.array([0.95, 0.65, 0.2, 0.9]),
    "torso": np.array([0.95, 0.95, 0.95, 0.9]),
}
_AXIS_RGBA = [
    np.array([1.0, 0.2, 0.2, 0.9]),  # x red
    np.array([0.2, 1.0, 0.2, 0.9]),  # y green
    np.array([0.3, 0.4, 1.0, 0.9]),  # z blue
]
_AXIS_LEN = 0.09  # [m]
_BALL_R = 0.02  # [m]




class FakeInputReader:
    """PICO stand-in for headless testing: synthetic (24,7) Unity-frame SMPL poses.

    Standing human, wrists tracing slow circles in front of the chest.
    Only joints 0 (root), 12 (neck), 22/23 (wrists) are populated — exactly the
    ones _process_3pt_pose consumes. ``freeze=True`` holds the current frame
    (for the "targets stay still when operator stops" check).
    """

    def __init__(self, hz: float = 30.0):
        self.hz = hz
        self.t = 0.0
        self.freeze = False

    @staticmethod
    def _robot_to_unity_pos(p):
        # Inverse of Q=[[-1,0,0],[0,0,1],[0,1,0]]: robot [X,Y,Z] -> unity [-X, Z, Y]
        return np.array([-p[0], p[2], p[1]])

    def get_latest(self):
        if not self.freeze:
            self.t += 1.0 / self.hz
        poses = np.zeros((24, 7), dtype=np.float64)
        poses[:, 6] = 1.0  # identity quats, scalar-last
        root_r = np.array([0.0, 0.0, 0.95])  # robot frame [m]
        neck_r = root_r + np.array([0.0, 0.0, 0.45])
        w = 2.0 * np.pi * 0.25  # 0.25 Hz circles
        lw_r = root_r + np.array(
            [0.30 + 0.10 * np.cos(w * self.t), 0.25, 0.25 + 0.10 * np.sin(w * self.t)]
        )
        rw_r = root_r + np.array(
            [0.30 + 0.10 * np.cos(w * self.t + np.pi), -0.25, 0.25 + 0.10 * np.sin(w * self.t + np.pi)]
        )
        for idx, p in ((0, root_r), (12, neck_r), (22, lw_r), (23, rw_r)):
            poses[idx, :3] = self._robot_to_unity_pos(p)
        return {"body_poses_np": poses, "timestamp_ns": int(self.t * 1e9)}

    def start(self):
        pass

    def stop(self):
        pass


def _add_marker_geoms(scn, pos_w: np.ndarray, quat_wxyz: np.ndarray, rgba: np.ndarray):
    """Append one sphere + 3 axis capsules for a target pose to user_scn."""
    rot = sRot.from_quat(quat_wxyz, scalar_first=True).as_matrix()
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([_BALL_R, 0, 0]), pos_w, np.eye(3).flatten(), rgba
    )
    scn.ngeom += 1
    for ax in range(3):
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
            _AXIS_RGBA[ax],
        )
        mujoco.mjv_connector(
            g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.004, pos_w, pos_w + _AXIS_LEN * rot[:, ax]
        )
        scn.ngeom += 1


def _targets_to_world(vr_3pt_pose: np.ndarray, pelvis_pos: np.ndarray, pelvis_quat_wxyz: np.ndarray):
    """Pelvis-relative targets -> world poses. Returns list of (pos, quat_wxyz)."""
    rot_p = sRot.from_quat(pelvis_quat_wxyz, scalar_first=True)
    out = []
    for i in range(3):
        pos_w = pelvis_pos + rot_p.apply(vr_3pt_pose[i, :3])
        quat_w = (rot_p * sRot.from_quat(vr_3pt_pose[i, 3:], scalar_first=True)).as_quat(
            scalar_first=True
        )
        out.append((pos_w, quat_w))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hz", type=float, default=30.0, help="Target update rate [Hz]")
    ap.add_argument("--fake-input", action="store_true",
                    help="Synthetic input instead of PICO (self-test, auto-calibrates)")
    ap.add_argument("--headless-frames", type=int, default=0,
                    help="Render N frames offscreen and exit (requires --fake-input or PICO)")
    ap.add_argument("--out-dir", default="/tmp/h2_vr3pt_viewer",
                    help="Output dir for headless frames / logs")
    ap.add_argument("--auto-calibrate", type=float, default=0.0, metavar="SEC",
                    help="Auto-calibrate SEC seconds after the first body frame "
                         "(stand in the H2 zero posture until then)")
    ap.add_argument("--ik", action="store_true",
                    help="Stage 2-B: solve upper-body IK so the H2 pose follows the "
                         "targets (kinematic preview; no dynamics, no policy)")
    ap.add_argument("--pub-port", type=int, default=0,
                    help="If set, ZMQ-PUB pelvis-relative targets on tcp://*:PORT "
                         "(topic 'h2_vr3pt') for the ~/workspace/Mujoco marker bridge")
    args = ap.parse_args()

    # --- MuJoCo H2 model (held static at spawn; mj_forward only, never stepped)
    # h2.xml expects meshdir="meshes/" next to it; the meshes live under urdf/h2.
    # Inject the absolute meshdir at load time instead of touching the asset.
    mesh_dir = os.path.join(os.path.dirname(os.path.dirname(_H2_MJCF)), "urdf", "h2", "meshes")
    xml_text = open(_H2_MJCF).read().replace('meshdir="meshes/"', f'meshdir="{mesh_dir}/"')
    model = mujoco.MjModel.from_xml_string(xml_text)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    pelvis_pos = data.qpos[0:3].copy()  # freejoint: [x y z, qw qx qy qz]
    pelvis_quat = data.qpos[3:7].copy()
    print(f"[Viewer] H2 spawn pelvis: pos={np.round(pelvis_pos,3)} quat(wxyz)={np.round(pelvis_quat,3)}")

    # --- Input source
    if args.fake_input:
        reader = FakeInputReader(hz=args.hz)
    else:
        # Same XRT bring-up as the existing --vr3pt_realtime path
        from gear_sonic.scripts.pico_manager_thread_server import xrt

        if xrt is None:
            raise ImportError("xrobotoolkit_sdk not available - install it or use --fake-input")
        import subprocess

        subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
        xrt.init()
        print("[Viewer] Waiting for PICO body data...")
        while not xrt.is_body_data_available():
            print("  waiting for body data...")
            time.sleep(1)
        reader = PicoReader()
        reader.start()
        print("[Viewer] PICO body data available.")
    three_point = H2ThreePointPose()

    ik = None
    joint_qposadr = {}
    if args.ik:
        from gear_sonic.utils.teleop.h2_upper_ik import H2UpperBodyIK, IK_JOINTS

        ik = H2UpperBodyIK(three_point._robot_model)
        for n in IK_JOINTS:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if jid >= 0:
                joint_qposadr[n] = model.jnt_qposadr[jid]
        print(f"[Viewer] Upper-body IK enabled ({len(joint_qposadr)} joints)")

    pub_socket = None
    if args.pub_port:
        import zmq

        ctx = zmq.Context.instance()
        pub_socket = ctx.socket(zmq.PUB)
        pub_socket.bind(f"tcp://*:{args.pub_port}")
        print(f"[Viewer] Publishing targets on tcp://*:{args.pub_port} (topic 'h2_vr3pt')")

    def publish(vr_3pt: np.ndarray):
        if pub_socket is None:
            return
        payload = {
            "vr_3pt": vr_3pt.tolist(),  # (3,7) [x y z, qw qx qy qz] rel-pelvis [m]
            "calibrated": bool(three_point.is_calibrated),
            "t": time.time(),
        }
        if ik is not None and three_point.is_calibrated:
            payload["q_upper"] = ik.joint_dict()  # {joint_name: rad}
        # Single-part message: the subscriber uses ZMQ CONFLATE, which does not
        # support multipart. The port is dedicated, so no topic frame is needed.
        pub_socket.send(json.dumps(payload).encode())

    calibrate_requested = {"flag": args.fake_input}  # fake input: calibrate on first frame
    ik_state = {"res": None}
    auto_calib_deadline = {"t": None}

    def key_callback(keycode):
        if keycode in (ord("C"), ord("c")):
            calibrate_requested["flag"] = True
            print("[Viewer] Calibration requested (key C) — hold the H2 zero posture")

    def process_frame(sample):
        """One input frame -> world target poses (or None if no sample)."""
        if sample is None:
            return None
        body_poses_np = sample["body_poses_np"]
        if calibrate_requested["flag"]:
            three_point.calibrate_now(body_poses_np)
            calibrate_requested["flag"] = False
        vr_3pt = three_point.process_smpl_pose(body_poses_np)
        if ik is not None and three_point.is_calibrated:
            ik_state["res"] = ik.solve(vr_3pt)
            for n, adr in joint_qposadr.items():
                data.qpos[adr] = ik.joint_dict()[n]
            mujoco.mj_forward(model, data)
        return _targets_to_world(vr_3pt, pelvis_pos, pelvis_quat), vr_3pt

    period = 1.0 / args.hz
    last_print = 0.0

    if args.headless_frames > 0:
        os.makedirs(args.out_dir, exist_ok=True)
        # Offscreen rendering is optional: the math checks below (ranges, mirror,
        # stationarity) do not need pixels. On hosts without EGL/OSMesa we log only.
        renderer = None
        try:
            renderer = mujoco.Renderer(model, height=480, width=640)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = (pelvis_pos[0] + 0.2, pelvis_pos[1], 0.9)
            cam.distance, cam.elevation, cam.azimuth = 2.2, -15.0, 160.0
        except Exception as e:  # noqa: BLE001
            print(f"[Headless] offscreen renderer unavailable ({type(e).__name__}); log-only mode")
        log = []
        for k in range(args.headless_frames):
            if args.fake_input and k == args.headless_frames - 30:
                reader.freeze = True  # last second: operator stops -> targets must stop
            res = process_frame(reader.get_latest())
            if res is None:
                time.sleep(period)
                continue
            targets, vr_3pt = res
            publish(vr_3pt)
            if renderer is not None:
                renderer.update_scene(data, camera=cam)
                for (pos_w, quat_w), key in zip(targets, ["left", "right", "torso"]):
                    _add_marker_geoms(renderer.scene, pos_w, quat_w, _MARKER_RGBA[key])
                if k % 30 == 0 or k == args.headless_frames - 1:
                    frame = renderer.render()
                    try:
                        from PIL import Image

                        Image.fromarray(frame).save(
                            os.path.join(args.out_dir, f"frame_{k:04d}.png")
                        )
                    except ImportError:
                        np.save(os.path.join(args.out_dir, f"frame_{k:04d}.npy"), frame)
            log.append(np.concatenate([vr_3pt[i, :3] for i in range(3)]))
            time.sleep(period if args.pub_port else 0.0)
        log = np.asarray(log)
        np.save(os.path.join(args.out_dir, "targets_rel.npy"), log)
        frozen = log[-25:]
        print(f"[Headless] frames={len(log)}  rel-target ranges [m]:")
        for i, name in enumerate(["L-wrist", "R-wrist", "torso "]):
            seg = log[:, 3 * i : 3 * i + 3]
            print(f"  {name}: min={np.round(seg.min(0),3)} max={np.round(seg.max(0),3)}")
        drift = np.abs(frozen - frozen[-1]).max()
        print(f"[Headless] frozen-input target drift over last 25 frames: {drift:.2e} m (expect 0)")
        print(f"[Headless] frames + targets_rel.npy saved to {args.out_dir}")
        return

    # --- Live interactive viewer
    from mujoco import viewer as mj_viewer

    with mj_viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        print("[Viewer] Running. 'C' (or controllers A+B+X+Y) = calibrate in H2 zero posture.")
        prev_combo = False
        while viewer.is_running():
            t0 = time.time()
            sample = reader.get_latest()
            # Optional timed auto-calibration (operator holds the zero posture)
            if args.auto_calibrate > 0 and sample is not None and not three_point.is_calibrated:
                if auto_calib_deadline["t"] is None:
                    auto_calib_deadline["t"] = time.time() + args.auto_calibrate
                    print(f"[Viewer] Auto-calibrating in {args.auto_calibrate:.0f} s - "
                          "stand in the H2 zero posture")
                elif time.time() >= auto_calib_deadline["t"]:
                    calibrate_requested["flag"] = True
                    auto_calib_deadline["t"] = None
            # Controller combo A+B+X+Y -> calibrate (edge-triggered)
            if not args.fake_input and sample is not None:
                a, b, x, y = get_abxy_buttons(reader)
                combo = a and b and x and y
                if combo and not prev_combo:
                    calibrate_requested["flag"] = True
                    print("[Viewer] Calibration requested (A+B+X+Y)")
                prev_combo = combo
            res = process_frame(sample)
            if res is not None:
                targets, vr_3pt = res
                publish(vr_3pt)
                viewer.user_scn.ngeom = 0
                for (pos_w, quat_w), key in zip(targets, ["left", "right", "torso"]):
                    _add_marker_geoms(viewer.user_scn, pos_w, quat_w, _MARKER_RGBA[key])
                now = time.time()
                if now - last_print > 1.0:
                    msg = " | ".join(
                        f"{n}=({vr_3pt[i,0]:+.3f},{vr_3pt[i,1]:+.3f},{vr_3pt[i,2]:+.3f})"
                        for i, n in enumerate(["L", "R", "T"])
                    )
                    calib = "calibrated" if three_point.is_calibrated else                         "NOT CALIBRATED -> press C (robot/IK stays frozen until then)"
                    ikmsg = ""
                    if ik is not None:
                        ikmsg = (f" | IK res={ik_state['res']:.3f} q_upper SENT"
                                 if ik_state["res"] is not None else " | IK waiting for calibration")
                    print(f"[{calib}]{ikmsg}\n[targets rel-pelvis, m] {msg}")
                    last_print = now
            viewer.sync()
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    if not args.fake_input:
        reader.stop()


if __name__ == "__main__":
    main()
