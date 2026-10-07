"""PICO VR teleoperation of H2 in MuJoCo with the frozen SONIC policy.

Full dynamics (gravity, contacts, falls) with the same strict-native control
stack as sim/h2_edu_sim.py: 50 Hz policy, 4x 0.005 s physics substeps, PD
torques with training gains. The policy is the frozen H2 EDU checkpoint served
by h2_policy_server; this runner uses the server's raw ``encode``/``decode``
ops to drive the encoder's *teleop* and *smpl* universal-token branches with
live PICO data instead of a motion file. Nothing in h2_edu_sim.py or the
policy server changes.

Token layouts follow wbc_h12 (read-only references):
    h2_tools/sonic_tokens.py            field offsets (1790-wide tokenizer)
    config/.../sonic_h2_edu.yaml        encoder branch input lists
    envs/manager_env/mdp/observations.py / commands.py   field semantics

Modes:
    --mode teleop   hands + torso targets from PICO (VR 3-point), legs balance
                    on a standing lower-body command. Feet are NOT tracked.
    --mode smpl     full-body SMPL mimic (24 joints incl. feet) from the PICO
                    full-body stream. Requires the manager started with
                    --robot h2 --num_frames_to_send 10.
    --offline-smpl PKL   headless validation: replay a smpl_filtered clip's
                    SMPL source through the same smpl token path (no PICO).

Keys (viewer): S = start/pause policy, R = reset robot (respawn standing).

Run inside the sim container:
    (cd ~/workspace/h2_policy_server && ./run.sh up)      # policy server
    ../run.sh py sim/h2_vr_teleop_sim.py --viewer --mode teleop
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h2_edu_sim  # reuse reset_robot / base_angular_velocity_local / scene path

from h2_policy_if import H2PolicyClient, h2_joints, protocol

CONTROL_DT = protocol.CONTROL_DT  # 0.02 s
PHYSICS_SUBSTEPS = 4
NUM_DOF = protocol.NUM_DOF

# ---------------------------------------------------------------------------
# Tokenizer layout (wbc_h12/h2_tools/sonic_tokens.py). protocol.py names the
# g1 fields; the teleop/smpl fields are appended here with the same provenance.
# ---------------------------------------------------------------------------
_CMF = protocol.NUM_FUTURE_FRAMES * NUM_DOF * 2  # 620
TOK = dict(protocol.TOKENIZER_OFFSETS)
_base = 3 + _CMF + 6 + 60  # 689
TOK["command_multi_future_lower"] = (_base, 240)  # [jpos 10x12 | jvel 10x12] legs
TOK["vr_3point_local_target"] = (_base + 240, 9)  # [L,R,torso] x xyz
TOK["vr_3point_local_orn_target"] = (_base + 249, 12)  # [L,R,torso] x wxyz
TOK["smpl_joints_mf"] = (_base + 261, 720)  # 10f x 24 joints x 3 (root-local)
TOK["smpl_root_ori_b_mf"] = (_base + 981, 60)  # 10f x rot6d(inv(robot)*smpl_root)
TOK["joint_pos_mf_wrist_for_smpl"] = (_base + 1041, 60)  # 10f x 6 wrist joints
assert _base + 1101 == protocol.TOKENIZER_DIM

ENCODER_SCALAR = {"g1": 0.0, "teleop": 1.0, "smpl": 2.0}  # sonic_tokens.py:138
ONE_HOT = {"g1": (1, 0, 0), "teleop": (0, 1, 0), "smpl": (0, 0, 1)}

SMPL_FRAMES = 10  # smpl_num_future_frames (sonic_h2_edu.yaml)
SMPL_DT = 0.02  # smpl_dt_future_ref_frames

# G1 29-DOF wrist slots inside the manager's joint_pos stream. Semantics match
# the H2 field (L/R roll, L/R pitch, L/R yaw): pico_manager_thread_server.py
# G1_*_WRIST_* constants; H2 target indices are IsaacLab [23..28]
# (joint_pos_multi_future_wrist_for_smpl.yaml).
STREAM_WRIST_SLOTS = [23, 24, 25, 26, 27, 28]


# -- small quat utilities (match server/h2_policy_server.py + training) -------
def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_to_rot6d(q):
    """First two COLUMNS of the rotation matrix, row-major — matches training's
    matrix_from_quat(q)[..., :2].reshape(-1)."""
    return quat_to_mat(q)[:, :2].reshape(-1)


def heading_quat(q):
    """Yaw-only quaternion of q (training calc_heading_quat semantics)."""
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def gravity_dir_from_quat(q):
    """Unit gravity in base frame (server/h2_policy_server.py semantics)."""
    return quat_to_mat(q).T @ np.array([0.0, 0.0, -1.0])


class ProprioHistory:
    """Client-side replica of server ProprioHistory (same priming + layout)."""

    def __init__(self):
        n = protocol.HISTORY_LENGTH
        self.bufs = {
            "base_ang_vel": np.zeros((n, 3), np.float32),
            "joint_pos": np.zeros((n, NUM_DOF), np.float32),
            "joint_vel": np.zeros((n, NUM_DOF), np.float32),
            "actions": np.zeros((n, NUM_DOF), np.float32),
            "gravity_dir": np.zeros((n, 3), np.float32),
        }
        self.bufs["gravity_dir"][:] = (0, 0, -1)
        self.primed = False

    def push(self, ang_vel, joint_pos_rel, joint_vel, gravity, last_action):
        vals = {
            "base_ang_vel": ang_vel, "joint_pos": joint_pos_rel,
            "joint_vel": joint_vel, "gravity_dir": gravity, "actions": last_action,
        }
        for k, buf in self.bufs.items():
            if not self.primed:
                buf[:] = vals[k]
            else:
                buf[:-1] = buf[1:]
                buf[-1] = vals[k]
        self.primed = True

    def flat(self):
        out = np.empty(protocol.PROPRIO_DIM, np.float32)
        for k, (off, size) in protocol.PROPRIO_OFFSETS.items():
            out[off:off + size] = self.bufs[k].reshape(-1)
        return out


# -- PICO stream ------------------------------------------------------------
_POSE_HEADER_SIZE = 1280  # zmq_planner_sender.HEADER_SIZE (docstring says 1024; code says 1280)
_POSE_DTYPES = {"f32": "<f4", "f64": "<f8", "i32": "<i4", "i64": "<i8", "bool": "|b1"}


def unpack_pose_message(msg: bytes, topic: str = "pose"):
    """Inverse of gear_sonic zmq_planner_sender.pack_pose_message (v3)."""
    tl = len(topic)
    header = json.loads(msg[tl:tl + _POSE_HEADER_SIZE].rstrip(b"\x00").decode())
    blob = msg[tl + _POSE_HEADER_SIZE:]
    out, off = {}, 0
    for f in header["fields"]:
        n = int(np.prod(f["shape"])) if f["shape"] else 1
        dt = np.dtype(_POSE_DTYPES[f["dtype"]])
        out[f["name"]] = np.frombuffer(blob, dt, count=n, offset=off).reshape(f["shape"])
        off += n * dt.itemsize
    return out


class PicoStream:
    def __init__(self, host: str, port: int):
        ctx = zmq.Context.instance()
        self.sock = ctx.socket(zmq.SUB)
        self.sock.connect(f"tcp://{host}:{port}")
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "pose")
        self.latest = None

    def poll(self):
        while True:  # drain, keep newest
            try:
                msg = self.sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                return self.latest
            try:
                self.latest = unpack_pose_message(bytes(msg))
            except Exception as e:
                if not getattr(self, "_warned", False):
                    self._warned = True
                    print(f"[PicoStream] WARNING: cannot parse pose message ({type(e).__name__}: {e})")


class OfflineSmplStream:
    """Replays a smpl_filtered clip through the smpl token path (no PICO).

    Mirrors the LIVE producer (pico_manager process_smpl_joints) exactly:
        q_zup   = Rx(+90) * q_yup                       (smpl_root_ytoz_up)
        q_rm    = q_zup * conj([.5,.5,.5,.5])           (remove_smpl_base_rot)
        joints  = inv(q_rm) applied to root-centered z-up joints
    Wrist joint futures come from the paired retargeted pkl (--offline-dof),
    like training's joint_pos_multi_future_wrist_for_smpl.
    """

    RX90 = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])
    BASE = np.array([0.5, 0.5, 0.5, 0.5])

    @staticmethod
    def _load(path):
        if str(path).endswith(".npz"):
            return dict(np.load(path, allow_pickle=True))
        import joblib

        raw = joblib.load(path)
        return raw

    def __init__(self, pkl_path: str, dof_pkl: str | None = None):
        raw = self._load(pkl_path)
        if isinstance(raw, dict) and "pose_aa" not in raw:
            raw = raw[next(iter(raw))]
        self.pose_aa = np.asarray(raw["pose_aa"], np.float64).reshape(len(raw["pose_aa"]), -1)
        self.joints = np.asarray(raw["smpl_joints"], np.float64)  # (T,24,3) y-up
        self.dt_src = 1.0 / float(raw.get("fps", 50.0))
        self.T = len(self.pose_aa)
        self.dof = None
        if dof_pkl:
            d = self._load(dof_pkl)
            if isinstance(d, dict) and "dof" not in d:
                d = d[next(iter(d))]
            self.dof = np.asarray(d["dof"], np.float64)
            self.dof_dt = 1.0 / float(d.get("fps", 50.0))
        print(f"[OfflineSmpl] {Path(pkl_path).name}: {self.T} frames @ {1/self.dt_src:.0f} fps"
              + (f", wrists from dof ({len(self.dof)} frames)" if self.dof is not None else ""))

    def _aa_to_quat(self, aa):
        th = np.linalg.norm(aa)
        if th < 1e-9:
            return np.array([1.0, 0, 0, 0])
        ax = aa / th
        return np.array([np.cos(th / 2), *(np.sin(th / 2) * ax)])

    def poll_at(self, t):
        smpl_joints, body_quat, wrists = [], [], []
        for k in range(SMPL_FRAMES):
            tk = t + k * SMPL_DT
            i = min(int(tk / self.dt_src), self.T - 1)
            q_yup = self._aa_to_quat(self.pose_aa[i, :3])
            q_zup = quat_mul(self.RX90, q_yup)
            q_rm = quat_mul(q_zup, quat_inv(self.BASE))
            body_quat.append(q_rm)
            # dataset smpl_joints are stored z-up already (measured: height spans
            # axis 2); training rotates them raw by inv(root quat) — match exactly.
            smpl_joints.append(self.joints[i] @ quat_to_mat(q_rm))  # rows: R^T v
            if self.dof is not None:
                im = min(int(tk / self.dof_dt), len(self.dof) - 1)
                # H2 EDU wrist slots: IsaacLab [25..30] (sonic_h2_edu.yaml joints_idx)
                wrists.append(h2_joints.to_isaaclab(self.dof[im])[25:31])
            else:
                wrists.append(np.zeros(6))
        jp = np.zeros((SMPL_FRAMES, 29), np.float32)
        jp[:, STREAM_WRIST_SLOTS] = np.asarray(wrists, np.float32)
        return {
            "smpl_joints": np.asarray(smpl_joints, np.float32),
            "body_quat_w": np.asarray(body_quat, np.float32),
            "joint_pos": jp,
            "vr_position": np.zeros(9, np.float32),
            "vr_orientation": np.tile([1, 0, 0, 0], 3).astype(np.float32),
        }


# -- token builders -----------------------------------------------------------
def put(obs, name, val):
    off, size = TOK[name]
    flat = np.asarray(val, np.float32).reshape(-1)
    assert flat.size == size, f"{name}: {flat.size} != {size}"
    obs[off:off + size] = flat


def build_teleop_tokens(msg, robot_quat):
    obs = np.zeros(protocol.TOKENIZER_DIM, np.float32)
    put(obs, "encoder_index", ONE_HOT["teleop"])
    # standing lower-body command: default legs, zero velocity, 10 futures
    jpos = np.tile(h2_joints.DEFAULT_ANGLES_MUJOCO[:12], (protocol.NUM_FUTURE_FRAMES, 1))
    jvel = np.zeros_like(jpos)
    put(obs, "command_multi_future_lower", np.concatenate([jpos.reshape(-1), jvel.reshape(-1)]))
    # upright reference: heading-only target orientation
    rel = quat_mul(quat_inv(robot_quat), heading_quat(robot_quat))
    put(obs, "motion_anchor_ori_b", quat_to_rot6d(rel))
    put(obs, "vr_3point_local_target", msg["vr_position"])
    put(obs, "vr_3point_local_orn_target", msg["vr_orientation"])
    return obs


def build_smpl_tokens(msg, robot_quat):
    obs = np.zeros(protocol.TOKENIZER_DIM, np.float32)
    put(obs, "encoder_index", ONE_HOT["smpl"])
    sj = np.asarray(msg["smpl_joints"], np.float32)
    bq = np.asarray(msg["body_quat_w"], np.float32)
    jp = np.asarray(msg["joint_pos"], np.float32)
    n = sj.shape[0]
    if n < SMPL_FRAMES:  # pad by repeating the newest frame
        pad = SMPL_FRAMES - n
        sj = np.concatenate([sj, np.repeat(sj[-1:], pad, 0)])
        bq = np.concatenate([bq, np.repeat(bq[-1:], pad, 0)])
        jp = np.concatenate([jp, np.repeat(jp[-1:], pad, 0)])
    sj, bq, jp = sj[-SMPL_FRAMES:], bq[-SMPL_FRAMES:], jp[-SMPL_FRAMES:]
    put(obs, "smpl_joints_mf", sj)
    r6 = [quat_to_rot6d(quat_mul(quat_inv(robot_quat), q)) for q in bq]
    put(obs, "smpl_root_ori_b_mf", np.asarray(r6))
    put(obs, "joint_pos_mf_wrist_for_smpl", jp[:, STREAM_WRIST_SLOTS])
    return obs


# -- main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["teleop", "smpl"], default="teleop")
    ap.add_argument("--endpoint", default=protocol.DEFAULT_ENDPOINT)
    ap.add_argument("--pico-host", default="127.0.0.1")
    ap.add_argument("--pico-port", type=int, default=5556)
    ap.add_argument("--offline-smpl", default=None, help="smpl_filtered pkl for headless validation")
    ap.add_argument("--offline-dof", default=None,
                    help="paired retargeted pkl for wrist joint reference (offline smpl)")
    ap.add_argument("--fixture", default=None,
                    help="JSON with constant vr_position/vr_orientation (teleop standing test)")
    ap.add_argument("--stand-fixture",
                    default=str(Path(__file__).resolve().parent / "teleop_stand_fixture.json"),
                    help="standing targets (EDU FK) used while WAITING, before tracking starts")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--seconds", type=float, default=0.0, help="auto-stop after N sim seconds (0=run)")
    ap.add_argument("--autostart", action="store_true", help="start policy immediately (no 'S')")
    ap.add_argument("--out", default=None, help="write a result JSON (diagnostic)")
    ap.add_argument("--video", default=None, help="offscreen mp4 (EGL, 25 fps)")
    args = ap.parse_args()

    scene = Path(__file__).resolve().parent.parent / "assets" / "h2_edu_scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)

    client = H2PolicyClient(endpoint=args.endpoint, session=f"teleop-{args.mode}")
    print(f"policy {client.info.get('decoder', '?')} @ {args.endpoint} | mode={args.mode}")

    if args.fixture:
        fx = json.loads(Path(args.fixture).read_text())
        fx = {k: np.asarray(v, np.float32) for k, v in fx.items()}

        class _Fixed:
            def poll(self):
                return fx
        stream = _Fixed()
        args.mode = "teleop"
        print(f"[Teleop] fixture input: {args.fixture}")
    elif args.offline_smpl:
        stream = OfflineSmplStream(args.offline_smpl, dof_pkl=args.offline_dof)
        args.mode = "smpl"
    else:
        stream = PicoStream(args.pico_host, args.pico_port)
        print(f"PICO stream: tcp://{args.pico_host}:{args.pico_port} (run the manager with --robot h2"
              + (" --num_frames_to_send 10" if args.mode == "smpl" else "") + ")")

    stand_msg = None
    sf = Path(args.stand_fixture)
    if sf.exists():
        fx = json.loads(sf.read_text())
        stand_msg = {k: np.asarray(v, np.float32) for k, v in fx.items()}
        print(f"[Teleop] stand fixture loaded: {sf.name} (policy balances in place until tracking starts)")
    else:
        print(f"[Teleop] WARNING: no stand fixture at {sf} - robot will PD-hold while waiting (will crouch)")

    kp, kd = h2_joints.KP_MUJOCO, h2_joints.KD_MUJOCO
    default_mj = h2_joints.DEFAULT_ANGLES_MUJOCO
    default_il = h2_joints.DEFAULT_ANGLES_ISAACLAB
    scale_il = h2_joints.ACTION_SCALE_ISAACLAB

    state = {"tracking": bool(args.autostart or args.offline_smpl or args.fixture),
             "reset": True}

    def key_cb(keycode):
        if keycode in (ord("S"), ord("s")):
            state["tracking"] = not state["tracking"]
            print(f"[Teleop] {'TRACKING (following you)' if state['tracking'] else 'STANDING (policy holds in place)'}")
        elif keycode in (ord("R"), ord("r")):
            state["reset"] = True
            print("[Teleop] reset requested")

    recorder = None
    if args.video:
        recorder = mujoco.Renderer(model, height=480, width=640)
        rec_cam = mujoco.MjvCamera()
        rec_cam.distance, rec_cam.elevation, rec_cam.azimuth = 3.5, -18.0, 135.0
        rec_frames = []

    viewer = None
    if args.viewer:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(model, data, key_callback=key_cb)
        print("[Teleop] keys: S=track/stand toggle (or squeeze BOTH grips), R=reset. "
              "Robot stands in place until tracking starts.")

    hist = ProprioHistory()
    last_action = np.zeros(NUM_DOF, np.float32)
    step = 0
    falls = 0
    fallen = False
    t_wall = time.perf_counter()
    log = {"t": [], "pelvis_z": [], "fall_t": []}

    try:
        while True:
            if state["reset"]:
                h2_edu_sim.reset_robot(model, data)
                hist = ProprioHistory()
                last_action = np.zeros(NUM_DOF, np.float32)
                step = 0
                fallen = False
                state["reset"] = False
                t_wall = time.perf_counter()
                print("[Teleop] robot respawned (standing, 1.04 m)")

            sim_t = step * CONTROL_DT
            msg = stream.poll_at(sim_t) if args.offline_smpl else stream.poll()

            # Controller toggle: squeeze BOTH grips (edge-triggered) = S key
            if msg is not None and "left_grip" in msg and "right_grip" in msg:
                combo = float(msg["left_grip"][0]) > 0.8 and float(msg["right_grip"][0]) > 0.8
                if combo and not state.get("_grip_prev", False):
                    state["tracking"] = not state["tracking"]
                    print(f"[Teleop] {'TRACKING' if state['tracking'] else 'STANDING'} (grip combo)")
                state["_grip_prev"] = combo

            if msg is not None and not state.get("_stream_seen", False):
                state["_stream_seen"] = True
                print("[Teleop] PICO stream RECEIVING - squeeze BOTH grips (or press S) to start tracking")

            use_live = state["tracking"] and msg is not None
            if use_live or stand_msg is not None:
                robot_quat = data.qpos[3:7].copy()
                if use_live:
                    obs = (build_smpl_tokens(msg, robot_quat) if args.mode == "smpl"
                           else build_teleop_tokens(msg, robot_quat))
                    enc_idx = ENCODER_SCALAR[args.mode]
                else:
                    # WAITING: policy balances at the EDU default stance (teleop branch)
                    obs = build_teleop_tokens(stand_msg, robot_quat)
                    enc_idx = ENCODER_SCALAR["teleop"]
                tokens = client.encode(obs, encoder_index=enc_idx)
                q_il = h2_joints.to_isaaclab(data.qpos[7:])
                v_il = h2_joints.to_isaaclab(data.qvel[6:])
                hist.push(
                    ang_vel=h2_edu_sim.base_angular_velocity_local(data).astype(np.float32),
                    joint_pos_rel=(q_il - default_il).astype(np.float32),
                    joint_vel=v_il.astype(np.float32),
                    gravity=gravity_dir_from_quat(robot_quat).astype(np.float32),
                    last_action=last_action,
                )
                action = client.decode(np.concatenate([tokens, hist.flat()]))
                last_action = action.astype(np.float32)
                target = h2_joints.to_mujoco(default_il + action * scale_il)
            else:
                target = default_mj  # no fixture and no stream yet: PD hold

            for _ in range(PHYSICS_SUBSTEPS):
                data.ctrl[:] = kp * (target - data.qpos[7:]) - kd * data.qvel[6:]
                mujoco.mj_step(model, data)
            step += 1

            if recorder is not None and step % 2 == 0:
                rec_cam.lookat[:] = (data.qpos[0], data.qpos[1], 0.9)
                recorder.update_scene(data, camera=rec_cam)
                rec_frames.append(recorder.render())

            z = float(data.qpos[2])
            log["t"].append(sim_t)
            log["pelvis_z"].append(round(z, 4))
            if z < 0.35 and not fallen:
                fallen = True
                falls += 1
                log["fall_t"].append(round(sim_t, 2))
                print(f"[Teleop] FALL at t={sim_t:.2f} s (pelvis {z:.2f} m) — press R to reset")

            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
                # real-time pacing
                lag = (time.perf_counter() - t_wall) - sim_t
                if lag < 0:
                    time.sleep(-lag)
            if args.seconds and sim_t >= args.seconds:
                break
            if step % 250 == 0:
                grips = ""
                if msg is not None and "left_grip" in msg and "right_grip" in msg:
                    grips = (f" grips L={float(msg['left_grip'][0]):.2f}"
                             f" R={float(msg['right_grip'][0]):.2f}")
                print(f"[Teleop] t={sim_t:5.1f}s pelvis_z={z:.3f} falls={falls} "
                      f"{'TRACKING' if state['tracking'] else 'STANDING'}{grips}")
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()

    xy0 = np.array([0.0, 0.0])
    xy_disp = float(np.linalg.norm(data.qpos[0:2] - xy0))
    print(f"[Teleop] done: {step * CONTROL_DT:.1f} s, falls={falls}, root XY displacement {xy_disp:.2f} m")
    if recorder is not None and rec_frames:
        import imageio.v2 as imageio

        imageio.mimwrite(args.video, rec_frames, fps=25, codec="libx264",
                         output_params=["-pix_fmt", "yuv420p"])
        print(f"[Teleop] video: {args.video}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"mode": args.mode, "seconds": step * CONTROL_DT, "falls": falls,
             "root_xy_disp_m": xy_disp,
             "fall_t": log["fall_t"], "pelvis_z_min": min(log["pelvis_z"] or [0]),
             "pelvis_z_final": (log["pelvis_z"] or [0])[-1]}, indent=1))
        print(f"[Teleop] wrote {args.out}")


if __name__ == "__main__":
    main()
