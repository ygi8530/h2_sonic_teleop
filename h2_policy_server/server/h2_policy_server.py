#!/usr/bin/env python3
"""ZeroMQ REQ/REP inference server for the H2 EDU SONIC motion-tracking policy.

Wraps the ONNX encoder/decoder pair and the sample motions that ship in
``wbc_h12`` without writing anything into it, and without importing any Python
from it: ``wbc_h12`` is mounted read-only and only its data files (the two ONNX
graphs, the motion pkls and ``mjcf/h2_edu.xml``) are read. Everything the policy
needs on top of the raw graphs -- tokenizer layout, joint reindexing, the
ten-frame proprio history, ``action_scale`` and the default pose -- lives here, so
the simulator clients stay thin and an upstream ``wbc_h12`` swap needs no client
change.

Not importing ``h2_tools/sonic_tokens.py`` is deliberate: it packs the encoder's
``command_multi_future`` in MuJoCo joint order, while the training path reindexes
``dof`` to IsaacLab order first (``motion_lib_base.py:1599``). The ``ref_order``
argument of ``reset`` exposes both so the discrepancy can be settled by
experiment rather than inherited silently.

Run::

    python server/h2_policy_server.py --wbc-root /opt/wbc_h12 --bind tcp://0.0.0.0:5555
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
import onnxruntime as ort
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h2_policy_if import h2_joints, protocol  # noqa: E402

LOG = logging.getLogger("h2_policy_server")

# Observation layouts live in the shared protocol module so clients and tools
# describe them with the same constants.
TOK_OFF = protocol.TOKENIZER_OFFSETS
PROP_OFF = protocol.PROPRIO_OFFSETS


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two ``wxyz`` quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Conjugate of a unit ``wxyz`` quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_to_rot6d(q: np.ndarray) -> np.ndarray:
    """First two columns of the rotation matrix, row-major over ``(3, 2)``.

    Matches ``isaaclab.utils.math.matrix_from_quat(...)[..., :2].reshape(-1)``,
    which is how ``motion_anchor_ori_b`` is built at training time.
    """
    w, x, y, z = q
    m = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return m[:, :2].reshape(-1)


def gravity_dir_from_quat(q_wxyz: np.ndarray) -> np.ndarray:
    """Project world gravity into the base frame.

    Args:
        q_wxyz: Pelvis orientation, world->body, ``wxyz``.

    Returns:
        ``quat_conj(q) applied to (0, 0, -1)``, shape ``[3]``. Upright is
        ``(0, 0, -1)``.
    """
    qi = quat_inv(q_wxyz)
    v = np.array([0.0, 0.0, 0.0, -1.0])
    return quat_mul(quat_mul(qi, v), q_wxyz)[1:]


class MotionReference:
    """Reference clip loaded from a ``motion_lib`` pkl.

    ``dof`` in the pkl is in MuJoCo order; ``ref_order`` decides whether it is
    reindexed to IsaacLab order before being handed to the encoder.

    Args:
        pkl_path: Path to the motion pkl inside the read-only ``wbc_h12``.
        ref_order: ``"isaaclab"`` (training path) or ``"mujoco"``.
        loop: Wrap time around the clip instead of clamping at the last frame.
    """

    def __init__(self, pkl_path: Path, ref_order: str, loop: bool) -> None:
        import joblib  # local: only the server needs it

        entry = next(iter(joblib.load(pkl_path).values()))
        dof = np.asarray(entry["dof"], dtype=np.float32)
        if dof.shape[1] != protocol.NUM_DOF:
            raise ValueError(f"{pkl_path.name}: {dof.shape[1]} DOF, expected {protocol.NUM_DOF}")
        self.name = pkl_path.stem
        self.ref_order = ref_order
        self.loop = loop
        self.fps = float(entry.get("fps", 50.0))
        self.frames = dof.shape[0]
        self.root_rot_xyzw = np.asarray(entry["root_rot"], dtype=np.float32)

        # Finite-difference velocities, the same convention motion_lib uses.
        dof_vel = np.zeros_like(dof)
        dof_vel[:-1] = (dof[1:] - dof[:-1]) * self.fps
        dof_vel[-1] = dof_vel[-2] if self.frames > 1 else 0.0

        if ref_order == "isaaclab":
            self.dof = h2_joints.to_isaaclab(dof)
            self.dof_vel = h2_joints.to_isaaclab(dof_vel)
        else:
            self.dof, self.dof_vel = dof, dof_vel

    @property
    def duration(self) -> float:
        """Clip length [s]."""
        return self.frames / self.fps

    def frame_at(self, t_sec: float) -> int:
        """Index of the frame nearest ``t_sec`` [s], looped or clamped."""
        if self.loop and self.duration > 0:
            t_sec = t_sec % self.duration
        return int(np.clip(round(t_sec * self.fps), 0, self.frames - 1))

    def command_multi_future(self, t_sec: float) -> np.ndarray:
        """Encoder command block: ``[jpos 10x31 | jvel 10x31]``, shape ``[620]``."""
        idxs = [
            self.frame_at(t_sec + k * protocol.DT_FUTURE_REF_FRAMES)
            for k in range(protocol.NUM_FUTURE_FRAMES)
        ]
        return np.concatenate(
            [self.dof[idxs].reshape(-1), self.dof_vel[idxs].reshape(-1)]
        ).astype(np.float32)

    def root_quat_wxyz(self, frame: int) -> np.ndarray:
        """Reference root orientation at ``frame`` as ``wxyz``."""
        x, y, z, w = self.root_rot_xyzw[frame]
        return np.array([w, x, y, z], dtype=np.float64)


class ProprioHistory:
    """Ten-frame history of the five proprio terms, oldest frame first.

    Fresh sessions are primed with the robot's first observation repeated ten
    times, which is what a simulator reset looks like to the policy: no motion
    yet and no action taken.
    """

    def __init__(self) -> None:
        n = protocol.HISTORY_LENGTH
        self.ang_vel = np.zeros((n, 3), np.float32)
        self.joint_pos = np.zeros((n, protocol.NUM_DOF), np.float32)
        self.joint_vel = np.zeros((n, protocol.NUM_DOF), np.float32)
        self.actions = np.zeros((n, protocol.NUM_DOF), np.float32)
        self.gravity = np.zeros((n, 3), np.float32)
        self.gravity[:] = (0.0, 0.0, -1.0)
        self.primed = False

    def push(
        self,
        ang_vel: np.ndarray,
        joint_pos_rel: np.ndarray,
        joint_vel: np.ndarray,
        gravity: np.ndarray,
        last_action: np.ndarray,
    ) -> None:
        """Append one frame, dropping the oldest. All arrays in IsaacLab order."""
        if not self.primed:
            self.ang_vel[:] = ang_vel
            self.joint_pos[:] = joint_pos_rel
            self.joint_vel[:] = joint_vel
            self.gravity[:] = gravity
            self.actions[:] = last_action
            self.primed = True
            return
        for buf, value in (
            (self.ang_vel, ang_vel),
            (self.joint_pos, joint_pos_rel),
            (self.joint_vel, joint_vel),
            (self.gravity, gravity),
            (self.actions, last_action),
        ):
            buf[:-1] = buf[1:]
            buf[-1] = value

    def copy(self) -> "ProprioHistory":
        """Return an independent copy, for evaluating without advancing state."""
        clone = ProprioHistory()
        clone.ang_vel = self.ang_vel.copy()
        clone.joint_pos = self.joint_pos.copy()
        clone.joint_vel = self.joint_vel.copy()
        clone.actions = self.actions.copy()
        clone.gravity = self.gravity.copy()
        clone.primed = self.primed
        return clone

    def flat(self) -> np.ndarray:
        """Concatenate into the decoder's 990-wide proprio tail."""
        out = np.empty(protocol.PROPRIO_DIM, np.float32)
        for key, buf in (
            ("base_ang_vel", self.ang_vel),
            ("joint_pos", self.joint_pos),
            ("joint_vel", self.joint_vel),
            ("actions", self.actions),
            ("gravity_dir", self.gravity),
        ):
            off, size = PROP_OFF[key]
            out[off : off + size] = buf.reshape(-1)
        return out


class Session:
    """Per-client state: the reference clip, the history and the last action.

    Args:
        motion: The reference clip this session tracks.
        default_offset: Per-joint offset added to the default pose, IsaacLab
            order. Training's ``randomize_joint_default_pos`` shifts the
            articulation's default joint positions, which Isaac Lab uses both as
            the reset pose and as the action offset. The simulator applies it to
            the former; this applies it to the latter, so the two cannot drift
            apart. Zeros for an unperturbed run.
    """

    def __init__(self, motion: MotionReference, default_offset: np.ndarray | None = None) -> None:
        self.motion = motion
        self.history = ProprioHistory()
        self.last_action = np.zeros(protocol.NUM_DOF, np.float32)
        self.steps = 0
        self.default_pose = h2_joints.DEFAULT_ANGLES_ISAACLAB.astype(np.float64)
        if default_offset is not None:
            self.default_pose = self.default_pose + np.asarray(default_offset, np.float64)
        self.default_offset = (
            np.zeros(protocol.NUM_DOF) if default_offset is None
            else np.asarray(default_offset, np.float64)
        )


class PolicyServer:
    """Loads the ONNX pair once and serves requests until interrupted.

    Args:
        wbc_root: Path to the read-only ``wbc_h12`` checkout.
        encoder: Encoder ONNX path. Defaults to the newest ``*_encoder.onnx``
            under ``h2_tools/models``.
        decoder: Decoder ONNX path, resolved the same way.
        providers: onnxruntime execution providers, highest priority first.
        motion_roots: Extra read-only directories to discover ``*.pkl`` clips in,
            searched after the upstream sample set. A clip name that appears in
            more than one root resolves to the first root that has it.
    """

    def __init__(
        self,
        wbc_root: Path,
        encoder: Path | None = None,
        decoder: Path | None = None,
        providers: list[str] | None = None,
        motion_roots: list[str] | None = None,
    ) -> None:
        self.wbc_root = wbc_root.resolve()
        self.models_dir = self.wbc_root / "h2_tools" / "models"
        # Motion roots, searched in order. The first is upstream's three-clip
        # demo subset; the rest come from H2_MOTION_ROOTS so an evaluation can
        # serve its own retargeted clips without anything being written into
        # wbc_h12. Every root is read-only as far as this server is concerned.
        roots = [self.wbc_root / "h2_tools" / "sample_motions"]
        for extra in (motion_roots or []):
            path = Path(extra).resolve()
            if path not in roots:
                roots.append(path)
        self.motion_roots = roots
        self.motions_dir = roots[0]  # kept for messages that name the default root
        self.mjcf_path = (
            self.wbc_root / "gear_sonic/data/assets/robot_description/mjcf/h2_edu.xml"
        )
        if not self.models_dir.is_dir():
            raise SystemExit(
                f"{self.models_dir} not found. Is --wbc-root pointing at the wbc_h12 checkout, "
                "and was `git lfs pull` run there?"
            )

        # The joint tables are copies of wbc_h12 values; refuse to serve if the
        # asset they were copied from has changed underneath us.
        h2_joints.check_against_mjcf(str(self.mjcf_path))

        self.encoder_path = encoder or self._newest("*_encoder.onnx")
        self.decoder_path = decoder or self._newest("*_decoder.onnx")
        providers = providers or ["CPUExecutionProvider"]
        LOG.info("encoder %s", self.encoder_path.name)
        LOG.info("decoder %s", self.decoder_path.name)
        self.enc = ort.InferenceSession(str(self.encoder_path), providers=providers)
        self.dec = ort.InferenceSession(str(self.decoder_path), providers=providers)
        self.enc_input = self.enc.get_inputs()[0].name
        self.dec_input = self.dec.get_inputs()[0].name
        self._check_dims()
        self.sessions: dict[str, Session] = {}
        self.started = time.time()

    def _newest(self, pattern: str) -> Path:
        matches = sorted(self.models_dir.glob(pattern))
        if not matches:
            raise SystemExit(f"no {pattern} in {self.models_dir}")
        return matches[-1]

    def _check_dims(self) -> None:
        """Fail at start-up, not mid-episode, if an export changed shape."""
        enc_in = int(self.enc.get_inputs()[0].shape[1])
        dec_in = int(self.dec.get_inputs()[0].shape[1])
        dec_out = int(self.dec.get_outputs()[0].shape[1])
        if enc_in != protocol.ENCODER_INPUT_DIM:
            raise SystemExit(
                f"encoder input is {enc_in}, this server speaks {protocol.ENCODER_INPUT_DIM} "
                f"(1 + tokenizer {protocol.TOKENIZER_DIM}). The tokenizer layout changed upstream."
            )
        if dec_in != protocol.DECODER_INPUT_DIM:
            raise SystemExit(
                f"decoder input is {dec_in}, expected {protocol.DECODER_INPUT_DIM} "
                f"(64 tokens + {protocol.PROPRIO_DIM} proprio)."
            )
        if dec_out != protocol.NUM_DOF:
            raise SystemExit(f"decoder output is {dec_out}, expected {protocol.NUM_DOF}")
        LOG.info("layout OK: encoder 1+%d, decoder %d = 64 + %d -> %d",
                 protocol.TOKENIZER_DIM, dec_in, protocol.PROPRIO_DIM, dec_out)

    # -- inference ---------------------------------------------------------

    def _tokens(self, motion: MotionReference, t: float, base_quat: np.ndarray) -> np.ndarray:
        obs = np.zeros(protocol.TOKENIZER_DIM, np.float32)
        off, size = TOK_OFF["encoder_index"]
        obs[off : off + size] = (1.0, 0.0, 0.0)  # one-hot [g1, teleop, smpl]
        off, size = TOK_OFF["command_multi_future"]
        obs[off : off + size] = motion.command_multi_future(t)
        frame = motion.frame_at(t)
        rel = quat_mul(quat_inv(base_quat), motion.root_quat_wxyz(frame))
        r6 = quat_to_rot6d(rel).astype(np.float32)
        off, size = TOK_OFF["motion_anchor_ori_b"]
        obs[off : off + size] = r6
        off, size = TOK_OFF["motion_anchor_ori_b_mf"]
        obs[off : off + size] = np.tile(r6, protocol.NUM_FUTURE_FRAMES)
        return self._run_encoder(obs, 0.0)

    def _run_encoder(self, tokenizer_obs: np.ndarray, encoder_index: float) -> np.ndarray:
        full = np.concatenate(
            [np.array([encoder_index], np.float32), tokenizer_obs.astype(np.float32)]
        )[None, :]
        return self.enc.run(None, {self.enc_input: full})[0].reshape(-1).astype(np.float32)

    def _run_decoder(self, obs: np.ndarray) -> np.ndarray:
        return (
            self.dec.run(None, {self.dec_input: obs.astype(np.float32)[None, :]})[0]
            .reshape(-1)
            .astype(np.float32)
        )

    # -- handlers ----------------------------------------------------------

    def handle(self, header: dict, arrays: dict) -> tuple[dict, dict]:
        """Dispatch one request.

        Returns:
            ``(reply_fields, reply_arrays)``. ``ok`` is added by the caller.
        """
        op = header.get("op")
        if op not in protocol.OPS:
            raise ValueError(f"unknown op {op!r}; known ops are {', '.join(protocol.OPS)}")
        return getattr(self, f"_op_{op}")(header, arrays)

    def _op_ping(self, header: dict, arrays: dict) -> tuple[dict, dict]:
        return {
            "encoder": self.encoder_path.name,
            "decoder": self.decoder_path.name,
            "wbc_root": str(self.wbc_root),
            "num_dof": protocol.NUM_DOF,
            "control_dt": protocol.CONTROL_DT,
            "history_length": protocol.HISTORY_LENGTH,
            "providers": self.enc.get_providers(),
            "uptime_s": round(time.time() - self.started, 1),
            "sessions": sorted(self.sessions),
        }, {}

    def _discover(self) -> dict[str, Path]:
        """Map every discoverable clip name to the file that first provides it.

        Searched recursively so a retargeted set can keep its session layout.
        Earlier roots win, which makes the upstream demo clips authoritative.
        """
        found: dict[str, Path] = {}
        for root in self.motion_roots:
            if not root.is_dir():
                continue
            for pkl in sorted(root.rglob("*.pkl")):
                found.setdefault(pkl.stem, pkl)
        return found

    def _op_motions(self, header: dict, arrays: dict) -> tuple[dict, dict]:
        found = self._discover()
        return {
            "motions": sorted(found),
            "roots": [str(r) for r in self.motion_roots],
            "counts": {str(r): sum(1 for p in found.values() if str(p).startswith(str(r)))
                       for r in self.motion_roots},
        }, {}

    def _resolve_motion(self, name: str) -> Path:
        direct = Path(name)
        if direct.is_file():
            return direct
        found = self._discover()
        if name in found:
            return found[name]
        available = ", ".join(sorted(found)[:12])
        more = "" if len(found) <= 12 else f" (+{len(found) - 12} more)"
        raise FileNotFoundError(
            f"motion {name!r} not found in {[str(r) for r in self.motion_roots]}. "
            f"Available: {available}{more}"
        )

    def _op_reset(self, header: dict, arrays: dict) -> tuple[dict, dict]:
        name = header["session"]
        motion = MotionReference(
            self._resolve_motion(header["motion"]),
            ref_order=header.get("ref_order", "isaaclab"),
            loop=bool(header.get("loop", True)),
        )
        offset = arrays.get("joint_default_offset")
        if offset is not None and offset.shape != (protocol.NUM_DOF,):
            raise ValueError(
                f"joint_default_offset must be [{protocol.NUM_DOF}], got {list(offset.shape)}"
            )
        self.sessions[name] = Session(motion, default_offset=offset)
        LOG.info(
            "session %r reset: motion=%s frames=%d fps=%.1f ref_order=%s",
            name, motion.name, motion.frames, motion.fps, motion.ref_order,
        )
        return {
            "motion": motion.name,
            "frames": motion.frames,
            "fps": motion.fps,
            "duration": round(motion.duration, 3),
            "ref_order": motion.ref_order,
            "joint_default_offset_abs_max": float(np.abs(self.sessions[name].default_offset).max()),
        }, {}

    def _op_act(self, header: dict, arrays: dict) -> tuple[dict, dict]:
        name = header["session"]
        session = self.sessions.get(name)
        if session is None:
            raise KeyError(f"session {name!r} has no reference motion; call reset first")

        order = header.get("order", "mujoco")
        joint_pos = np.asarray(arrays["joint_pos"], np.float64)
        joint_vel = np.asarray(arrays["joint_vel"], np.float64)
        if order == "mujoco":
            joint_pos_il = h2_joints.to_isaaclab(joint_pos)
            joint_vel_il = h2_joints.to_isaaclab(joint_vel)
        else:
            joint_pos_il, joint_vel_il = joint_pos, joint_vel

        base_quat = np.asarray(arrays["base_quat"], np.float64)
        norm = np.linalg.norm(base_quat)
        if not 0.9 < norm < 1.1:
            raise ValueError(f"base_quat is not a unit quaternion (|q| = {norm:.4f}); expected wxyz")
        base_quat = base_quat / norm

        t = float(header["t"])
        tokens = self._tokens(session.motion, t, base_quat)

        # update_history=False evaluates the policy against the history as it
        # already stands, leaving the session untouched. That makes repeated calls
        # independent, which is what an open-loop sweep over a clip needs: with
        # the robot held still, feeding the action back would let the action
        # history ratchet against a body that never moves.
        update = bool(header.get("update_history", True))
        history = session.history if update else session.history.copy()
        history.push(
            ang_vel=np.asarray(arrays["base_ang_vel"], np.float32),
            joint_pos_rel=(joint_pos_il - session.default_pose).astype(np.float32),
            joint_vel=joint_vel_il.astype(np.float32),
            gravity=gravity_dir_from_quat(base_quat).astype(np.float32),
            last_action=session.last_action,
        )
        action = self._run_decoder(np.concatenate([tokens, history.flat()]))
        if update:
            session.last_action = action
            session.steps += 1

        target_il = session.default_pose + action * h2_joints.ACTION_SCALE_ISAACLAB
        target = h2_joints.to_mujoco(target_il) if order == "mujoco" else target_il
        return {"ref_frame": session.motion.frame_at(t), "steps": session.steps}, {
            "joint_pos_target": target,
            "action": action,
            "tokens": tokens,
        }

    def _op_encode(self, header: dict, arrays: dict) -> tuple[dict, dict]:
        tokens = self._run_encoder(
            np.asarray(arrays["tokenizer_obs"]), float(header.get("encoder_index", 0.0))
        )
        return {}, {"tokens": tokens}

    def _op_decode(self, header: dict, arrays: dict) -> tuple[dict, dict]:
        return {}, {"action": self._run_decoder(np.asarray(arrays["obs"]))}

    # -- loop --------------------------------------------------------------

    def serve_forever(self, bind: str) -> None:
        """Bind a REP socket and answer requests until interrupted."""
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.REP)
        sock.bind(bind)
        LOG.info("listening on %s", bind)
        try:
            while True:
                frames = sock.recv_multipart()
                try:
                    header, arrays = protocol.unpack(frames)
                    fields, out_arrays = self.handle(header, arrays)
                    fields["ok"] = True
                except Exception as exc:  # one bad request must not kill the server
                    LOG.warning("%s: %s", type(exc).__name__, exc)
                    fields = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    out_arrays = {}
                sock.send_multipart(protocol.pack(fields, out_arrays))
        except KeyboardInterrupt:
            LOG.info("interrupted, shutting down")
        finally:
            sock.close(linger=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--wbc-root",
        type=Path,
        default=Path(os.environ.get("WBC_H12_ROOT", "/opt/wbc_h12")),
        help="read-only wbc_h12 checkout (env: WBC_H12_ROOT)",
    )
    parser.add_argument("--encoder", type=Path, default=None, help="override encoder ONNX path")
    parser.add_argument("--decoder", type=Path, default=None, help="override decoder ONNX path")
    parser.add_argument(
        "--bind",
        default=os.environ.get("H2_POLICY_BIND", f"tcp://0.0.0.0:{protocol.DEFAULT_PORT}"),
        help="ZeroMQ bind endpoint (env: H2_POLICY_BIND)",
    )
    parser.add_argument(
        "--providers",
        default=os.environ.get("H2_POLICY_PROVIDERS", "CPUExecutionProvider"),
        help="comma-separated onnxruntime execution providers",
    )
    parser.add_argument(
        "--motion-root", action="append", default=None, metavar="DIR",
        help="extra read-only directory of motion .pkl files; repeatable "
             "(env: H2_MOTION_ROOTS, colon-separated)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(), format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    roots = list(args.motion_root or [])
    roots += [r for r in os.environ.get("H2_MOTION_ROOTS", "").split(":") if r]
    server = PolicyServer(
        args.wbc_root,
        encoder=args.encoder,
        decoder=args.decoder,
        providers=[p.strip() for p in args.providers.split(",") if p.strip()],
        motion_roots=roots,
    )
    server.serve_forever(args.bind)


if __name__ == "__main__":
    main()
