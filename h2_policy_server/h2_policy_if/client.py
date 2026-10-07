"""Client for the H2 EDU policy server.

Both the Isaac Lab and the MuJoCo simulation containers import this module, so it
depends on nothing but ``numpy`` and ``pyzmq``.

Typical use::

    from h2_policy_if import H2PolicyClient

    policy = H2PolicyClient()                    # tcp://127.0.0.1:5555
    policy.reset(motion="idle_loop_003__A041")
    for step in range(500):
        out = policy.act(
            t=step * policy.control_dt,
            base_quat=base_quat_wxyz,
            joint_pos=q,                          # [rad], MuJoCo order
            joint_vel=qd,                         # [rad/s], MuJoCo order
            base_ang_vel=omega_base_local,        # [rad/s], base frame
        )
        q_target = out["joint_pos_target"]        # [rad], MuJoCo order
"""

from __future__ import annotations

import numpy as np
import zmq

from . import protocol
from .protocol import ProtocolError


class H2PolicyError(RuntimeError):
    """Raised when the server reports an error or stops answering."""


class H2PolicyClient:
    """Synchronous REQ/REP client for one policy session.

    The socket is recreated whenever a request times out, because a ZeroMQ REQ
    socket that has sent a request it never got an answer to cannot be reused.

    Args:
        endpoint: ZeroMQ endpoint of the server.
        session: Session name. The server keeps one proprio history per session,
            so two clients sharing an endpoint must use different names.
        timeout_ms: Per-request receive timeout [ms].
        retries: How many times to rebuild the socket and resend before failing.
    """

    def __init__(
        self,
        endpoint: str = protocol.DEFAULT_ENDPOINT,
        session: str = "default",
        timeout_ms: int = 5000,
        retries: int = 2,
    ) -> None:
        self.endpoint = endpoint
        self.session = session
        self.timeout_ms = timeout_ms
        self.retries = retries
        self._ctx = zmq.Context.instance()
        self._sock: zmq.Socket | None = None
        self._connect()
        self.info = self.ping()

    # -- lifecycle ---------------------------------------------------------

    def _connect(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
        sock = self._ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        sock.connect(self.endpoint)
        self._sock = sock

    def close(self) -> None:
        """Close the socket. Further calls raise."""
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None

    def __enter__(self) -> "H2PolicyClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- transport ---------------------------------------------------------

    def _request(self, op: str, fields: dict | None = None, arrays: dict | None = None):
        """Send one request and return ``(header, arrays)`` from the reply.

        Raises:
            H2PolicyError: On timeout after all retries, or if the server replies
                with ``ok: false``.
        """
        if self._sock is None:
            raise H2PolicyError("client is closed")
        header = {"op": op, "session": self.session}
        header.update(fields or {})
        frames = protocol.pack(header, arrays)

        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                self._sock.send_multipart(frames)
                reply = self._sock.recv_multipart()
                rh, ra = protocol.unpack(reply)
                if not rh.get("ok", False):
                    raise H2PolicyError(f"server rejected {op!r}: {rh.get('error', '<no message>')}")
                return rh, ra
            except (zmq.Again, ProtocolError) as exc:
                last_exc = exc
                self._connect()  # a REQ socket mid-exchange is unusable
        raise H2PolicyError(
            f"no usable reply to {op!r} from {self.endpoint} after {self.retries + 1} attempts "
            f"({type(last_exc).__name__}: {last_exc}). Is the policy server running?"
        )

    # -- API ---------------------------------------------------------------

    def ping(self) -> dict:
        """Return server and model metadata (model paths, dims, control rate)."""
        header, _ = self._request("ping")
        return header

    @property
    def control_dt(self) -> float:
        """Control period the policy was trained at [s]."""
        return float(self.info.get("control_dt", protocol.CONTROL_DT))

    def motions(self) -> list[str]:
        """Return the reference motion names the server can load."""
        header, _ = self._request("motions")
        return list(header["motions"])

    def reset(
        self,
        motion: str,
        ref_order: str = "isaaclab",
        loop: bool = True,
        joint_default_offset=None,
    ) -> dict:
        """Start or restart this session.

        Clears the proprio history and loads the reference motion. Call this once
        before the first :meth:`act` and again after every simulator reset,
        otherwise the policy sees ten frames of a robot it is no longer driving.

        Args:
            motion: Motion name from :meth:`motions`, or a path the server can read.
            ref_order: Joint order in which the reference command is fed to the
                encoder. ``"isaaclab"`` matches the training path
                (``motion_lib_base.py:1599`` reindexes ``dof`` before the
                tokenizer). ``"mujoco"`` reproduces what
                ``h2_tools/sonic_tokens.py`` sends, for an A/B check.
            loop: Wrap ``t`` around the clip length instead of clamping at the end.
            joint_default_offset: Optional ``[31]`` per-joint offset in IsaacLab
                order, added to the default pose the server uses for both the
                proprio ``joint_pos_rel`` term and the action-to-target
                conversion. Reproduces training's ``randomize_joint_default_pos``.
                The caller must apply the SAME offset to the simulator's reset
                pose, or the two halves disagree silently.

        Returns:
            Reply header: clip ``frames``, ``fps``, ``duration`` [s].
        """
        if ref_order not in protocol.JOINT_ORDERS:
            raise ValueError(f"ref_order must be one of {protocol.JOINT_ORDERS}, got {ref_order!r}")
        arrays = None
        if joint_default_offset is not None:
            arrays = {"joint_default_offset": np.asarray(joint_default_offset).reshape(protocol.NUM_DOF)}
        header, _ = self._request(
            "reset", {"motion": motion, "ref_order": ref_order, "loop": bool(loop)}, arrays
        )
        return header

    def act(
        self,
        t: float,
        base_quat: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        order: str = "mujoco",
        update_history: bool = True,
    ) -> dict:
        """Run one control step.

        Args:
            t: Reference clip time [s]. The encoder is fed frames at
                ``t + k * 0.1`` for ``k`` in 0..9.
            base_quat: Pelvis orientation, world->body, ``wxyz``, shape ``[4]``.
            joint_pos: Measured joint positions [rad], shape ``[31]``.
            joint_vel: Measured joint velocities [rad/s], shape ``[31]``.
            base_ang_vel: Pelvis angular velocity in the **base frame** [rad/s],
                shape ``[3]``.
            order: Joint order of ``joint_pos``/``joint_vel`` and of the returned
                ``joint_pos_target``. One of ``"mujoco"``, ``"isaaclab"``.
            update_history: Whether to advance the session's proprio history and
                remember this action. Pass ``False`` for an open-loop sweep over a
                clip, where consecutive calls must not influence one another.

        Returns:
            ``joint_pos_target`` [rad] in ``order``, ``action`` (raw policy output,
            IsaacLab order), ``tokens`` (64 FSQ tokens), and ``ref_frame``.
        """
        if order not in protocol.JOINT_ORDERS:
            raise ValueError(f"order must be one of {protocol.JOINT_ORDERS}, got {order!r}")
        _, arrays = self._request(
            "act",
            {"t": float(t), "order": order, "update_history": bool(update_history)},
            {
                "base_quat": np.asarray(base_quat).reshape(4),
                "joint_pos": np.asarray(joint_pos).reshape(protocol.NUM_DOF),
                "joint_vel": np.asarray(joint_vel).reshape(protocol.NUM_DOF),
                "base_ang_vel": np.asarray(base_ang_vel).reshape(3),
            },
        )
        return arrays

    def encode(self, tokenizer_obs: np.ndarray, encoder_index: float = 0.0) -> np.ndarray:
        """Raw encoder passthrough.

        Args:
            tokenizer_obs: Shape ``[1790]``, laid out as documented in
                ``wbc_h12/h2_tools/sonic_tokens.py``.
            encoder_index: Which encoder to run: ``0.0`` = g1 (motion tracking).

        Returns:
            64 FSQ tokens.
        """
        _, arrays = self._request(
            "encode",
            {"encoder_index": float(encoder_index)},
            {"tokenizer_obs": np.asarray(tokenizer_obs).reshape(protocol.TOKENIZER_DIM)},
        )
        return arrays["tokens"]

    def decode(self, obs: np.ndarray) -> np.ndarray:
        """Raw decoder passthrough.

        Args:
            obs: Shape ``[1054]`` = 64 tokens + 990 proprio.

        Returns:
            31 actions in IsaacLab order (residuals, before ``action_scale``).
        """
        _, arrays = self._request(
            "decode", None, {"obs": np.asarray(obs).reshape(protocol.DECODER_INPUT_DIM)}
        )
        return arrays["action"]
