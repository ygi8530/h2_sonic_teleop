"""Wire protocol for the H2 EDU policy server.

Transport: ZeroMQ REQ/REP over TCP. One request = one reply, both as a two-frame
multipart message::

    frame 0   UTF-8 JSON header  (op, fields, array descriptors)
    frame 1   raw little-endian float32 payload (may be empty)

Arrays are declared in the header under ``arrays`` as an ordered list of
``{"name": str, "shape": [int, ...]}`` and packed back-to-back into frame 1 in
that order. This keeps the numeric path zero-copy while the header stays
human-readable for debugging with ``zmqc``/``tcpdump``.

Ops
---
``ping``    -> server/model info. No arrays.
``motions`` -> names of the reference motions the server can load.
``reset``   -> start or restart a session; clears its proprio history.
``act``     -> one control step: robot state in, joint position targets out.
``encode``  -> raw encoder passthrough (1790 tokenizer obs -> 64 tokens).
``decode``  -> raw decoder passthrough (1054 obs -> 31 actions).

The payload is always float32: the models are float32 and the proprio history is
float32, so no other dtype is needed and allowing one would only invite silent
precision mismatches.
"""

from __future__ import annotations

import json

import numpy as np

DEFAULT_PORT = 5555
DEFAULT_ENDPOINT = f"tcp://127.0.0.1:{DEFAULT_PORT}"

PROTOCOL_VERSION = 1

# Model dimensions, asserted against the ONNX graphs at server start-up.
NUM_DOF = 31
TOKEN_DIM = 64
TOKENIZER_DIM = 1790
ENCODER_INPUT_DIM = 1 + TOKENIZER_DIM  # 1791
PROPRIO_DIM = 990
DECODER_INPUT_DIM = TOKEN_DIM + PROPRIO_DIM  # 1054

HISTORY_LENGTH = 10
NUM_FUTURE_FRAMES = 10
DT_FUTURE_REF_FRAMES = 0.1  # [s] spacing of the reference frames fed to the encoder
CONTROL_DT = 0.02  # [s] 50 Hz: sim.dt 0.005 * decimation 4

OPS = ("ping", "motions", "reset", "act", "encode", "decode")

# --------------------------------------------------------------------------
# Observation layouts. Kept here rather than in the server so tools and clients
# that build raw observations describe them with the same constants.
# --------------------------------------------------------------------------

# Offsets into the 1790-wide tokenizer observation, as ``{name: (offset, size)}``.
# Mirrors wbc_h12/h2_tools/sonic_tokens.py:11-27. The fields after
# ``motion_anchor_ori_b_mf`` belong to the teleop and smpl encoders and stay zero
# in g1 (motion-tracking) mode, so only the four that are written are named.
_CMF_SIZE = NUM_FUTURE_FRAMES * NUM_DOF * 2  # 620
TOKENIZER_OFFSETS = {
    "encoder_index": (0, 3),  # one-hot [g1, teleop, smpl]
    "command_multi_future": (3, _CMF_SIZE),  # [jpos 10x31 | jvel 10x31]
    "motion_anchor_ori_b": (3 + _CMF_SIZE, 6),  # rot6d
    "motion_anchor_ori_b_mf": (3 + _CMF_SIZE + 6, 60),  # the rot6d tiled x10
}

# Offsets into the decoder's 990-wide proprio tail. The order is the field
# DECLARATION order of PolicyCfg (gear_sonic .../mdp/observations.py:107-128),
# which is what the concatenated observation follows -- not the order of the
# hydra defaults list in policy/local_dir_hist.yaml. Each block is
# oldest-frame-first (Isaac Lab CircularBuffer.buffer: "most recent entry at the
# end"), and every per-joint block is in IsaacLab joint order.
PROPRIO_OFFSETS = {
    "base_ang_vel": (0, 30),  # 10x3, base frame [rad/s]
    "joint_pos": (30, 310),  # 10x31, q - q_default [rad]
    "joint_vel": (340, 310),  # 10x31 [rad/s]
    "actions": (650, 310),  # 10x31, raw policy output
    "gravity_dir": (960, 30),  # 10x3, unit gravity in base frame
}

# Joint orders the wire accepts. "mujoco" is the order of mjcf/h2_edu.xml, which
# is also the hardware/URDF order and the order the motion pkl files use.
# "isaaclab" is the breadth-first order the policy's own vectors use.
JOINT_ORDERS = ("mujoco", "isaaclab")


class ProtocolError(RuntimeError):
    """Raised when a peer sends a message this module cannot interpret."""


def pack(header: dict, arrays: dict[str, np.ndarray] | None = None) -> list[bytes]:
    """Serialise one message into ZeroMQ multipart frames.

    Args:
        header: JSON-serialisable fields. The ``arrays`` key is added here and
            must not already be present.
        arrays: Named float32 arrays, packed into frame 1 in iteration order.

    Returns:
        ``[header_bytes, payload_bytes]``.

    Raises:
        ProtocolError: If ``header`` already declares ``arrays``.
    """
    if "arrays" in header:
        raise ProtocolError("header must not set 'arrays'; pass arrays= instead")
    arrays = arrays or {}
    descriptors = []
    chunks = []
    for name, value in arrays.items():
        a = np.ascontiguousarray(value, dtype=np.float32)
        descriptors.append({"name": name, "shape": list(a.shape)})
        chunks.append(a.tobytes())
    full = dict(header)
    full["v"] = PROTOCOL_VERSION
    full["arrays"] = descriptors
    payload = b"".join(chunks) if chunks else b""
    return [json.dumps(full).encode("utf-8"), payload]


def unpack(frames: list[bytes]) -> tuple[dict, dict[str, np.ndarray]]:
    """Deserialise ZeroMQ multipart frames produced by :func:`pack`.

    Args:
        frames: Exactly two frames, header then payload.

    Returns:
        ``(header, arrays)``. ``header`` still carries the ``arrays``
        descriptors; ``arrays`` maps each name to a reshaped float32 array.

    Raises:
        ProtocolError: On a frame count, version, or payload length mismatch.
    """
    if len(frames) != 2:
        raise ProtocolError(f"expected 2 frames, got {len(frames)}")
    try:
        header = json.loads(frames[0].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"bad header: {exc}") from exc
    version = header.get("v")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"protocol version {version!r}, this build speaks {PROTOCOL_VERSION}")

    buf = np.frombuffer(frames[1], dtype="<f4")
    arrays: dict[str, np.ndarray] = {}
    offset = 0
    for desc in header.get("arrays", []):
        shape = tuple(int(d) for d in desc["shape"])
        count = int(np.prod(shape)) if shape else 1
        if offset + count > buf.size:
            raise ProtocolError(
                f"payload too short for {desc['name']!r}: need {count} float32 at offset {offset}, "
                f"payload holds {buf.size}"
            )
        arrays[desc["name"]] = buf[offset : offset + count].reshape(shape)
        offset += count
    if offset != buf.size:
        raise ProtocolError(f"payload has {buf.size - offset} trailing float32 not declared in header")
    return header, arrays
