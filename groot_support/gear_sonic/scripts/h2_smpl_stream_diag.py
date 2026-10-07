"""Record the manager's pose stream for offline axis diagnosis (no robot involved).

Run while the manager is in POSE mode, then follow the on-screen motion script.
Saves body_quat_w / smpl_joints / vr fields to an .npz for analysis.

Usage:
    python gear_sonic/scripts/h2_smpl_stream_diag.py --seconds 20 --out /tmp/smpl_diag.npz
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import zmq

_HEADER = 1280
_DTYPES = {"f32": "<f4", "f64": "<f8", "i32": "<i4", "i64": "<i8", "bool": "|b1"}


def unpack(msg: bytes, topic: str = "pose"):
    tl = len(topic)
    header = json.loads(msg[tl:tl + _HEADER].rstrip(b"\x00").decode())
    blob = msg[tl + _HEADER:]
    out, off = {}, 0
    for f in header["fields"]:
        n = int(np.prod(f["shape"])) if f["shape"] else 1
        dt = np.dtype(_DTYPES[f["dtype"]])
        out[f["name"]] = np.frombuffer(blob, dt, count=n, offset=off).reshape(f["shape"]).copy()
        off += n * dt.itemsize
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--out", default="/tmp/smpl_diag.npz")
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://{args.host}:{args.port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "pose")

    print("Waiting for stream (manager must be in POSE mode)...")
    sub.recv()
    print("Stream OK. Recording starts in 2 s — follow this script:")
    print("  0-5 s   : 정면 직립, 정지")
    print("  5-10 s  : 허리를 '앞으로' 숙였다가 천천히 복귀")
    print("  10-15 s : 몸을 '왼쪽'으로 기울였다가 복귀")
    print("  15-end  : 제자리에서 '왼쪽'으로 90도 돌기")
    time.sleep(2.0)

    rec = {"t": [], "body_quat_w": [], "head_local": [], "vr_position": []}
    t0 = time.time()
    n = 0
    while time.time() - t0 < args.seconds:
        msg = unpack(bytes(sub.recv()))
        t = time.time() - t0
        rec["t"].append(t)
        rec["body_quat_w"].append(msg["body_quat_w"][-1])          # newest frame, wxyz
        rec["head_local"].append(msg["smpl_joints"][-1, 15])        # SMPL head joint, root-local
        rec["vr_position"].append(msg["vr_position"])               # calibrated [L,R,neck]
        n += 1
        if n % 50 == 0:
            print(f"  {t:5.1f} s  ({n} frames)")
    np.savez(args.out, **{k: np.asarray(v) for k, v in rec.items()})
    print(f"saved {n} frames -> {args.out}")


if __name__ == "__main__":
    main()
