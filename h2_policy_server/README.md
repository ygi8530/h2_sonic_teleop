# h2_policy_server

Inference server for the H2 EDU SONIC motion-tracking policy, and the wrapper
layer that keeps `/workspace/wbc_h12` untouched.

```
                       ┌──────────────────────┐
                       │  h2-policy-server    │  ONNX encoder + decoder
                       │  ZeroMQ REP :5555    │  reference clips, proprio history
                       └──────────┬───────────┘  joint reindex, action scale
                                  │
                   ZeroMQ REQ/REP over TCP (host network)
                    ┌─────────────┴─────────────┐
          ┌─────────▼──────────┐      ┌──────────▼─────────┐
          │ isaac-lab-base     │      │ h2-mujoco          │
          │ dev container      │      │ dev container      │
          └────────────────────┘      └────────────────────┘

  /workspace/wbc_h12  ──read-only bind mount──▶  all three containers
```

## Why the wrapper exists

`wbc_h12` is upstream-owned and must stay byte-identical so it can be replaced
wholesale. It is therefore mounted `read_only` everywhere, and nothing in it is
imported: the server reads only its **data** files — the two ONNX graphs, the
motion pkls and `mjcf/h2_edu.xml`.

Everything the raw graphs do not carry lives here instead:

| concern | where |
|---|---|
| wire format, observation layouts | `h2_policy_if/protocol.py` |
| joint orders, PD gains, action scale, default pose, armature, torque limits | `h2_policy_if/h2_joints.py` |
| client both simulators import | `h2_policy_if/client.py` |
| tokenizer packing, proprio history, inference | `server/h2_policy_server.py` |

`h2_joints.py` **copies** numbers out of `wbc_h12` (they cannot be imported —
`robots/h2.py` imports `isaaclab` at module scope, which this slim image
deliberately lacks). `tools/verify_constants.py` re-derives every one of them from
the `wbc_h12` sources with an AST parser and fails on drift, so an upstream swap
is a one-command check:

```bash
./run.sh verify
```

## Quick start

```bash
./run.sh build          # ~15 s, no Isaac Sim / torch / mujoco in this image
./run.sh verify         # confirm the copied constants still match wbc_h12
./run.sh up             # detached, tcp://127.0.0.1:5555
./run.sh smoke          # end-to-end request check, no simulator needed
./run.sh logs
./run.sh down
```

## Protocol

ZeroMQ REQ/REP over TCP. One request is one reply, each a two-frame multipart
message: a UTF-8 JSON header, then a raw little-endian float32 payload whose
arrays the header describes. Chosen over gRPC (no codegen step, ~1 ms round trip
instead of ~1 ms of framework overhead on top), raw TCP (framing comes for free)
and Unix sockets (no shared volume needed between containers).

| op | request | reply |
|---|---|---|
| `ping` | — | model names, `num_dof`, `control_dt`, providers |
| `motions` | — | available reference clip names |
| `reset` | `motion`, `ref_order`, `loop` | clip `frames`, `fps`, `duration` |
| `act` | `t`, `order`, `update_history`; `base_quat[4]`, `joint_pos[31]`, `joint_vel[31]`, `base_ang_vel[3]` | `joint_pos_target[31]`, `action[31]`, `tokens[64]` |
| `encode` | `tokenizer_obs[1790]`, `encoder_index` | `tokens[64]` |
| `decode` | `obs[1054]` | `action[31]` |

`act` is the op simulators use; it hides the tokenizer layout, the ten-frame
history, the joint reindexing and `action_scale`. `encode`/`decode` are raw
passthroughs for debugging.

### Sessions

The server keeps one reference clip, proprio history and last action per
`session` name. Two clients sharing the endpoint must use different names —
`H2PolicyClient(session=...)`. Call `reset` after every simulator reset,
otherwise the policy is shown ten frames of a robot it is no longer driving.

`update_history=False` evaluates against the history as it stands without
advancing it, which is what an open-loop sweep over a clip needs.

## Model I/O

Control rate **50 Hz** (`sim.dt` 0.005 × `decimation` 4).

**Encoder** `[1, 1791] → [1, 64]` — `[encoder_index scalar | tokenizer_obs 1790]`,
output 64 FSQ tokens bounded to ±1.

| offset | size | field |
|---|---|---|
| 0 | 3 | `encoder_index` one-hot `[g1, teleop, smpl]` |
| 3 | 620 | `command_multi_future`: reference `[jpos 10×31 | jvel 10×31]`, future frames 0.1 s apart |
| 623 | 6 | `motion_anchor_ori_b`: rot6d of `quat_inv(base) ⊗ ref_root` |
| 629 | 60 | the same rot6d tiled ×10 |
| 689 | 1101 | teleop and smpl fields, zero in g1 mode |

**Decoder** `[1, 1054] → [1, 31]` — `[tokens 64 | proprio 990]`, output is a
residual on the default pose in **IsaacLab joint order**.

| offset | size | field |
|---|---|---|
| 0 | 64 | `token_state` |
| 64 | 30 | `base_ang_vel` 10×3, base frame [rad/s] |
| 94 | 310 | `joint_pos_rel` 10×31, `q − q_default` [rad] |
| 404 | 310 | `joint_vel` 10×31 [rad/s] |
| 714 | 310 | `last_action` 10×31, raw policy output |
| 1024 | 30 | `gravity_dir` 10×3, `quat_conj(base) ⊗ (0,0,−1)` |

Every history block is **oldest frame first**. Joint target:
`q_target = q_default + action × action_scale`.

## Joint orders

Two exist, and confusing them produces a policy that runs and then falls:

* **MuJoCo** — `mjcf/h2_edu.xml`, also the URDF and hardware motor order, and the
  order motion pkls store `dof` in.
* **IsaacLab** — breadth-first articulation order. Every vector the policy itself
  consumes or produces is in this order.

`act` takes `order=` so each client sends its native order.

### `ref_order`: a real discrepancy, now settled

`wbc_h12/h2_tools/sonic_tokens.py:110` packs `command_multi_future` from the raw
pkl, i.e. **MuJoCo order**. The training path reindexes `dof` to **IsaacLab
order** first (`motion_lib_base.py:1599`), and the ONNX export wrapper
(`inference_helpers.py:200`) only slices the flat vector by name — it applies no
permutation. So the encoder expects IsaacLab order.

Settled by experiment, 12 s of `walk_forward_loop_001__A029` in closed loop:

| `ref_order` | result |
|---|---|
| `isaaclab` | stayed up, 30 s, min pelvis 0.673 m |
| `mujoco` | **fell at 6.7 s**, min pelvis 0.079 m |

`isaaclab` is the client default. `reset(ref_order="mujoco")` reproduces the
`sonic_tokens.py` behaviour if the comparison is needed again.

Reproduce: `cd ../Mujoco && ./run.sh sim --motion walk_forward_loop_001__A029 --ab-order`

## Layout

```
h2_policy_server/
├── docker/Dockerfile              python:3.10-slim + onnxruntime, numpy, pyzmq, joblib
├── docker/docker-compose.yaml     network_mode: host, wbc_h12 mounted read_only
├── h2_policy_if/                  shared with both simulator containers (read-only there)
│   ├── protocol.py                frames, dims, observation layouts
│   ├── h2_joints.py               joint orders, gains, scale, default pose, armature, torque
│   └── client.py                  H2PolicyClient
├── server/h2_policy_server.py     the REP loop
├── tools/smoke_test.py            transport + open-loop sweep
├── tools/verify_constants.py      re-derive h2_joints.py from wbc_h12
└── run.sh
```
