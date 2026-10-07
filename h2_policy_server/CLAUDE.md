# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

An inference server for the already-trained H2 EDU SONIC motion-tracking policy, plus the
interface and evaluation layer that both simulator repositories import. It holds no training
code and no policy weights: the ONNX graphs live in `../wbc_h12/h2_tools/models/`, and this
repo wraps them so that `wbc_h12` can be replaced wholesale without touching anything here.

`README.md` is the reference for the wire protocol, the exact observation layouts and the
model I/O tables. Read it before changing `protocol.py`, `h2_joints.py` or the server loop —
the byte offsets documented there are the contract, not a description.

## Commands

Every command runs on the **host**. The containers have no `docker` CLI, so `./run.sh` cannot
be used from inside one.

```bash
./run.sh build     # ~15 s; python:3.10-slim + onnxruntime/numpy/pyzmq/joblib, no torch or Isaac Sim
./run.sh up        # detached, host network, tcp://127.0.0.1:5555
./run.sh verify    # re-derive every copied constant from wbc_h12 and fail on drift
./run.sh smoke     # transport + open-loop sweep against a running server
./run.sh logs      # follow
./run.sh down
```

There is no unit-test suite. `verify` and `smoke` are the checks: run `verify` after any
change to `h2_joints.py` or any `wbc_h12` update, and `smoke` after any change to the server
loop or `protocol.py`. `verify` and `smoke` each run in a throw-away container built from the
image, so edits under `tools/` take effect immediately, while edits under `server/` need
`./run.sh down && ./run.sh up`.

`tools/` is a library for the simulator repos rather than a set of standalone entry points.
They invoke it from their own containers, e.g. `python3 ../../h2_policy_server/tools/ofat_sweep.py …`
or by loading `tools/evaluation_matrix.py` by path (`Mujoco/run_evaluation.py`). Run those
scripts from the simulator side, not here.

## The `wbc_h12` boundary

`wbc_h12` is upstream-owned, is mounted `read_only` in all three containers, and is never
imported. Only its **data** files are read: the two ONNX graphs, the motion pkls and
`mjcf/h2_edu.xml`.

Consequently `h2_policy_if/h2_joints.py` **copies** joint orders, PD gains, action scale,
default pose, armature and torque limits out of `wbc_h12` instead of importing them
(`robots/h2.py` imports `isaaclab` at module scope, which this slim image deliberately
lacks). `tools/verify_constants.py` re-derives each value from the `wbc_h12` sources with an
AST parser. Never hand-edit a number in `h2_joints.py` to make something work — change it
only to match upstream, and prove it with `./run.sh verify`.

## Invariants that fail silently when broken

**Two joint orders exist.** MuJoCo order is `mjcf/h2_edu.xml`, the URDF, the hardware motor
order, and the order motion pkls store `dof` in. IsaacLab order is breadth-first articulation
order, and **every vector the policy consumes or produces is in IsaacLab order**. Mixing them
yields a policy that runs, looks plausible, and then falls. `act` takes `order=` so each
client sends its native order.

**`ref_order` is settled, do not re-litigate it casually.** The encoder expects
`command_multi_future` in IsaacLab order, even though `wbc_h12/h2_tools/sonic_tokens.py:110`
packs it in MuJoCo order. Closed-loop experiment: `isaaclab` stayed up 30 s (min pelvis
0.673 m), `mujoco` fell at 6.7 s (0.079 m). `isaaclab` is the client default;
`reset(ref_order="mujoco")` reproduces the old behaviour if the comparison is needed again.

**History blocks are oldest frame first**, ten frames each, and the joint target is
`q_target = q_default + action × action_scale`. Control rate is 50 Hz (`sim.dt` 0.005 ×
decimation 4). Decoder input is 1054 = 64 tokens + 990 proprio; encoder input is 1791 =
3-wide `encoder_index` one-hot + 1790 tokenizer obs. The server asserts these against the
loaded graphs at start-up, so a differently-shaped policy is rejected rather than silently
mis-fed.

**Sessions are per-name state.** The server keeps one reference clip, proprio history and
last action per `session`. Two clients on one endpoint must use different names. Call `reset`
after every simulator reset, or the policy sees ten frames of a robot it no longer drives.
`update_history=False` evaluates without advancing history, which is what an open-loop sweep
needs.

## Evaluation layer (`h2_policy_if/evaluation.py`, `tools/`)

This layer exists to keep evaluation claims honest, and its rules are deliberate:

- **Provenance grades must never be merged.** `STRICT_BASELINE`, `TRAIN_DIST` (inside the
  ranges training randomised, sourced in `TRAIN_RANDOMIZATION`) and `OOD_SIM2REAL` (an axis
  training never randomised) are reported in separate tables. `OOD_SIM2REAL` requires
  `--allow-ood` and the value is the operator's choice, so it maps a boundary but carries no
  training provenance.
- **`OOD` here means perturbation axes, not unseen motions.** Every campaign uses the same
  in-distribution clip set. Do not describe these results as motion-level generalization.
- **An axis a backend cannot express is refused, never approximated** (`AXIS_SUPPORT`).
  Writing it anyway would produce a flat sweep that reads as insensitivity instead of as a
  value that never took effect.
- **`diverged` is not `fall`.** A diverged run left the physical band and was stopped;
  merging the two corrupts fall statistics.
- `collect_provenance` records the policy SHA-256, `wbc_h12` git HEAD, backend version,
  solver overrides and the source file:line of every physical constant. Keep it that way:
  results without provenance cannot be compared later.

`tools/evaluation_matrix.py:SUMMARY_COLUMNS` defines `summary.csv` for every backend. Two
parsing traps: **`survival_s` is the requested episode length even for falls** (20.0 with
`fell_at_s` 1.4), so use `fell_at_s` for fall timing and `termination_reason == "completed"`
for success; and **`condition` is the axis name concatenated with the value and no separator**
(`body_mass_scale2.5`, `torso_com_y-0.05`, `observation_latency0.02`), so parse by
longest-known-axis-prefix rather than by regex on digits. `tools/campaign_report.py` merges
result directories into `summary.md` + `failures.md`.

## Neighbours

```
workspace/
├── h2_policy_server/                    this repo — inference + interface + eval library
├── wbc_h12/                             upstream SONIC port; read-only, never modified
├── Mujoco/                              MuJoCo sim2sim runner and its campaign results
├── IsaacLab/sim2sim_motion_retargeting/ Isaac Lab (PhysX / Newton MJWarp) runner
├── IsaacLab/work_h2_motion_retargeting/ retargeting research (raw / GMR / SOMA, PD replay)
└── WORKFLOWS.md                         which container does what, and the ops procedures
```

`h2_policy_if/` is bind-mounted read-only into the simulator containers as `/opt/h2_policy_if`
so all three share one source of truth for the wire protocol and the joint tables. A change
here changes both simulators; check both before assuming it is local.

`sim2sim_motion_retargeting/` in this repo is an empty mount point, not source.

Read `../WORKFLOWS.md` before creating a new work folder inside a devcontainer: a directory
under the container workspace without a bind mount lives only in the container's writable
layer and is destroyed when the container is recreated, and `/workspace/isaaclab/logs` is a
volume that gets emptied by container teardown. Two research folders were lost that way on
2026-09-28.
