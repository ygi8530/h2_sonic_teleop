"""Build a rollout command for each simulator the sweep drivers can target.

The drivers (``ofat_sweep.py``, ``monte_carlo.py``) only decide *what* to run;
this decides *how* to run it, so that adding a backend does not mean editing
every driver. Each backend advertises which perturbation axes it can express
exactly, taken from ``evaluation.AXIS_SUPPORT`` rather than restated here -- an
axis a simulator cannot express is refused, never approximated.
"""

from __future__ import annotations

from pathlib import Path

# The drivers run on the host; the runners run inside their own containers. Only
# host paths are used to *launch* from, and the result path stays relative to the
# launch directory so it resolves the same on both sides of the container
# boundary -- an absolute host path would not exist inside the container.
_WORKSPACE = Path(__file__).resolve().parents[2]

# Where each backend is driven from and how. ``argv`` is the prefix; the driver
# appends the run's own flags, which are spelled identically on both runners.
BACKENDS = {
    "mujoco": {
        "cwd": str(_WORKSPACE / "Mujoco"),
        "argv": ["./run.sh", "sim"],
        "axis_key": "mujoco",
    },
    "isaacsim_physx": {
        "cwd": str(_WORKSPACE / "IsaacLab"),
        "argv": ["./run.sh", "py", "sim2sim_motion_retargeting/h2_policy_runner/run_h2_policy.py",
                 "--headless", "--physics", "isaacsim_physx"],
        "axis_key": "isaacsim_physx",
    },
    "newton_mjwarp": {
        "cwd": str(_WORKSPACE / "IsaacLab"),
        "argv": ["./run.sh", "py", "sim2sim_motion_retargeting/h2_policy_runner/run_h2_policy.py",
                 "--headless", "--physics", "newton_mjwarp"],
        "axis_key": "newton_mjwarp",
    },
}


def support(ev, axis: str, backend: str) -> str:
    """How exactly ``backend`` can apply ``axis``.

    Args:
        ev: The loaded ``evaluation`` module.
        axis: A sweep axis name, e.g. ``torso_com_x``.
        backend: A key of :data:`BACKENDS`.

    Returns:
        ``"exact"``, ``"caveated"``, ``"unsupported"``, or ``"undeclared"`` when
        ``AXIS_SUPPORT`` has no column for that backend. ``undeclared`` is not a
        silent pass: the driver reports it so the gap gets filled rather than
        assumed away.
    """
    entry = ev.AXIS_SUPPORT.get(_support_name(axis))
    if entry is None:
        return "undeclared"
    return entry.get(BACKENDS[backend]["axis_key"], "undeclared")


def _support_name(axis: str) -> str:
    """Map a sweep axis to its AXIS_SUPPORT entry (the three CoM axes share one)."""
    return "torso_com_offset" if axis.startswith("torso_com_") else axis


def host_path(backend: str, out: str) -> str:
    """Resolve a runner-relative result path to one the driver can open.

    The driver's own working directory is unrelated to the backend's, so a path
    passed to the runner as relative has to be re-anchored before it is read
    back.

    Args:
        backend: A key of :data:`BACKENDS`.
        out: The path handed to the runner's ``--out``.

    Returns:
        The same file as seen from the driver.
    """
    return str(Path(BACKENDS[backend]["cwd"]) / out)


def command(backend: str, *, motion: str, seconds: float, out: str,
            perturbations: dict[str, float], profile: str = "strict-native",
            seed: int | None = None, push: bool = False) -> tuple[list[str], str]:
    """Return the argv and working directory for one rollout.

    Args:
        backend: A key of :data:`BACKENDS`.
        motion: Reference clip name.
        seconds: Episode length [s].
        out: Where the runner writes its canonical JSON result, relative to the
            backend's launch directory.
        perturbations: Axis name -> value. Range checking stays in the runner,
            which owns the training ranges.
        profile: ``strict-native`` or ``backend-recommended``.
        seed: Seeds the push schedule; the same seed replays the identical
            disturbance on every backend.
        push: Replay training's push event.

    Returns:
        ``(argv, cwd)``.
    """
    spec = BACKENDS[backend]
    argv = list(spec["argv"]) + [
        "--motion", motion, "--seconds", str(seconds),
        "--profile", profile, "--out", out,
    ]
    if seed is not None:
        argv += ["--seed", str(seed)]
    for axis, value in perturbations.items():
        argv += ["--perturb", f"{axis}={value}"]
    if push:
        argv.append("--push")
    return argv, spec["cwd"]
