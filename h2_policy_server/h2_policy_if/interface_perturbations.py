"""OOD_SIM2REAL perturbations that act between the policy and the simulator.

Latency, timing jitter and sensor noise are properties of the *interface*, not
of any one physics engine, so they belong in one place rather than being
re-implemented per runner. Both the MuJoCo and the Isaac Lab runner drive this
class, which means an ``action_latency=0.02`` run means the same thing on both
and a cross-backend comparison stays a comparison.

None of these axes were randomised during SONIC training, so every one is graded
``OOD_SIM2REAL`` and carries no swept range -- only the value the operator asked
for, recorded in the result. The runners refuse them without ``--allow-ood``.
"""

from __future__ import annotations

import collections
import math

import numpy as np

from . import evaluation as ev


class InterfaceLayer:
    """Applies interface-level perturbations to observations and commands.

    A neutral instance (no specs) is a pass-through: :meth:`observe` returns its
    inputs unchanged and :meth:`substeps` always returns the nominal decimation,
    so a strict baseline pays nothing for this object existing.

    Args:
        specs: Axis name -> value, restricted to axes whose ``layer`` is
            ``interface``. Unknown axes raise.
        control_dt: Control period [s]; latencies are quantised to it.
        decimation: Nominal physics steps per control step.
        seed: Seeds the noise and jitter draws. Required when any stochastic
            axis is active, so a run can be replayed on another backend.

    Raises:
        ValueError: On an unknown axis, an axis that is not interface-level, or
            a stochastic axis without a seed.
    """

    #: Axes whose value is a noise standard deviation rather than a fixed offset.
    STOCHASTIC = ("control_period_jitter", "joint_pos_noise", "joint_vel_noise",
                  "imu_orientation_noise", "imu_angular_velocity_noise")

    def __init__(self, specs: dict[str, float] | None, control_dt: float,
                 decimation: int, seed: int | None = None):
        self.specs = dict(specs or {})
        self.control_dt = float(control_dt)
        self.decimation = int(decimation)
        for axis in self.specs:
            entry = ev.OOD_PERTURBATIONS.get(axis)
            if entry is None:
                raise ValueError(f"unknown interface axis {axis!r}")
            if entry.layer != "interface":
                raise ValueError(f"{axis} is a {entry.layer}-level axis, not an interface one")
        if any(a in self.specs and self.specs[a] for a in self.STOCHASTIC) and seed is None:
            raise ValueError(
                "a stochastic interface axis needs a seed, so the same disturbance "
                "can be replayed on another backend"
            )
        self.rng = np.random.default_rng(seed)

        # Latency is realised as a fixed-length queue of past values. Quantising
        # to whole control steps is exact for the loop and is reported back, so a
        # requested 0.015 s at 50 Hz is recorded as the 0.02 s actually applied.
        self.obs_delay = self._steps("observation_latency")
        self.act_delay = self._steps("action_latency")
        self._obs_q: collections.deque = collections.deque(maxlen=self.obs_delay + 1)
        self._act_q: collections.deque = collections.deque(maxlen=self.act_delay + 1)
        self.motor_strength = float(self.specs.get("motor_strength_error", 1.0))
        self.jitter = float(self.specs.get("control_period_jitter", 0.0))

    def _steps(self, axis: str) -> int:
        """Whole control steps of delay for a latency axis.

        Rounds halves up rather than to even: Python's ``round`` would turn a
        requested 10 ms at 50 Hz into no delay at all, which looks like the axis
        was ignored. The realised value is reported by :meth:`as_dict` either way.
        """
        return int(math.floor(float(self.specs.get(axis, 0.0)) / self.control_dt + 0.5))

    @property
    def active(self) -> bool:
        """Whether anything at all is perturbed."""
        return bool(self.specs)

    def observe(self, base_quat: np.ndarray, joint_pos: np.ndarray,
                joint_vel: np.ndarray, base_ang_vel: np.ndarray) -> tuple:
        """Return the (possibly delayed and noisy) state the policy should see.

        Noise is drawn on the true state and then queued, so a delayed
        observation carries the noise it had when it was measured -- which is
        what a real sensor pipeline does.

        Args:
            base_quat: Base orientation, WXYZ, shape [4].
            joint_pos: Joint positions [rad], shape [31].
            joint_vel: Joint velocities [rad/s], shape [31].
            base_ang_vel: Base angular velocity in the base frame [rad/s], shape [3].

        Returns:
            The same four arrays, perturbed.
        """
        quat = np.asarray(base_quat, dtype=np.float64).copy()
        pos = np.asarray(joint_pos, dtype=np.float64).copy()
        vel = np.asarray(joint_vel, dtype=np.float64).copy()
        ang = np.asarray(base_ang_vel, dtype=np.float64).copy()

        if self.specs.get("joint_pos_noise"):
            pos += self.rng.normal(0.0, self.specs["joint_pos_noise"], pos.shape)
        if self.specs.get("joint_vel_noise"):
            vel += self.rng.normal(0.0, self.specs["joint_vel_noise"], vel.shape)
        if self.specs.get("imu_angular_velocity_noise"):
            ang += self.rng.normal(0.0, self.specs["imu_angular_velocity_noise"], ang.shape)
        if self.specs.get("imu_orientation_noise"):
            quat = _perturb_quat(quat, self.rng.normal(0.0, self.specs["imu_orientation_noise"], 3))

        if self.obs_delay == 0:
            return quat, pos, vel, ang
        self._obs_q.append((quat, pos, vel, ang))
        # Before the queue fills, the oldest sample is the best available stand-in
        # for a state the policy has not been shown yet.
        return self._obs_q[0]

    def command(self, target: np.ndarray, default_pose: np.ndarray) -> np.ndarray:
        """Return the joint position target the simulator should actually track.

        Args:
            target: Target the policy asked for [rad], shape [31].
            default_pose: The pose the action is an offset from [rad], shape [31].
                Motor strength scales the displacement from this pose, not the
                absolute angle, so a "weak" actuator falls back toward the
                default rather than toward zero.

        Returns:
            The possibly delayed and rescaled target.
        """
        cmd = np.asarray(target, dtype=np.float64).copy()
        if self.motor_strength != 1.0:
            base = np.asarray(default_pose, dtype=np.float64)
            cmd = base + (cmd - base) * self.motor_strength
        if self.act_delay == 0:
            return cmd
        self._act_q.append(cmd)
        return self._act_q[0]

    def substeps(self) -> int:
        """Physics steps to run for this control step.

        Jitter varies how long a control period lasts; it never changes the
        physics timestep, which stays at the trained 0.005 s. At least one step
        always runs, so time cannot stand still.
        """
        if not self.jitter:
            return self.decimation
        physics_dt = self.control_dt / self.decimation
        n = round(self.rng.normal(self.decimation, self.jitter / physics_dt))
        return max(1, int(n))

    def as_dict(self) -> dict:
        """Return the record for the result file, including what was quantised."""
        if not self.specs:
            return {}
        record = {
            axis: {
                "value": value,
                "unit": ev.OOD_PERTURBATIONS[axis].unit,
                "grade": "OOD_SIM2REAL",
                "layer": "interface",
                "note": ev.OOD_PERTURBATIONS[axis].description,
            }
            for axis, value in self.specs.items()
        }
        for axis, steps in (("observation_latency", self.obs_delay),
                            ("action_latency", self.act_delay)):
            if axis in record:
                record[axis]["applied_control_steps"] = steps
                record[axis]["applied_s"] = round(steps * self.control_dt, 6)
        return record


def _perturb_quat(quat_wxyz: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    """Rotate a WXYZ quaternion by a small rotation vector [rad].

    Args:
        quat_wxyz: Unit quaternion, WXYZ, shape [4].
        rotvec: Axis-angle perturbation [rad], shape [3].

    Returns:
        The perturbed unit quaternion, WXYZ.
    """
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return quat_wxyz
    axis = rotvec / angle
    half = 0.5 * angle
    dq = np.concatenate(([np.cos(half)], axis * np.sin(half)))
    w1, x1, y1, z1 = dq
    w2, x2, y2, z2 = quat_wxyz
    out = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    return out / np.linalg.norm(out)
