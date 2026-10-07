"""Strict-evaluation scaffolding shared by every backend runner.

The question this evaluation answers is *not* "can the robot be made to stay
up". It is "how does the already-trained SONIC policy behave when nothing about
it is touched and only the simulator changes". So this module exists to make two
things impossible to get wrong:

* **Provenance.** Every physical number a runner feeds a simulator is declared
  here with the file and line it came from, and printed at the start of a run.
  A value with no training source is labelled as such rather than quietly used.
* **Profile separation.** Numerical solver tuning that has no training
  provenance is confined to a named profile, so a `strict-native` result can
  never be silently replaced by a tuned one.

Nothing here changes policy semantics. It records, classifies and reports.

Three provenance grades are used throughout, and results must never mix them:

``TRAIN_DIST``
    A range the SONIC training actually randomised over. Sweeping inside it
    measures robustness the policy was trained for.
``OOD_SIM2REAL``
    Not in training, but physically real on hardware (latency, sensor noise,
    motor-strength error). Always reported separately.
``NUMERICAL``
    Solver settings, not robot properties. A difference here is a simulator
    artefact, not a robot behaviour.

One caveat is baked in on purpose. The shipped checkpoint is real step 66,000 of
a run that resumed from step 31,500 of an earlier run, and neither run's
resolved hydra config is available on this machine. The physics-related files
have not changed since 2026-09-07 10:15 while the visible run started
2026-09-14 00:20, which makes the values below very likely correct for the whole
66k -- but "likely" is not "verified". Everything here is therefore
:data:`CURRENT_CODE_PROVENANCE`, never ``TRAIN_RUN_VERIFIED``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import subprocess
import time
from typing import Any

PROVENANCE_GRADE = "CURRENT_CODE_PROVENANCE"
"""Values are read from the current wbc_h12 checkout, not from the training run's
own resolved config, which lives only on the training server. See module docstring."""

WBC_H12_COMMIT_EXPECTED = "e279627ba8a2489bb46dc152a222f8c8e8ecfb27"
"""The commit the provenance below was read from; runners re-check this at start-up."""


# ---------------------------------------------------------------------------
# Training physics, with sources. Paths are relative to the wbc_h12 checkout.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Sourced:
    """One physical value together with where it came from.

    Args:
        value: The value a runner should feed the simulator.
        source: ``path:line`` inside wbc_h12, or an explicit note when there is
            no training source.
        grade: One of ``TRAIN_DIST``, ``OOD_SIM2REAL``, ``NUMERICAL``, or
            ``NOMINAL`` for a single training value that was not randomised.
        note: Anything a reader needs in order not to misuse the value.
    """

    value: Any
    source: str
    grade: str = "NOMINAL"
    note: str = ""


TERRAIN_MATERIAL = {
    "static_friction": Sourced(1.0, "gear_sonic/envs/manager_env/modular_tracking_env_cfg.py:345"),
    "dynamic_friction": Sourced(1.0, "gear_sonic/envs/manager_env/modular_tracking_env_cfg.py:346"),
    "restitution": Sourced(
        0.0,
        "gear_sonic/envs/manager_env/modular_tracking_env_cfg.py:342-347",
        note="not set on the terrain material, so Isaac Lab's RigidBodyMaterialCfg default applies",
    ),
    "friction_combine_mode": Sourced(
        "multiply", "gear_sonic/envs/manager_env/modular_tracking_env_cfg.py:343"
    ),
    "restitution_combine_mode": Sourced(
        "multiply", "gear_sonic/envs/manager_env/modular_tracking_env_cfg.py:344"
    ),
}
"""Terrain material for the trimesh branch, which ``sonic_h2_edu.yaml`` selects.

The plane branch (:339-334 of the same file) carries identical friction, so the
numbers do not depend on which terrain a given evaluation uses.
"""

TERRAIN_TYPE = Sourced(
    "trimesh",
    "gear_sonic/config/exp/manager/universal_token/all_modes/sonic_h2_edu.yaml (config.terrain_type)",
    note=(
        "resolves to isaaclab.terrains.config.rough.ROUGH_TERRAINS_CFG with "
        "max_init_terrain_level=10. That suite is NOT near-flat: stairs 0.05-0.23 m, "
        "random grid boxes 0.05-0.20 m, uniform noise 0.02-0.10 m, slopes 0-0.4. "
        "The 0.005 in the config is vertical_scale, the heightfield quantisation "
        "step, not the roughness amplitude. Evaluating on flat ground is therefore "
        "an EASIER condition than training, not an equivalent one."
    ),
)

CONTROL = {
    "physics_dt": Sourced(0.005, "gear_sonic/envs/manager_env/modular_tracking_env_cfg.py:969"),
    "decimation": Sourced(4, "gear_sonic/envs/manager_env/modular_tracking_env_cfg.py:965"),
    "control_dt": Sourced(0.02, "physics_dt * decimation"),
}

INITIAL_STATE = {
    "pelvis_height": Sourced(1.04, "gear_sonic/envs/manager_env/robots/h2.py:234"),
    "joint_pos": Sourced(
        "h2_joints.DEFAULT_ANGLES_MUJOCO", "gear_sonic/envs/manager_env/robots/h2.py:235-244"
    ),
    "joint_vel": Sourced(0.0, "gear_sonic/envs/manager_env/robots/h2.py:245"),
    "settle": Sourced(
        0.0,
        "NO TRAINING SOURCE",
        note=(
            "A pre-policy PD hold does not exist in the training env. It was added "
            "during bring-up and is forced to 0 under any strict profile. Left "
            "available only as a debug knob."
        ),
    ),
}

ACTUATORS = {
    "stiffness": Sourced("h2_joints.KP_MUJOCO", "gear_sonic/envs/manager_env/robots/h2.py:7-23,268-287"),
    "damping": Sourced("h2_joints.KD_MUJOCO", "gear_sonic/envs/manager_env/robots/h2.py:7-23,274-287"),
    "armature": Sourced("h2_joints.ARMATURE_MUJOCO", "gear_sonic/envs/manager_env/robots/h2.py actuator armature="),
    "effort_limit": Sourced(
        "h2_joints.EFFORT_LIMIT_MUJOCO",
        "gear_sonic/envs/manager_env/robots/h2_edu.py effort_limit_sim overrides",
        note="production H2 EDU torques, e.g. ankle_roll 19.0 N m rather than H1031's 150.0",
    ),
    "action_scale": Sourced(
        "h2_joints.ACTION_SCALE_MUJOCO",
        "gear_sonic/envs/manager_env/robots/h2.py:379-390 via h2_edu.py:61",
        note=(
            "0.25 * effort / stiffness computed from H1031's effort limits, because "
            "h2_edu.py aliases H2_EDU_ACTION_SCALE = H2_ACTION_SCALE. This is the "
            "action semantics the policy was trained with and must NOT be recomputed "
            "from the real H2 torques."
        ),
    ),
}

# ---------------------------------------------------------------------------
# What training randomised. Sweep domains, not values to apply to a nominal run.
# ---------------------------------------------------------------------------

TRAIN_RANDOMIZATION = {
    "robot_material.static_friction": Sourced(
        (0.3, 1.6), "gear_sonic/config/manager_env/events/terms/physics_material.yaml", "TRAIN_DIST",
        note="startup mode, robot bodies '.*', 64 buckets, combined with the terrain by multiply",
    ),
    "robot_material.dynamic_friction": Sourced(
        (0.3, 1.2), "gear_sonic/config/manager_env/events/terms/physics_material.yaml", "TRAIN_DIST"
    ),
    "robot_material.restitution": Sourced(
        (0.0, 0.5), "gear_sonic/config/manager_env/events/terms/physics_material.yaml", "TRAIN_DIST"
    ),
    "body_mass_scale": Sourced(
        (0.8, 2.5), "gear_sonic/config/manager_env/events/tracking/level0_4.yaml", "TRAIN_DIST",
        note="startup, only '.*wrist_yaw.*|torso_link'. The 0.8-1.2 all-bodies term in "
             "terms/randomize_rigid_body_mass.yaml is overridden by level0_4",
    ),
    "torso_com_offset_m": Sourced(
        {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        "gear_sonic/config/manager_env/events/terms/base_com.yaml", "TRAIN_DIST",
    ),
    "joint_default_pos_offset_rad": Sourced(
        (-0.01, 0.01), "gear_sonic/config/manager_env/events/terms/add_joint_default_pos.yaml",
        "TRAIN_DIST", note="startup, added to every joint's default position",
    ),
    "push_interval_s": Sourced(
        (4.0, 6.0), "gear_sonic/config/manager_env/events/tracking/level0_4.yaml", "TRAIN_DIST"
    ),
    "push_velocity": Sourced(
        {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2),
         "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)},
        "gear_sonic/config/manager_env/events/tracking/level0_4.yaml", "TRAIN_DIST",
        note="push_by_setting_velocity, i.e. the base velocity is SET, not a force impulse",
    ),
}

# ---------------------------------------------------------------------------
# Known simulator non-equivalences. Recorded, never patched: inserting a clamp a
# simulator does not natively have would be a simulator modification, not a
# translation, and would hide exactly the kind of gap this evaluation exists to
# surface. Every result file carries this block so a reader of the JSON alone
# knows which cross-backend comparisons are exact and which are approximate.
# ---------------------------------------------------------------------------

SIMULATOR_ASYMMETRIES = {
    "joint_velocity_limit": {
        "affects": ["mujoco"],
        "training_value": "velocity_limit_sim, 20 rad/s on the legs",
        "source": "gear_sonic/envs/manager_env/robots/h2_edu.py (_EDU_LEG_VEL) and h2.py velocity_limit_sim",
        "physx": "enforced by the articulation; measured |qd|max saturates at 20.01 rad/s",
        "mujoco": "MuJoCo `motor` actuators have no velocity limit, so the same "
                  "policy reaches ~41 rad/s on the same clip",
        "resolution": "NOT patched. Compare joint-velocity metrics across these "
                      "backends only with this in mind; pelvis height, torque and "
                      "survival remain comparable.",
        "status": "open",
    },
    "friction_triple": {
        "affects": ["mujoco"],
        "training_value": "static_friction = dynamic_friction = 1.0, combine multiply",
        "source": "gear_sonic/envs/manager_env/modular_tracking_env_cfg.py:343-346",
        "physx": "separate static and dynamic coefficients",
        "mujoco": "one sliding coefficient plus torsional and rolling terms. The "
                  "sliding term maps exactly because training sets static = dynamic; "
                  "MuJoCo's torsional 0.005 and rolling 0.0001 defaults have no "
                  "counterpart in the training material",
        "resolution": "sliding term mapped exactly; torsional/rolling left at MuJoCo "
                      "defaults and declared an approximation",
        "status": "approximated",
    },
    "torque_saturation": {
        "affects": [],
        "training_value": "effort_limit_sim per joint, e.g. 360 N m at the knee",
        "source": "gear_sonic/envs/manager_env/robots/h2_edu.py effort_limit_sim",
        "physx": "ImplicitActuator clips the PD term to the limit before the solver",
        "mujoco": "ctrlrange clips the control signal, which is joint torque for a "
                  "`motor` actuator with gear=1, also before integration",
        "resolution": "verified equivalent by tools/torque_semantics.py: the clamp "
                      "first bites at the same 360 N m on both sides",
        "status": "verified_equivalent",
    },
    "terrain": {
        "affects": ["isaacsim_physx", "newton_mjwarp", "mujoco"],
        "training_value": "trimesh / ROUGH_TERRAINS_CFG",
        "source": "sonic_h2_edu.yaml config.terrain_type",
        "physx": "these evaluations run flat ground",
        "mujoco": "these evaluations run flat ground",
        "resolution": "flat ground is an EASIER condition than training, not an "
                      "equivalent one. See TERRAIN_TYPE.note for the actual ranges.",
        "status": "open",
    },
    "runtime_com_change": {
        "affects": ["newton_mjwarp", "newton_vbd"],
        "training_value": "randomize_rigid_body_com on torso_link, +/-0.05 m",
        "source": "isaaclab/envs/mdp/events.py:922-933 (randomize_rigid_body_com docstring)",
        "physx": "set_coms_index takes the full CoM pose and the change is honoured",
        "mujoco": "the offset is written into the MJCF before the model is compiled, "
                  "so the mass matrix is built from the perturbed CoM",
        "resolution": "NOT patched. Isaac Lab states that on Newton, "
                      "notify_model_changed(BODY_INERTIAL_PROPERTIES) does not fully "
                      "recompute the mass matrix after a runtime body_ipos change. A "
                      "torso_com_* sweep is therefore reported on PhysX and MuJoCo; on "
                      "Newton it is run only with this caveat attached to the result.",
        "status": "open",
    },
}


# ---------------------------------------------------------------------------
# Which backend can apply which TRAIN_DIST axis, and how exactly. A sweep must
# never quietly approximate: an axis that a simulator cannot express natively is
# marked unsupported there and simply not run, rather than mapped onto the
# nearest-looking parameter.
# ---------------------------------------------------------------------------

AXIS_SUPPORT = {
    "robot_friction": {
        "train_key": "robot_material.static_friction",
        "mujoco": "exact",
        "isaacsim_physx": "exact",
        "newton_mjwarp": "exact",
        "note": "MuJoCo geom_friction[:,0] and PhysX static_friction both scale the "
                "Coulomb sliding term on the robot's own geoms. On Newton this is the "
                "single friction coefficient, which is the same quantity.",
    },
    "robot_dynamic_friction": {
        "train_key": "robot_material.dynamic_friction",
        "mujoco": "unsupported",
        "isaacsim_physx": "exact",
        "newton_mjwarp": "unsupported",
        "note": "MuJoCo's contact model carries ONE sliding coefficient, so static "
                "and dynamic friction cannot be set independently. Training "
                "randomised them separately, so this axis is measured on PhysX only "
                "rather than folded into the single MuJoCo coefficient. Newton has "
                "the same limitation and states it explicitly: "
                "isaaclab/envs/mdp/events.py:577-582, \"Newton uses a single friction "
                "coefficient, so dynamic_friction_range ... [is] ignored\".",
    },
    "robot_restitution": {
        "train_key": "robot_material.restitution",
        "mujoco": "unsupported",
        "isaacsim_physx": "exact",
        "newton_mjwarp": "exact",
        "note": "PhysX has a restitution coefficient; MuJoCo expresses bounce through "
                "solref's (timeconst, dampratio), which is a different parameterisation "
                "with no exact inverse. Mapping one onto the other would make a "
                "MuJoCo restitution number mean something other than the training one.",
    },
    "body_mass_scale": {
        "train_key": "body_mass_scale",
        "mujoco": "exact",
        "isaacsim_physx": "exact",
        "newton_mjwarp": "exact",
        "note": "mass and inertia of *_wrist_yaw_link and torso_link scaled together.",
    },
    "torso_com_offset": {
        "train_key": "torso_com_offset_m",
        "mujoco": "exact",
        "isaacsim_physx": "exact",
        "newton_mjwarp": "caveated",
        "note": "torso_link centre of mass moved in the body frame. MuJoCo bakes the "
                "offset into the model before compilation and PhysX accepts the full "
                "CoM pose at runtime, so both are exact; on Newton the change is "
                "applied but the mass matrix is not fully rebuilt -- see "
                "SIMULATOR_ASYMMETRIES['runtime_com_change'].",
    },
    "joint_default_offset": {
        "train_key": "joint_default_pos_offset_rad",
        "mujoco": "exact",
        "isaacsim_physx": "exact",
        "newton_mjwarp": "exact",
        "note": "Training's randomize_joint_default_pos shifts the articulation's "
                "default joint positions, which Isaac Lab uses BOTH for the reset pose "
                "AND as the action offset. Reproducing it therefore requires the "
                "simulator and the policy server to apply the same offset -- the "
                "server takes it on reset so the two cannot drift apart.",
    },
    "push": {
        "train_key": "push_velocity",
        "mujoco": "exact",
        "isaacsim_physx": "exact",
        "newton_mjwarp": "exact",
        "note": "push_by_setting_velocity SETS the base velocity at intervals; it is "
                "not a force impulse. The schedule is seeded so every backend replays "
                "the identical disturbance sequence.",
    },
}


class PushSchedule:
    """Seeded reproduction of training's ``push_robot`` event.

    ``gear_sonic.envs.manager_env.mdp:push_by_setting_velocity`` **sets** the base
    velocity every ``interval_range_s`` seconds, sampling each component from
    ``velocity_range``. Generating the whole schedule up front from a seed lets
    PhysX, MJWarp and MuJoCo replay the identical disturbance, which is what
    makes a cross-backend robustness comparison meaningful rather than three
    different random experiments.

    Args:
        seed: RNG seed; recorded in the result so a run can be replayed.
        duration_s: Episode length to cover [s].
        interval_range_s: Seconds between pushes. Defaults to the training range.
        velocity_range: Per-component ranges. Defaults to the training ranges.
    """

    ORDER = ("x", "y", "z", "roll", "pitch", "yaw")

    def __init__(
        self,
        seed: int,
        duration_s: float,
        interval_range_s: tuple[float, float] | None = None,
        velocity_range: dict | None = None,
    ) -> None:
        import random  # noqa: PLC0415  (only this class needs it)

        self.seed = seed
        self.interval_range_s = interval_range_s or TRAIN_RANDOMIZATION["push_interval_s"].value
        self.velocity_range = velocity_range or TRAIN_RANDOMIZATION["push_velocity"].value
        rng = random.Random(seed)
        self.events: list[tuple[float, list[float]]] = []
        t = rng.uniform(*self.interval_range_s)
        while t < duration_s:
            vel = [rng.uniform(*self.velocity_range[k]) for k in self.ORDER]
            self.events.append((round(t, 4), [round(v, 5) for v in vel]))
            t += rng.uniform(*self.interval_range_s)

    def due(self, t_prev: float, t_now: float) -> list[float] | None:
        """Return the velocity to set if a push falls in ``(t_prev, t_now]``."""
        for when, vel in self.events:
            if t_prev < when <= t_now:
                return vel
        return None

    def as_dict(self) -> dict:
        """Serialisable record for the result metadata."""
        return {
            "seed": self.seed,
            "interval_range_s": list(self.interval_range_s),
            "velocity_range": {k: list(v) for k, v in self.velocity_range.items()},
            "events": self.events,
            "grade": "TRAIN_DIST",
            "source": TRAIN_RANDOMIZATION["push_velocity"].source,
        }


OOD_AXES = (
    "action_latency", "observation_latency", "control_period_jitter",
    "motor_strength_error", "pd_gain_error",
    "joint_pos_noise", "joint_vel_noise", "imu_orientation_noise", "imu_angular_velocity_noise",
)
"""Perturbations training did NOT apply. Any result using one is OOD_SIM2REAL."""


# ---------------------------------------------------------------------------
# OOD_SIM2REAL axes the runners can apply.
#
# These are real sim2real effects, but SONIC training randomised none of them,
# so there is no defensible range to sweep -- only a value the operator chooses
# and the result records. Every entry therefore carries a unit and a neutral
# value, and NO range: inventing one would dress a guess up as provenance.
# Both runners refuse them unless --allow-ood is passed, which keeps a strict
# baseline from ever silently containing one.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class OodAxis:
    """One OOD_SIM2REAL perturbation axis.

    Args:
        unit: SI unit of the value, or ``"x"`` for a dimensionless multiplier.
        neutral: The value at which the axis has no effect.
        layer: ``interface`` when it acts between policy and simulator and is
            therefore backend-independent, ``simulator`` when it changes the
            model.
        description: What the axis does, in one line.
        support: Backend -> ``exact`` / ``caveated`` / ``unsupported``.
        note: Why a backend cannot express it, when one cannot.
    """

    unit: str
    neutral: float
    layer: str
    description: str
    support: dict
    note: str = ""


_ALL_EXACT = {"mujoco": "exact", "isaacsim_physx": "exact", "newton_mjwarp": "exact"}

OOD_PERTURBATIONS = {
    # -- interface layer: identical code on every backend, so identical semantics
    "action_latency": OodAxis(
        unit="s", neutral=0.0, layer="interface",
        description="Delay between the policy emitting a target and the simulator receiving it.",
        support=dict(_ALL_EXACT),
    ),
    "observation_latency": OodAxis(
        unit="s", neutral=0.0, layer="interface",
        description="Age of the state the policy sees, relative to the simulator's true state.",
        support=dict(_ALL_EXACT),
    ),
    "control_period_jitter": OodAxis(
        unit="s", neutral=0.0, layer="interface",
        description=("Standard deviation of the control period. Realised as a varying number "
                     "of physics substeps per control step, so the physics dt itself is "
                     "never changed."),
        support=dict(_ALL_EXACT),
    ),
    "joint_pos_noise": OodAxis(
        unit="rad", neutral=0.0, layer="interface",
        description="Gaussian noise added to the joint positions the policy observes.",
        support=dict(_ALL_EXACT),
    ),
    "joint_vel_noise": OodAxis(
        unit="rad/s", neutral=0.0, layer="interface",
        description="Gaussian noise added to the joint velocities the policy observes.",
        support=dict(_ALL_EXACT),
    ),
    "imu_orientation_noise": OodAxis(
        unit="rad", neutral=0.0, layer="interface",
        description="Random small rotation applied to the base orientation the policy observes.",
        support=dict(_ALL_EXACT),
    ),
    "imu_angular_velocity_noise": OodAxis(
        unit="rad/s", neutral=0.0, layer="interface",
        description="Gaussian noise added to the base angular velocity the policy observes.",
        support=dict(_ALL_EXACT),
    ),
    "motor_strength_error": OodAxis(
        unit="x", neutral=1.0, layer="interface",
        description=("Multiplier on the commanded joint displacement from the default pose, "
                     "standing in for an actuator that under- or over-delivers."),
        support=dict(_ALL_EXACT),
    ),
    # -- simulator layer
    "kp_scale": OodAxis(
        unit="x", neutral=1.0, layer="simulator",
        description="Multiplier on every joint's position gain.",
        support=dict(_ALL_EXACT),
    ),
    "kd_scale": OodAxis(
        unit="x", neutral=1.0, layer="simulator",
        description="Multiplier on every joint's damping gain.",
        support=dict(_ALL_EXACT),
    ),
    "armature_scale": OodAxis(
        unit="x", neutral=1.0, layer="simulator",
        description="Multiplier on every joint's rotor armature.",
        support=dict(_ALL_EXACT),
    ),
    "effort_limit_scale": OodAxis(
        unit="x", neutral=1.0, layer="simulator",
        description="Multiplier on every joint's torque limit.",
        support=dict(_ALL_EXACT),
    ),
    "velocity_limit_scale": OodAxis(
        unit="x", neutral=1.0, layer="simulator",
        description="Multiplier on every joint's velocity limit.",
        support={"mujoco": "unsupported", "isaacsim_physx": "exact", "newton_mjwarp": "exact"},
        note=("MuJoCo `motor` actuators carry no velocity limit at all, which is the "
              "already-recorded joint_velocity_limit asymmetry. There is nothing to scale, "
              "so the axis is refused rather than emulated."),
    ),
    "body_inertia_scale": OodAxis(
        unit="x", neutral=1.0, layer="simulator",
        description="Multiplier on every body's inertia tensor, leaving mass unchanged.",
        support=dict(_ALL_EXACT),
    ),
    "init_height_offset": OodAxis(
        unit="m", neutral=0.0, layer="simulator",
        description="Offset added to the spawn height of the pelvis.",
        support=dict(_ALL_EXACT),
    ),
    "init_joint_pos_noise": OodAxis(
        unit="rad", neutral=0.0, layer="simulator",
        description="Gaussian noise on the spawn joint positions.",
        support=dict(_ALL_EXACT),
    ),
    "init_root_vel_noise": OodAxis(
        unit="m/s", neutral=0.0, layer="simulator",
        description="Gaussian noise on the spawn root linear and angular velocity.",
        support=dict(_ALL_EXACT),
    ),
}


def axis_grade(axis: str) -> str:
    """Return ``TRAIN_DIST`` or ``OOD_SIM2REAL`` for a perturbation axis.

    Raises:
        KeyError: When the axis is not registered anywhere.
    """
    if axis in OOD_PERTURBATIONS:
        return "OOD_SIM2REAL"
    if _support_key(axis) in AXIS_SUPPORT:
        return "TRAIN_DIST"
    raise KeyError(axis)


def _support_key(axis: str) -> str:
    """Map a perturbation axis to its AXIS_SUPPORT key (the CoM axes share one)."""
    return "torso_com_offset" if axis.startswith("torso_com_") else axis


def axis_support(axis: str, backend: str) -> str:
    """How exactly ``backend`` can apply ``axis``.

    Returns:
        ``exact``, ``caveated``, ``unsupported``, or ``undeclared`` when the
        table has no column for that backend. ``undeclared`` is never treated as
        a pass: callers report it so the gap is filled rather than assumed away.
    """
    if axis in OOD_PERTURBATIONS:
        return OOD_PERTURBATIONS[axis].support.get(backend, "undeclared")
    entry = AXIS_SUPPORT.get(_support_key(axis))
    return "undeclared" if entry is None else entry.get(backend, "undeclared")


def axis_note(axis: str) -> str:
    """Return the recorded reason a backend can or cannot express the axis."""
    if axis in OOD_PERTURBATIONS:
        return OOD_PERTURBATIONS[axis].note or OOD_PERTURBATIONS[axis].description
    entry = AXIS_SUPPORT.get(_support_key(axis))
    return entry.get("note", "") if entry else ""


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Profile:
    """An evaluation profile.

    Args:
        name: Identifier that appears in every result file.
        strict: Whether only training-sourced physical parameters may be applied.
        allow_solver_tuning: Whether a backend may apply non-default numerical
            solver settings. False means the backend's own defaults are used.
        description: One line for the banner.
    """

    name: str
    strict: bool
    allow_solver_tuning: bool
    description: str


STRICT_NATIVE = Profile(
    name="strict-native",
    strict=True,
    allow_solver_tuning=False,
    description=(
        "Training-sourced physical parameters only; the backend's own default "
        "numerical settings; no settle, no post-processing, no stabiliser."
    ),
)

BACKEND_RECOMMENDED = Profile(
    name="backend-recommended",
    strict=True,
    allow_solver_tuning=True,
    description=(
        "Same policy and same physical parameters as strict-native, plus the "
        "solver settings the backend's own documentation or official humanoid "
        "examples recommend. Every non-default value is recorded."
    ),
)

PROFILES = {p.name: p for p in (STRICT_NATIVE, BACKEND_RECOMMENDED)}


# ---------------------------------------------------------------------------
# Provenance capture
# ---------------------------------------------------------------------------


def sha256_file(path: str) -> str:
    """Return the hex sha256 of a file, or ``"<missing>"`` when it is absent."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return "<missing>"


def _head_from_dotgit(repo: str) -> str:
    """Read HEAD's commit straight out of ``.git``, without the git binary.

    The simulation containers are deliberately slim and carry no git, but the
    commit still has to appear in every result file. This resolves the symbolic
    ref by hand and covers packed refs.

    Args:
        repo: Path to the checkout.

    Returns:
        The 40-character sha, or a bracketed reason.
    """
    git_dir = os.path.join(repo, ".git")
    try:
        with open(os.path.join(git_dir, "HEAD")) as fh:
            head = fh.read().strip()
    except OSError:
        return "<not a git checkout>"
    if not head.startswith("ref:"):
        return head
    ref = head[4:].strip()
    try:
        with open(os.path.join(git_dir, ref)) as fh:
            return fh.read().strip()
    except OSError:
        pass
    try:
        with open(os.path.join(git_dir, "packed-refs")) as fh:
            for line in fh:
                if line.rstrip().endswith(" " + ref):
                    return line.split()[0]
    except OSError:
        pass
    return "<ref unresolved>"


def git_head(repo: str) -> str:
    """Return the checkout's HEAD, with its clean/dirty state when checkable.

    Returns ``<sha>`` when the tree is provably clean, ``<sha>-DIRTY`` when it is
    not, and ``<sha>-UNVERIFIED`` when the commit is known but no git binary is
    available to check the tree. The last case is normal inside the slim
    simulation containers, where wbc_h12 is a read-only mount and therefore
    cannot be dirtied from inside; the host-side ``run.sh verify`` and
    ``bootstrap.sh --check`` cover the tree state.
    """
    try:
        sha = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        if sha.returncode != 0:
            return _head_from_dotgit(repo)
        head = sha.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", repo, "status", "--porcelain", "--ignored"],
            capture_output=True, text=True, timeout=60,
        )
        return head if not dirty.stdout.strip() else f"{head}-DIRTY"
    except (OSError, subprocess.SubprocessError):
        head = _head_from_dotgit(repo)
        return head if head.startswith("<") else f"{head}-UNVERIFIED"


def collect_provenance(
    wbc_root: str, backend: str, backend_version: str, profile: Profile,
    motion: str, motion_path: str | None, seed: int | None,
    solver_overrides: dict[str, Any] | None = None,
) -> dict:
    """Assemble the provenance block that every run prints and stores.

    Args:
        wbc_root: Path to the read-only wbc_h12 checkout.
        backend: Physics backend identifier, e.g. ``isaacsim_physx``.
        backend_version: Version string for that backend.
        profile: The evaluation profile in force.
        motion: Reference clip name.
        motion_path: Absolute path of the clip, when the runner knows it.
        seed: Random seed, or None when the run is fully deterministic.
        solver_overrides: Non-default numerical settings actually applied.

    Returns:
        A JSON-serialisable dict.
    """
    models = os.path.join(wbc_root, "h2_tools", "models")
    enc = os.path.join(models, "model_step_066000_encoder.onnx")
    dec = os.path.join(models, "model_step_066000_decoder.onnx")
    head = git_head(wbc_root)
    return {
        "provenance_grade": PROVENANCE_GRADE,
        "wbc_h12": {
            "root": wbc_root,
            "git_head": head,
            "matches_audited_commit": head.split("-")[0] == WBC_H12_COMMIT_EXPECTED,
            "tree_state": ("clean" if head == WBC_H12_COMMIT_EXPECTED
                           else "dirty" if head.endswith("-DIRTY")
                           else "unverified" if head.endswith("-UNVERIFIED") else "unknown"),
            "audited_commit": WBC_H12_COMMIT_EXPECTED,
        },
        "policy": {
            "encoder": os.path.basename(enc),
            "encoder_sha256": sha256_file(enc),
            "decoder": os.path.basename(dec),
            "decoder_sha256": sha256_file(dec),
        },
        "motion": {"name": motion, "path": motion_path,
                   "sha256": sha256_file(motion_path) if motion_path else None},
        "backend": {"name": backend, "version": backend_version},
        "profile": {"name": profile.name, "strict": profile.strict,
                    "allow_solver_tuning": profile.allow_solver_tuning,
                    "description": profile.description},
        "seed": seed,
        "sources": {
            "terrain_material": {k: dataclasses.asdict(v) for k, v in TERRAIN_MATERIAL.items()},
            "terrain_type": dataclasses.asdict(TERRAIN_TYPE),
            "control": {k: dataclasses.asdict(v) for k, v in CONTROL.items()},
            "initial_state": {k: dataclasses.asdict(v) for k, v in INITIAL_STATE.items()},
            "actuators": {k: dataclasses.asdict(v) for k, v in ACTUATORS.items()},
        },
        "solver_overrides": solver_overrides or {},
        "simulator_asymmetries": SIMULATOR_ASYMMETRIES,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def print_banner(prov: dict) -> None:
    """Print the provenance block a strict run must show before it starts."""
    p, w, pol, prof = prov["profile"], prov["wbc_h12"], prov["policy"], prov["profile"]
    print("=" * 78)
    print(f"  EVALUATION PROFILE : {p['name']}   strict={p['strict']}  "
          f"solver_tuning={'allowed' if p['allow_solver_tuning'] else 'DISABLED'}")
    print(f"                       {prof['description']}")
    print(f"  provenance grade   : {prov['provenance_grade']}")
    print(f"  wbc_h12 HEAD       : {w['git_head']}"
          f"{'' if w['matches_audited_commit'] else '   *** NOT the audited commit ***'}")
    print(f"  encoder sha256     : {pol['encoder_sha256'][:16]}...  {pol['encoder']}")
    print(f"  decoder sha256     : {pol['decoder_sha256'][:16]}...  {pol['decoder']}")
    m = prov["motion"]
    print(f"  motion             : {m['name']}"
          + (f"  sha256 {m['sha256'][:16]}..." if m.get("sha256") else ""))
    b = prov["backend"]
    print(f"  backend            : {b['name']} {b['version']}")
    c = prov["sources"]["control"]
    print(f"  physics dt         : {c['physics_dt']['value']}   <- {c['physics_dt']['source']}")
    print(f"  control dt         : {c['control_dt']['value']}    decimation "
          f"{c['decimation']['value']}")
    t = prov["sources"]["terrain_material"]
    print(f"  terrain friction   : static {t['static_friction']['value']} / dynamic "
          f"{t['dynamic_friction']['value']}, combine {t['friction_combine_mode']['value']}")
    print(f"                       <- {t['static_friction']['source']}")
    a = prov["sources"]["actuators"]
    for key in ("stiffness", "damping", "armature", "effort_limit", "action_scale"):
        print(f"  {key:<18} : {a[key]['source']}")
    s = prov["sources"]["initial_state"]["settle"]
    print(f"  settle             : {s['value']}   <- {s['source']}")
    if prov["solver_overrides"]:
        print("  solver overrides   : (NUMERICAL, no training provenance)")
        for k, v in prov["solver_overrides"].items():
            print(f"      {k} = {v}")
    else:
        print("  solver overrides   : none (backend defaults)")
    print(f"  seed               : {prov['seed']}")
    asym = [k for k, v in prov.get("simulator_asymmetries", {}).items()
            if v["status"] == "open" and (not v["affects"] or prov["backend"]["name"] in v["affects"])]
    if asym:
        print(f"  known asymmetries  : {', '.join(asym)}   (recorded, not patched)")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

RESULT_SCHEMA_VERSION = 1


def write_result(path: str, provenance: dict, metrics: dict, series: dict | None = None) -> None:
    """Write one run's canonical JSON result.

    Args:
        path: Output file. Parent directories are created.
        provenance: The dict from :func:`collect_provenance`.
        metrics: Scalar metrics; see the runners for the agreed key set.
        series: Optional per-step arrays (root pose, joint state, torques).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "provenance": provenance,
        "metrics": metrics,
        "series": series or {},
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)


METRIC_KEYS = (
    "backend", "profile", "motion", "seed",
    "requested_s", "survival_s", "terminated", "termination_reason",
    "pelvis_height_min", "pelvis_height_final", "pelvis_height_mean",
    "root_xy_displacement", "joint_pos_abs_max", "joint_vel_abs_max",
    "torque_abs_max", "torque_saturation_steps", "torque_saturation_rate",
    "foot_contact_rate", "nan", "no_motion",
    "policy_rtt_ms_median", "policy_rtt_ms_p99", "control_steps",
)
"""Scalar keys every backend must fill, so results line up in one table."""


# ---------------------------------------------------------------------------
# Helpers both runners share, so an axis means the same thing on either side
# ---------------------------------------------------------------------------


def split_perturbations(specs: list[str] | None, allow_ood: bool) -> tuple[dict, list]:
    """Separate interface-level axes from the ones a simulator applies.

    Interface axes (latency, jitter, sensor noise) are handled by
    :class:`~h2_policy_if.interface_perturbations.InterfaceLayer`, which is
    shared; everything else is applied by the runner against its own model.

    Args:
        specs: ``AXIS=VALUE`` strings, as collected by ``--perturb``.
        allow_ood: Whether OOD_SIM2REAL axes may be used at all.

    Returns:
        ``(interface_specs, simulator_specs)`` -- a dict for the first, and the
        original strings for the second so each runner keeps its own parsing.

    Raises:
        SystemExit: On a malformed spec, or an OOD axis without ``allow_ood``.
    """
    iface: dict = {}
    sim: list = []
    for spec in specs or []:
        if "=" not in spec:
            raise SystemExit(f"--perturb expects AXIS=VALUE, got {spec!r}")
        axis, raw = spec.split("=", 1)
        entry = OOD_PERTURBATIONS.get(axis)
        if entry is not None and entry.layer == "interface":
            if not allow_ood:
                raise SystemExit(
                    f"{axis} is OOD_SIM2REAL -- SONIC training never randomised it, so no "
                    "range justifies a value. Pass --allow-ood to run it anyway; the "
                    "result is then graded OOD_SIM2REAL and is not comparable to a "
                    "strict baseline."
                )
            iface[axis] = float(raw)
        else:
            sim.append(spec)
    return iface, sim


def result_grade(perturbations: dict) -> str:
    """Return the strongest grade present, which is how a result must be filed.

    A run carrying even one OOD_SIM2REAL axis is an OOD_SIM2REAL result; one
    carrying only training-distribution axes is TRAIN_DIST; an unperturbed run
    is a STRICT_BASELINE.
    """
    grades = {rec.get("grade") for rec in perturbations.values()}
    if "OOD_SIM2REAL" in grades:
        return "OOD_SIM2REAL"
    if grades:
        return "TRAIN_DIST"
    return "STRICT_BASELINE"


def print_perturbations(perturbations: dict) -> None:
    """Print what was perturbed, grouped so the grade cannot be missed."""
    if not perturbations:
        return
    grade = result_grade(perturbations)
    print(f"[perturbation] {grade} -- this run is NOT a strict baseline:")
    for axis, rec in sorted(perturbations.items()):
        if rec.get("grade") == "OOD_SIM2REAL":
            unit = rec.get("unit", "")
            print(f"    {axis} = {rec['value']} {unit}   OOD_SIM2REAL, no training range "
                  f"({rec.get('layer', '?')} layer)")
        else:
            print(f"    {axis} = {rec['value']}   range {rec.get('train_range')}   "
                  f"<- {rec.get('source')}")


def print_axes(backend: str) -> None:
    """Print every perturbation axis this backend can apply, with its grade."""
    print(f"perturbation axes for {backend}\n")
    print(f"  {'axis':28s} {'grade':14s} {'support':12s} {'unit':7s} range / neutral")
    print("  " + "-" * 94)
    for axis in sorted(AXIS_SUPPORT):
        keys = (["torso_com_x", "torso_com_y", "torso_com_z"]
                if axis == "torso_com_offset" else [axis])
        for key in keys:
            entry = TRAIN_RANDOMIZATION.get(AXIS_SUPPORT[axis]["train_key"])
            rng = entry.value if entry else None
            if isinstance(rng, dict):
                rng = rng.get(key[-1])
            print(f"  {key:28s} {'TRAIN_DIST':14s} {axis_support(key, backend):12s} "
                  f"{'':7s} {rng}")
    print()
    for axis, entry in sorted(OOD_PERTURBATIONS.items()):
        print(f"  {axis:28s} {'OOD_SIM2REAL':14s} {entry.support.get(backend, 'undeclared'):12s} "
              f"{entry.unit:7s} neutral {entry.neutral}  ({entry.layer})")
    print("\n  TRAIN_DIST axes are bounded by the range training randomised them over.")
    print("  OOD_SIM2REAL axes have no training range; they need --allow-ood and are")
    print("  recorded as OOD_SIM2REAL, never merged with a strict baseline.")


#: Pelvis heights outside this band mean the solver has diverged, not that the
#: robot fell. Recorded as its own termination reason so a blown-up run is never
#: filed as an ordinary fall, and so post-divergence garbage cannot leak into
#: pelvis_height_min. The band is generous: nothing physical leaves it.
DIVERGENCE_BAND_M = (-5.0, 20.0)


def diverged(pelvis_height_m: float) -> bool:
    """Whether a pelvis height is outside the physically possible band."""
    lo, hi = DIVERGENCE_BAND_M
    return not (lo <= pelvis_height_m <= hi)


#: Backends whose runner can record video.
#:
#: Offscreen capture goes through Isaac Lab's Camera sensor, which resolves to
#: ``isaaclab_physx.renderers.isaac_rtx_renderer:IsaacRtxRenderer`` and therefore
#: needs ``omni.usd`` -- the Kit RTX renderer. The Newton backends do not launch
#: Kit, so the sensor cannot even be constructed there; the run aborts while the
#: scene is being built, before a single step. This is a property of Isaac Lab
#: 3.0, not something the runner can work around, so it is declared rather than
#: patched: a batch simply records what it can and says which rows it could not.
VIDEO_BACKENDS = ("mujoco", "isaacsim_physx")


def can_record(backend: str) -> bool:
    """Whether ``backend`` can produce a video."""
    return backend in VIDEO_BACKENDS
