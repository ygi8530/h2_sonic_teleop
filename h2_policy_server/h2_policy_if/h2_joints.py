"""H2 EDU joint orders, PD gains, action scale and default pose.

Every number here is a copy of a value that lives in ``wbc_h12`` and is quoted
with its source so it can be re-checked after an upstream swap. They are copied
rather than imported because ``gear_sonic/envs/manager_env/robots/h2.py`` imports
``isaaclab`` at module scope, which the slim server container deliberately does
not carry. ``tools/verify_constants.py`` re-derives all of them from the
``wbc_h12`` sources and fails loudly on any drift.

Two joint orders exist and mixing them up is the single most likely cause of a
policy that "runs" but falls over:

* **MuJoCo order** - the order of ``mjcf/h2_edu.xml``, which is also the URDF and
  hardware motor order, and the order the motion ``.pkl`` files store ``dof`` in.
* **IsaacLab order** - Isaac Lab's breadth-first articulation order. Every vector
  the policy itself consumes or produces (proprio history, decoder action) is in
  this order.

Conversions (``robots/h2.py:61,94``)::

    q_isaaclab = q_mujoco[MUJOCO_TO_ISAACLAB]
    q_mujoco   = q_isaaclab[ISAACLAB_TO_MUJOCO]
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

NUM_DOF = 31

# gear_sonic/envs/manager_env/robots/h2.py:94 (H2_MUJOCO_TO_ISAACLAB_DOF)
MUJOCO_TO_ISAACLAB = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 17, 24, 4, 10, 16, 18, 25,
     5, 11, 19, 26, 20, 27, 21, 28, 22, 29, 23, 30],
    dtype=np.int64,
)
# gear_sonic/envs/manager_env/robots/h2.py:61 (H2_ISAACLAB_TO_MUJOCO_DOF)
ISAACLAB_TO_MUJOCO = np.array(
    [0, 3, 6, 9, 14, 19, 1, 4, 7, 10, 15, 20, 2, 5, 8, 11, 16, 12,
     17, 21, 23, 25, 27, 29, 13, 18, 22, 24, 26, 28, 30],
    dtype=np.int64,
)

# Joint names in MuJoCo order, for error messages and for the MJCF cross-check.
JOINT_NAMES_MUJOCO = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    # H2's ankle chain is INVERTED vs G1/H1-2: knee -> ankle_roll -> ankle_pitch,
    # so *_ankle_pitch_link is the sole. See h2_tools/README.md.
    "left_ankle_roll_joint", "left_ankle_pitch_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_roll_joint", "right_ankle_pitch_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "head_pitch_joint", "head_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# --------------------------------------------------------------------------
# Controller gains, MuJoCo order. Derived in robots/h2.py:7-23 from the motor
# armature and a 10 Hz / damping-ratio-2 target:
#     kp = armature * (2*pi*10)**2        kd = 2 * 2.0 * armature * (2*pi*10)
# The "feet", "waist"(roll/pitch) and "head" groups carry a 2.0 factor.
# --------------------------------------------------------------------------
KP_MUJOCO = np.array(
    [99.0984, 99.0984, 40.1792, 99.0984, 28.5012, 28.5012,
     99.0984, 99.0984, 40.1792, 99.0984, 28.5012, 28.5012,
     40.1792, 28.5012, 28.5012, 28.5012, 28.5012,
     14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783,
     14.2506, 14.2506, 14.2506, 14.2506, 14.2506, 16.7783, 16.7783],
    dtype=np.float64,
)
KD_MUJOCO = np.array(
    [6.3088, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144,
     6.3088, 6.3088, 2.5579, 6.3088, 1.8144, 1.8144,
     2.5579, 1.8144, 1.8144, 1.8144, 1.8144,
     0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681,
     0.9072, 0.9072, 0.9072, 0.9072, 0.9072, 1.0681, 1.0681],
    dtype=np.float64,
)

# robots/h2.py:379-390, H2_ACTION_SCALE = 0.25 * effort_limit / stiffness.
# NOTE the effort limits used are H1031's, NOT H2 EDU's: h2_edu.py:61 sets
# H2_EDU_ACTION_SCALE = H2_ACTION_SCALE verbatim, so this is what training used
# and therefore what deployment must use. EDU's own (tighter) effort limits act
# only as actuator saturation.
ACTION_SCALE_MUJOCO = np.array(
    [1.051984, 1.051984, 1.642639, 1.051984, 1.315732, 1.315732,
     1.051984, 1.051984, 1.642639, 1.051984, 1.315732, 1.315732,
     1.642639, 1.315732, 1.315732, 1.315732, 1.315732,
     1.315732, 1.315732, 1.315732, 1.315732, 1.315732, 0.223503, 0.223503,
     1.315732, 1.315732, 1.315732, 1.315732, 1.315732, 0.223503, 0.223503],
    dtype=np.float64,
)

# robots/h2.py:235-244 init_state.joint_pos [rad]; every unlisted joint is 0.0.
DEFAULT_ANGLES_MUJOCO = np.array(
    [-0.312, 0.0, 0.0, 0.669, 0.0, -0.363,
     -0.312, 0.0, 0.0, 0.669, 0.0, -0.363,
     0.0, 0.0, 0.0, 0.0, 0.0,
     0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
     0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0],
    dtype=np.float64,
)

# Rotor inertia reflected at the joint [kg m^2], robots/h2.py actuator
# ``armature=``. Isaac Lab's implicit actuators add this to the joint inertia; a
# MuJoCo scene built without it is numerically unstable at these stiffnesses and
# a 5 ms timestep, so any hand-built scene must set it.
ARMATURE_MUJOCO = np.array(
    [0.025101925, 0.025101925, 0.010177520, 0.025101925, 0.007219450, 0.007219450,
     0.025101925, 0.025101925, 0.010177520, 0.025101925, 0.007219450, 0.007219450,
     0.010177520, 0.007219450, 0.007219450, 0.007219450, 0.007219450,
     0.003609725, 0.003609725, 0.003609725, 0.003609725, 0.003609725, 0.004250, 0.004250,
     0.003609725, 0.003609725, 0.003609725, 0.003609725, 0.003609725, 0.004250, 0.004250],
    dtype=np.float64,
)

# Peak joint torque [N m], from robots/h2_edu.py's ``effort_limit_sim`` overrides
# (the production H2 EDU values, taken from Unitree's URDF). These are the real
# machine's limits and differ sharply from H1031's: ankle_roll 19.0 vs 150.0.
# Unlike ACTION_SCALE_MUJOCO, which must keep H1031's numbers because training
# used them, a simulation should saturate at these.
EFFORT_LIMIT_MUJOCO = np.array(
    [360.0, 360.0, 360.0, 360.0, 19.0, 66.9,
     360.0, 360.0, 360.0, 360.0, 19.0, 66.9,
     120.0, 180.0, 180.0, 50.0, 50.0,
     130.0, 60.0, 60.0, 60.0, 60.0, 10.0, 10.0,
     130.0, 60.0, 60.0, 60.0, 60.0, 10.0, 10.0],
    dtype=np.float64,
)

# robots/h2.py:234 init_state.pos -- pelvis height at spawn [m].
DEFAULT_BASE_HEIGHT = 1.04

KP_ISAACLAB = KP_MUJOCO[MUJOCO_TO_ISAACLAB]
KD_ISAACLAB = KD_MUJOCO[MUJOCO_TO_ISAACLAB]
ACTION_SCALE_ISAACLAB = ACTION_SCALE_MUJOCO[MUJOCO_TO_ISAACLAB]
DEFAULT_ANGLES_ISAACLAB = DEFAULT_ANGLES_MUJOCO[MUJOCO_TO_ISAACLAB]
ARMATURE_ISAACLAB = ARMATURE_MUJOCO[MUJOCO_TO_ISAACLAB]
EFFORT_LIMIT_ISAACLAB = EFFORT_LIMIT_MUJOCO[MUJOCO_TO_ISAACLAB]


def to_isaaclab(q_mujoco: np.ndarray) -> np.ndarray:
    """Reindex a per-joint array from MuJoCo order to IsaacLab order.

    Args:
        q_mujoco: Array whose last axis is the 31 joints in MuJoCo order.

    Returns:
        The same values with the last axis in IsaacLab order.
    """
    return np.asarray(q_mujoco)[..., MUJOCO_TO_ISAACLAB]


def to_mujoco(q_isaaclab: np.ndarray) -> np.ndarray:
    """Reindex a per-joint array from IsaacLab order to MuJoCo order.

    Args:
        q_isaaclab: Array whose last axis is the 31 joints in IsaacLab order.

    Returns:
        The same values with the last axis in MuJoCo order.
    """
    return np.asarray(q_isaaclab)[..., ISAACLAB_TO_MUJOCO]


def read_mjcf_joint_names(mjcf_path: str) -> list[str]:
    """Read hinge joint names from an MJCF, in document (MuJoCo) order.

    Uses ElementTree rather than MuJoCo itself so the check needs no mesh files
    and no MuJoCo install, matching how ``motion_lib`` parses the same asset.

    Args:
        mjcf_path: Path to ``mjcf/h2_edu.xml``.

    Returns:
        Names of every non-free joint, in the order the file declares them.
    """
    root = ET.parse(mjcf_path).getroot()
    return [j.get("name") for j in root.iter("joint") if j.get("type") != "free" and j.get("name")]


def check_against_mjcf(mjcf_path: str) -> None:
    """Assert this module's joint order still matches the upstream asset.

    Args:
        mjcf_path: Path to ``mjcf/h2_edu.xml`` inside the read-only ``wbc_h12``.

    Raises:
        RuntimeError: If the asset's joint count or order differs from
            :data:`JOINT_NAMES_MUJOCO`, which invalidates every table here.
    """
    names = read_mjcf_joint_names(mjcf_path)
    if len(names) != NUM_DOF:
        raise RuntimeError(f"{mjcf_path} has {len(names)} hinge joints, expected {NUM_DOF}")
    mismatch = [
        f"  [{i}] asset={a!r} expected={b!r}"
        for i, (a, b) in enumerate(zip(names, JOINT_NAMES_MUJOCO))
        if a != b
    ]
    if mismatch:
        raise RuntimeError(
            "wbc_h12 joint order no longer matches h2_policy_if/h2_joints.py:\n"
            + "\n".join(mismatch)
            + "\nRe-run tools/verify_constants.py before trusting the server."
        )


def _check_permutations() -> None:
    ident = np.arange(NUM_DOF)
    if not np.array_equal(MUJOCO_TO_ISAACLAB[ISAACLAB_TO_MUJOCO], ident):
        raise RuntimeError("MUJOCO_TO_ISAACLAB and ISAACLAB_TO_MUJOCO are not inverses")
    if not np.array_equal(np.sort(MUJOCO_TO_ISAACLAB), ident):
        raise RuntimeError("MUJOCO_TO_ISAACLAB is not a permutation of 0..30")


_check_permutations()
