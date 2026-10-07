"""H2 key-frame FK for VR 3-point calibration (display-independent).

Frame names and local offsets follow the H2 EDU training configuration
(gear_sonic/config/manager_env/commands/terms/motion.yaml:48-49, not overridden
by sonic_h2_edu.yaml):

    vr_3point_body:        [left_wrist_yaw_link, right_wrist_yaw_link, torso_link]
    vr_3point_body_offset: [[0.18, -0.025, 0], [0.18, +0.025, 0], [0, 0, 0.35]]
    anchor_body:           pelvis

These happen to coincide with the G1 values; only the FK model differs.
"""

from typing import Dict

import numpy as np
from scipy.spatial.transform import Rotation as sRot

H2_FRAME_MAPPING = {
    "left_wrist": "left_wrist_yaw_link",
    "right_wrist": "right_wrist_yaw_link",
    "torso": "torso_link",
}

H2_KEY_FRAME_OFFSETS = {
    "left_wrist": np.array([0.18, -0.025, 0.0]),
    "right_wrist": np.array([0.18, 0.025, 0.0]),
    "torso": np.array([0.0, 0.0, 0.35]),
}


def get_h2_key_frame_poses(
    robot_model,
    q: np.ndarray = None,
    root_position: np.ndarray = None,
    apply_offset: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """H2 analogue of ``get_g1_key_frame_poses`` (same output contract).

    Args:
        robot_model: Pinocchio-based H2 RobotModel (``instantiate_h2_robot_model``).
        q: Joint configuration [rad]; ``None`` uses ``default_body_pose`` (zero).
        root_position: Pelvis position [m]; default origin.
        apply_offset: Apply the training VR-target local offsets.

    Returns:
        Dict with keys 'left_wrist', 'right_wrist', 'torso', each holding
        'position' [m], 'orientation_xyzw' and 'orientation_wxyz'.
    """
    if q is None:
        q = robot_model.default_body_pose
    if root_position is None:
        root_position = np.array([0.0, 0.0, 0.0])

    robot_model.cache_forward_kinematics(q, auto_clip=False)

    result = {}
    for key, frame_name in H2_FRAME_MAPPING.items():
        try:
            placement = robot_model.frame_placement(frame_name)
        except ValueError as e:
            raise RuntimeError(
                f"Cannot find frame '{frame_name}' (key='{key}') in H2 robot model. "
                f"Original error: {e}"
            ) from e

        rotation_matrix = placement.rotation
        if apply_offset and key in H2_KEY_FRAME_OFFSETS:
            world_offset = rotation_matrix @ H2_KEY_FRAME_OFFSETS[key]
            position = placement.translation + world_offset + root_position
        else:
            position = placement.translation + root_position

        quat_xyzw = sRot.from_matrix(rotation_matrix).as_quat()
        result[key] = {
            "position": position,
            "orientation_xyzw": quat_xyzw,
            "orientation_wxyz": np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            ),
        }
    return result
