"""H2 variant of the VR 3-point calibration (shared by teleop tools).

Swaps the FK calibration reference of ThreePointPose from G1 to H2. Frame
names and VR-target offsets follow the H2 EDU training configuration
(wbc_h12 motion.yaml vr_3point_body / vr_3point_body_offset).
"""

from __future__ import annotations

import numpy as np

from gear_sonic.data.robot_model.instantiation.h2 import instantiate_h2_robot_model
from gear_sonic.scripts.pico_manager_thread_server import ThreePointPose, _process_3pt_pose
from gear_sonic.utils.teleop.h2_key_frames import (
    H2_KEY_FRAME_OFFSETS,
    get_h2_key_frame_poses,
)


class H2ThreePointPose(ThreePointPose):
    """ThreePointPose with the FK calibration reference swapped to H2.

    Reuses the entire G1 calibration math (neck alignment + wrist offsets);
    only the robot model, the key-frame FK and the torso kinematic-chain
    constants change. Frame names and VR-target offsets follow the H2 EDU
    training config (motion.yaml: wrist_yaw links + torso_link, 0.18/±0.025,
    torso +0.35 z).
    """

    def __init__(self, log_prefix: str = "H2ThreePointPose"):
        super().__init__(
            enable_vis_vr3pt=False,
            with_g1_robot=False,
            enable_waist_tracking=False,
            enable_smpl_vis=False,
            log_prefix=log_prefix,
            robot_model=instantiate_h2_robot_model(),
        )
        # Torso target synthesis chain (pelvis -> torso_link -> +0.35 along torso z):
        # derive the base height from H2 FK instead of G1's hand-tuned 0.05.
        self._robot_model.cache_forward_kinematics(
            self._robot_model.default_body_pose, auto_clip=False
        )
        torso_z = float(self._robot_model.frame_placement("torso_link").translation[2])
        self.TORSO_LINK_OFFSET_Z = torso_z  # H2: 0.0898 m (pelvis -> torso_link)
        self.NECK_LINK_LENGTH = float(H2_KEY_FRAME_OFFSETS["torso"][2])  # 0.35 m
        print(
            f"[{log_prefix}] H2 chain: TORSO_LINK_OFFSET_Z={self.TORSO_LINK_OFFSET_Z:.4f} m, "
            f"NECK_LINK_LENGTH={self.NECK_LINK_LENGTH:.2f} m"
        )

    def _fk_key_frame_poses(self) -> dict:
        # H2 has no supplemental info: the model q IS the 31 actuated joints.
        q = self._override_robot_q if self._override_robot_q is not None else None
        return get_h2_key_frame_poses(self._robot_model, q=q)

    def calibrate_now(self, body_poses_np: np.ndarray) -> bool:
        """Calibrate against H2 FK at the zero (default) configuration."""
        try:
            vr_3pt_pose_raw = _process_3pt_pose(body_poses_np)
            self._override_robot_q = np.zeros(self._robot_model.num_dofs, dtype=np.float64)
            self._capture_calibration(vr_3pt_pose_raw)
            print(f"[{self.log_prefix}] Calibration completed (H2 zero-pose reference)")
            return True
        except Exception as e:  # noqa: BLE001 - mirror base-class behavior
            print(f"[{self.log_prefix}] Calibration failed: {e}")
            import traceback

            traceback.print_exc()
            return False

