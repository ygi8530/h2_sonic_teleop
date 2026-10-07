"""Upper-body differential IK for H2 VR3PT teleop stage 2-B (no policy, no dynamics).

Solves waist (3) + both arms (2x7) joint angles so that the two wrist targets
(position + orientation) and the torso orientation follow the calibrated VR
3-point pose. Damped least squares on the Pinocchio model from
``instantiate_h2_robot_model``; legs and head stay at zero. Everything is
kinematic: the result is a pose preview, not control.

Targets are expressed in the pelvis frame (exactly what H2ThreePointPose
outputs) and the wrist task point includes the training VR-target offset
(wrist_yaw_link + [0.18, -/+0.025, 0], see h2_key_frames.py).
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as sRot

from gear_sonic.utils.teleop.h2_key_frames import H2_FRAME_MAPPING, H2_KEY_FRAME_OFFSETS

IK_JOINTS = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Task weights: wrist position dominates; orientations keep hands/torso aligned
W_WRIST_POS = 1.0
W_WRIST_ORI = 0.25
W_TORSO_ORI = 0.4
DAMPING = 1e-4
MAX_STEP = 0.15  # [rad] per solve() call, keeps the preview jump-free
ITERS = 8


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class H2UpperBodyIK:
    def __init__(self, robot_model):
        self._rm = robot_model
        self._model = robot_model.pinocchio_wrapper.model
        self._data = robot_model.pinocchio_wrapper.data
        self._q = robot_model.default_body_pose.copy().astype(np.float64)
        self._idx = np.array([robot_model.dof_index(n) for n in IK_JOINTS])
        self._lo = np.array([robot_model.lower_joint_limits[i] for i in self._idx])
        self._hi = np.array([robot_model.upper_joint_limits[i] for i in self._idx])
        self._fids = {
            key: self._model.getFrameId(frame) for key, frame in H2_FRAME_MAPPING.items()
        }

    def reset(self):
        self._q = self._rm.default_body_pose.copy().astype(np.float64)

    @property
    def q(self) -> np.ndarray:
        """Full model configuration [rad] (legs/head zero, upper body solved)."""
        return self._q.copy()

    def joint_dict(self) -> dict:
        """{joint_name: angle [rad]} for the IK joints only."""
        return {n: float(self._q[i]) for n, i in zip(IK_JOINTS, self._idx)}

    def solve(self, vr_3pt_pose: np.ndarray) -> float:
        """One control-tick solve toward the calibrated VR 3-point pose.

        Args:
            vr_3pt_pose: (3,7) rows [L-wrist, R-wrist, torso], each
                         [x y z, qw qx qy qz] in the pelvis frame [m].

        Returns:
            Residual norm after the update (position terms, [m]-scale).
        """
        targets = {
            "left_wrist": vr_3pt_pose[0],
            "right_wrist": vr_3pt_pose[1],
            "torso": vr_3pt_pose[2],
        }
        res_norm = 0.0
        for _ in range(ITERS):
            pin.forwardKinematics(self._model, self._data, self._q)
            pin.updateFramePlacements(self._model, self._data)
            pin.computeJointJacobians(self._model, self._data, self._q)

            rows_J, rows_e = [], []
            for key in ("left_wrist", "right_wrist"):
                fid = self._fids[key]
                plc = self._data.oMf[fid]
                J6 = pin.getFrameJacobian(
                    self._model, self._data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )
                r = plc.rotation @ H2_KEY_FRAME_OFFSETS[key]
                p_cur = plc.translation + r
                Jp = J6[:3] - _skew(r) @ J6[3:]
                tgt = targets[key]
                e_pos = tgt[:3] - p_cur
                R_tgt = sRot.from_quat(tgt[3:], scalar_first=True).as_matrix()
                e_ori = pin.log3(R_tgt @ plc.rotation.T)
                rows_J += [W_WRIST_POS * Jp, W_WRIST_ORI * J6[3:]]
                rows_e += [W_WRIST_POS * e_pos, W_WRIST_ORI * e_ori]

            fid = self._fids["torso"]
            plc = self._data.oMf[fid]
            J6 = pin.getFrameJacobian(
                self._model, self._data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            R_tgt = sRot.from_quat(targets["torso"][3:], scalar_first=True).as_matrix()
            e_ori = pin.log3(R_tgt @ plc.rotation.T)
            rows_J.append(W_TORSO_ORI * J6[3:])
            rows_e.append(W_TORSO_ORI * e_ori)

            J = np.vstack(rows_J)[:, self._idx]
            e = np.concatenate(rows_e)
            res_norm = float(np.linalg.norm(e))
            # Damped least squares step
            H = J.T @ J + DAMPING * np.eye(len(self._idx))
            dq = np.linalg.solve(H, J.T @ e)
            self._q[self._idx] = np.clip(
                self._q[self._idx] + np.clip(dq, -MAX_STEP / ITERS, MAX_STEP / ITERS),
                self._lo,
                self._hi,
            )
        return res_norm
