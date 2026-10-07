"""Factory function to instantiate a Unitree H2 RobotModel from the bundled URDF.

Unlike G1 there is no supplemental info yet: the H2 URDF zero configuration is
used as the default body pose (matching the robot's spawn/default stance in the
bundled MJCF), which is sufficient for FK-based VR3PT calibration.
"""

import os
from pathlib import Path

from gear_sonic.data.robot_model.robot_model import RobotModel

# The deployed SONIC H2 policy is trained on the EDU kinematics (h2_edu.urdf);
# the repo-bundled urdf/h2 is the H1031 variant whose wrist-chain origins differ
# by up to ~44 mm. Prefer the EDU description from the read-only wbc_h12
# checkout (env H2_EDU_URDF_DIR overrides), fall back to the bundle with a
# warning so FK-based calibration matches what the policy expects.
_H2_BUNDLE_DIR = (
    Path(__file__).resolve().parents[3] / "data" / "assets" / "robot_description" / "urdf" / "h2"
)
_H2_EDU_DIR = Path(os.environ.get(
    "H2_EDU_URDF_DIR",
    str(Path.home() / "workspace" / "wbc_h12" / "gear_sonic" / "data" / "assets"
        / "robot_description" / "urdf" / "h2_edu"),
))


def instantiate_h2_robot_model() -> RobotModel:
    """Instantiate an H2 robot model (31 actuated DOFs, fixed base) for FK.

    Returns:
        RobotModel: H2 model with ``default_body_pose == q_zero`` (no supplemental info).
    """
    edu_urdf = _H2_EDU_DIR / "h2_edu.urdf"
    if edu_urdf.exists():
        return RobotModel(str(edu_urdf), str(_H2_EDU_DIR))
    print(f"[h2 robot model] WARNING: EDU URDF not found at {edu_urdf}; "
          "falling back to the bundled H1031-variant h2.urdf (wrist FK differs by up to ~44 mm)")
    urdf_path = _H2_BUNDLE_DIR / "h2.urdf"
    if not urdf_path.exists():
        raise FileNotFoundError(f"H2 URDF not found: {urdf_path}")
    return RobotModel(str(urdf_path), str(_H2_BUNDLE_DIR))
