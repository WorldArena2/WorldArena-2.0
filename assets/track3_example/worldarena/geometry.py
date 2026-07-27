"""Shared robot geometry helpers for WorldArena live evaluation."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from worldarena.protocol import (
    CAMERA_ROLE_GLOBAL,
    CAMERA_ROLE_LEFT_WRIST,
    CAMERA_ROLE_RIGHT_WRIST,
)
from worldarena.schema import CameraExtrinsics, CameraIntrinsics, Pose, Quaternion, Vector3

# AgileX dual-arm camera extrinsics (base frame), aligned with pi05_wma live server.
T_LEFT_BASE_TO_CAM = np.array([-0.102, -0.405, 0.655], dtype=np.float64)
T_RIGHT_BASE_TO_CAM = np.array([-0.102, 0.405, 0.655], dtype=np.float64)
R_CAM_FROM_BASE = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64)
R_Z180_LEFT = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)

THETA_X45_RAD = np.radians(45.0)
_C, _S = np.cos(THETA_X45_RAD), np.sin(THETA_X45_RAD)
ROT_X_45 = np.array([[1.0, 0.0, 0.0], [0.0, _C, -_S], [0.0, _S, _C]], dtype=np.float64)

DEFAULT_CAMERA_INTRINSICS = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0)

CAMERA_ROLE_CALIBRATION: Dict[str, Tuple[CameraIntrinsics, CameraExtrinsics]] = {
    CAMERA_ROLE_GLOBAL: (
        DEFAULT_CAMERA_INTRINSICS,
        CameraExtrinsics(
            parent_frame='base',
            camera_pose_parent=Pose(
                position_m=Vector3(float(T_LEFT_BASE_TO_CAM[0]), float(T_LEFT_BASE_TO_CAM[1]), float(T_LEFT_BASE_TO_CAM[2])),
                orientation_xyzw=Quaternion(0.0, 0.0, 0.0, 1.0),
                frame='base',
            ),
        ),
    ),
    CAMERA_ROLE_LEFT_WRIST: (
        CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0),
        CameraExtrinsics(parent_frame='left_ee', camera_pose_parent=None),
    ),
    CAMERA_ROLE_RIGHT_WRIST: (
        CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0),
        CameraExtrinsics(parent_frame='right_ee', camera_pose_parent=None),
    ),
}


def _require_scipy():
    from scipy.spatial.transform import Rotation as R

    return R


def quat_to_rot6d(pose_7d: np.ndarray) -> np.ndarray:
    R = _require_scipy()
    pos = pose_7d[..., :3]
    quat = pose_7d[..., 3:7]
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1e-6):
        quat = quat.copy()
        quat[norm[..., 0] < 1e-6] = [0.0, 0.0, 0.0, 1.0]
    pos = np.where(np.isclose(pos, -1.0), 0.0, pos)
    rot = R.from_quat(quat)
    rot_mat = rot.as_matrix()
    rot6d = rot_mat[..., :, :2].reshape(rot_mat.shape[:-2] + (6,))
    return np.concatenate([pos, rot6d], axis=-1)


def cont6d_to_matrix(cont6d: np.ndarray) -> np.ndarray:
    cont6d = cont6d.reshape(3, 2).transpose(1, 0).reshape(6)
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = x_raw / np.linalg.norm(x_raw, axis=-1, keepdims=True)
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z, axis=-1, keepdims=True)
    y = np.cross(z, x)
    x = x[..., None]
    y = y[..., None]
    z = z[..., None]
    return np.concatenate([x, y, z], axis=-1)


def eef_pose_base_to_camera(eef_pose: np.ndarray, is_left: bool = True) -> np.ndarray:
    R = _require_scipy()
    t_cam_from_base = T_LEFT_BASE_TO_CAM if is_left else T_RIGHT_BASE_TO_CAM

    pos_base = np.asarray(eef_pose[:3], dtype=np.float64)
    quat_base_xyzw = np.asarray(eef_pose[3:7], dtype=np.float64)
    r_eef_base = R.from_quat(quat_base_xyzw).as_matrix()

    if is_left:
        r_eef_base = r_eef_base @ R_Z180_LEFT

    pts_rel = pos_base - t_cam_from_base
    pos_cam = ROT_X_45 @ (R_CAM_FROM_BASE @ pts_rel)
    r_eef_cam = ROT_X_45 @ (R_CAM_FROM_BASE @ r_eef_base)
    quat_cam_xyzw = R.from_matrix(r_eef_cam).as_quat()
    return np.concatenate([pos_cam.astype(np.float32), quat_cam_xyzw.astype(np.float32)], axis=0)


def eef_pose_camera_to_base(eef_pose_cam: np.ndarray, is_left: bool = True) -> np.ndarray:
    R = _require_scipy()
    t_cam_from_base = T_LEFT_BASE_TO_CAM if is_left else T_RIGHT_BASE_TO_CAM

    pos_cam = np.asarray(eef_pose_cam[:3], dtype=np.float64)
    quat_cam_xyzw = np.asarray(eef_pose_cam[3:7], dtype=np.float64)
    r_eef_cam = R.from_quat(quat_cam_xyzw).as_matrix()

    pos_base = np.linalg.inv(R_CAM_FROM_BASE) @ (np.linalg.inv(ROT_X_45) @ pos_cam) + t_cam_from_base
    r_base_adj = np.linalg.inv(R_CAM_FROM_BASE) @ (np.linalg.inv(ROT_X_45) @ r_eef_cam)

    if is_left:
        r_eef_base = r_base_adj @ np.linalg.inv(R_Z180_LEFT)
    else:
        r_eef_base = r_base_adj

    quat_base_xyzw = R.from_matrix(r_eef_base).as_quat()
    return np.concatenate([pos_base.astype(np.float32), quat_base_xyzw.astype(np.float32)], axis=0)


def camera_calibration_for_role(camera_role: str) -> Tuple[CameraIntrinsics, CameraExtrinsics]:
    return CAMERA_ROLE_CALIBRATION.get(
        camera_role,
        (DEFAULT_CAMERA_INTRINSICS, CameraExtrinsics(parent_frame='base')),
    )
