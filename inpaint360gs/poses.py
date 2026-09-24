# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Virtual camera trajectory (paper Sec. 3.4), numpy only.

Ports ``utils/pose_utils.py`` of the official code: the capture poses are PCA-aligned, a
circle is laid in the PCA xy-plane (at the mean camera height) around the focus point of the
optical axes, every camera looks at that focus point, and the circle radius is chosen so that
the removed object fills ~70% of the field of view (``generate_virtual_radius``).

Poses in and out are OpenCV camera-to-world matrices (x right, y down, z forward), as
``aurafusion.scene.ViewCamera.c2w``. The official code works on OpenGL c2w internally and
returns w2c matrices whose rotation still carries the PCA scale; the images are identical, but
here the returned c2w is orthonormal so depths stay in world units.
"""

from __future__ import annotations

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """[3, 4] OpenGL camera-to-world whose +z axis is ``lookdir`` (the camera looks along -z)."""
    vec2 = _normalize(lookdir)
    vec0 = _normalize(np.cross(up, vec2))
    vec1 = _normalize(np.cross(vec2, vec0))
    return np.stack([vec0, vec1, vec2, position], axis=1)


def opencv_to_opengl(c2w: np.ndarray) -> np.ndarray:
    """Flip the camera y and z axes; the map is its own inverse."""
    out = np.array(c2w, dtype=np.float64, copy=True)
    out[..., :3, 1:3] *= -1
    return out


def pad_poses(p: np.ndarray) -> np.ndarray:
    bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def transform_poses_pca(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Official ``transform_poses_pca``: principal axes of the camera centres onto xyz, centred,
    y-up flipped if needed, scaled into [-1, 1]^3. Returns ([N, 3, 4] poses, [4, 4] transform)."""
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean
    eigval, eigvec = np.linalg.eig(t.T @ t)
    eigval, eigvec = eigval.real, eigvec.real
    eigvec = eigvec[:, np.argsort(eigval)[::-1]]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot
    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = (transform @ pad_poses(poses))[..., :3, :4]
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform
    scale_factor = 1.0 / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform
    return poses_recentered, transform


def focus_point(poses: np.ndarray) -> np.ndarray:
    """Least-squares point nearest to every optical axis (OpenGL poses: the axis is column 2)."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    return np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]


def _pca_frame(c2w_opencv: np.ndarray):
    poses, transform = transform_poses_pca(opencv_to_opengl(c2w_opencv))
    center = focus_point(poses)
    offset = np.array([center[0], center[1], 0.0])
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    return poses, transform, center, sc


def virtual_radius(c2w_opencv: np.ndarray, fov_x: float, fov_y: float, object_radius: float) -> float:
    """Official ``generate_virtual_radius``: circle radius, as a ratio of the capture's 90th
    percentile camera spread, at which an object of world radius ``object_radius`` fills 70% of
    the view."""
    _, transform, _, sc = _pca_frame(c2w_opencv)
    pca_radius = object_radius * np.linalg.norm(transform[:3, :3], axis=0).mean()
    distance = max(pca_radius / np.tan(fov_x / 2.0), pca_radius / np.tan(fov_y / 2.0)) / 0.7
    return float(distance / np.max(sc))


def circle_path(c2w_opencv: np.ndarray, n_frames: int = 30, circle_radius: float = 1.0) -> np.ndarray:
    """Official ``generate_ellipse_path(is_circle=True, z_variation=0, const_speed=True)``.

    Returns [n_frames, 4, 4] OpenCV camera-to-world matrices with orthonormal rotations.
    """
    poses, transform, center, sc = _pca_frame(c2w_opencv)
    r = np.max(sc) * circle_radius

    def positions_at(theta):
        return np.stack([center[0] + r * np.cos(theta), center[1] + r * np.sin(theta), np.zeros_like(theta)], -1)

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = positions_at(theta)
    # Constant speed: invert the CDF of segment lengths (official stepfun.sample_np, deterministic).
    lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    cdf = np.concatenate([[0.0], np.minimum(1, np.cumsum((lengths / lengths.sum())[:-1])), [1.0]])
    theta = np.interp(np.linspace(0, 1.0 - np.finfo(np.float32).eps, n_frames + 1), cdf, theta)
    positions = positions_at(theta)[:-1]

    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    inverse = np.linalg.inv(transform)
    out = []
    for p in positions:
        pose = inverse @ pad_poses(viewmatrix(p - center, up, p))
        pose[:3, :3] /= np.linalg.norm(pose[:3, :3], axis=0, keepdims=True)  # drop the PCA scale
        out.append(opencv_to_opengl(pose))
    return np.stack(out)
