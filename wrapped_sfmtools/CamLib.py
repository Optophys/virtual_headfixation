"""" Small library providing commonly used operations with cameras. """
import numpy as np
import cv2


def triangulate_opencv(K1, dist1, M1,
                       K2, dist2, M2,
                       points1, points2, invert_M=False):
    """ Triangulates a 3D point from 2D observations and calibration.
        K1, K2: Camera intrinsics
        M1, M2: Camera extrinsics, mapping from world -> cam coord if invert_M is False
        points1, points2: Nx2 np.array of observed points
        invert_M: flag, set true when given M1, M2 are the mapping from cam -> root
     """
    if invert_M:
        M1 = np.linalg.inv(M1)
        M2 = np.linalg.inv(M2)

    # assemble projection matrices
    P1 = np.matmul(K1, M1[:3, :])
    P2 = np.matmul(K2, M2[:3, :])

    # undistort points
    point_cam1_coord = undistort_points(points1, K1, dist1)
    point_cam2_coord = undistort_points(points2, K1, dist1)

    # triangulate to 3D
    points3d_h = cv2.triangulatePoints(P1, P2,
                                       np.transpose(point_cam1_coord), np.transpose(point_cam2_coord))
    points3d_h = np.transpose(points3d_h)
    return _from_hom(points3d_h)


def project(xyz_coords, K, dist=None):
    """ Projects a (x, y, z) tuple of world coords into the camera image frame. """
    xyz_coords = np.reshape(xyz_coords, [-1, 3])
    uv_coords = np.matmul(xyz_coords, np.transpose(K, [1, 0]))
    uv_coords = _from_hom(uv_coords)
    if dist is not None:
        uv_coords = distort_points(uv_coords,  K, dist)
    return uv_coords


def backproject(uv_coords, z_coords, K, dist=None):
    """ Backprojects a (u, v) distorted point observation within the image frame into the corresponding world frame. """
    uv_coords = np.reshape(uv_coords, [-1, 2])
    z_coords = np.reshape(z_coords, [-1, 1])
    assert uv_coords.shape[0] == z_coords.shape[0], "Number of points differs."

    if dist is not None:
        uv_coords = undistort_points(uv_coords,  K, dist)

    uv_coords_h = _to_hom(uv_coords)
    xyz_coords = z_coords * np.matmul(uv_coords_h, np.transpose(K, [1, 0]))
    return xyz_coords


def trafo_coords(xyz_coords, T):
    """ Applies a given a transformation T to a set of euclidean coordinates. """
    xyz_coords_h = _to_hom(xyz_coords)
    xyz_trafo_coords_h = np.matmul(xyz_coords_h, np.transpose(T, [1, 0]))
    return _from_hom(xyz_trafo_coords_h)


def _to_hom(coords):
    """ Turns the [N, D] coord matrix into homogeneous coordinates [N, D+1]. """
    coords_h = np.concatenate([coords, np.ones((coords.shape[0], 1))], 1)
    return coords_h


def _from_hom(coords_h):
    """ Turns the homogeneous coordinates [N, D+1] into [N, D]. """
    coords = coords_h[:, :-1] / (coords_h[:, -1:] + 1e-10)
    return coords


def distort_points(points, K, dist):
    """ Given points this function returns where the observed points would lie with lens distortion."""
    assert len(points.shape) == 2, "Shape mismatch."
    dist = np.reshape(dist, [5])
    points = points.copy()

    # To relative coordinates
    points[:, 0] = (points[:, 0] - K[0, 2]) / K[0, 0]
    points[:, 1] = (points[:, 1] - K[1, 2]) / K[1, 1]

    # squared radial distance
    r2 = points[:, 0]*points[:, 0] + points[:, 1]*points[:, 1]

    # get distortion params
    k1, k2, p1, p2, k3 = dist[0], dist[1], dist[2], dist[3], dist[4]

    # Radial distorsion
    dist_x = points[:, 0] * (1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)
    dist_y = points[:, 1] * (1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2)

    # Tangential distorsion
    x, y = points[:, 0], points[:, 1]
    dist_x += 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    dist_y += p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

    # Back to absolute coordinates.
    dist_x = dist_x * K[0, 0] + K[0, 2]
    dist_y = dist_y * K[1, 1] + K[1, 2]
    points = np.stack([dist_x, dist_y], 1)
    points = np.reshape(points, [-1, 2])
    return points


def undistort_points(points, K, dist):
    """ Given observed points this function returns where the point would lie when there would be no lens distortion."""
    points = np.reshape(points, [-1, 2])
    # Runs an iterative algorithm to invert what distort_points(..) does
    points_dist = cv2.undistortPoints(np.expand_dims(points, 0),
                                           K, np.squeeze(dist),
                                           P=K)
    return np.squeeze(points_dist, 0)
