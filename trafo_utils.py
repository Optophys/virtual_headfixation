import numpy as np
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

def trafo2local(kp_xyz):
    """ Transforms global keypoints into a rat local coordinate frame.

        The rat local system is spanned by:
            - x: Animal right  (perpendicular to ground plane normal and body axis)
            - y: The body axis (defined by the vector from tail to a point between the ears)
            - z: Animal up (perpendicular to x and y)
        And located in the point midway between the two ear keypoints.
    """
    mid_pt = 0.5 * (kp_xyz[0] + kp_xyz[4])  # point between ears
    body_axis = mid_pt - kp_xyz[9]  # vector from tail to mid ears, 'animal forward'
    body_axis /= np.linalg.norm(body_axis, 2, -1, keepdims=True)

    ground_up = np.array([0.0, -1.0, 0.0])  # vector pointing up
    ground_up /= np.linalg.norm(ground_up, 2)

    animal_right = np.cross(body_axis, ground_up)  # pointing into the animals' right direction
    animal_up = np.cross(animal_right, body_axis)

    R = np.stack([animal_right, body_axis, animal_up], 0)  # rotation matrix
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, -1:] = -np.matmul(R, np.reshape(mid_pt, [3, 1]))  # trans
    kp_xyz_local = trafo_coords(kp_xyz, M)
    return kp_xyz_local



def trafo2HEADlocal(kp_xyz):
    """ Transforms global keypoints into a rat local coordinate frame. Relative to the HEADaxis !

        The rat local system is spanned by:
            - x: Animal right  (perpendicular to ground plane normal and body axis)
            - y: The Head axis (defined by the vector from Nose to a point between the ears)
            - z: Animal up (perpendicular to x and y)
        And located in the point midway between the two ear keypoints.
    """
    mid_pt = 0.5 * (kp_xyz[0] + kp_xyz[4])  # point between ears
    head_axis = kp_xyz[8] - mid_pt    # vector from nose to mid ears, 'animal forward'
    head_axis /= np.linalg.norm(head_axis, 2, -1, keepdims=True)
    #body_axis[1]=0 #this is the difference to trafo2local
    ground_up = np.array([0.0, -1.0, 0.0])  # vector pointing up
    ground_up /= np.linalg.norm(ground_up, 2)

    animal_right = np.cross(head_axis, ground_up)  # pointing into the animals' right directcs-ion
    animal_up = np.cross(animal_right, head_axis)

    R = np.stack([animal_right, head_axis, animal_up], 0)  # rotation matrix
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, -1:] = -np.matmul(R, np.reshape(mid_pt, [3, 1]))  # trans
    kp_xyz_local = trafo_coords(kp_xyz, M)
    return kp_xyz_local


