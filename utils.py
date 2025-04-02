import numpy as np 
import pyquaternion as pq
import tensorflow as tf
# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def view_matrix_to_quaternion(vm):
    """
    Extracts the rotation part from a 4x4 view matrix and orthogonalizes it
    via SVD, then converts it to a quaternion.
    
    Assumes vm is of the form:
       [ R_3x3   0 ]
       [ t_3     1 ]
    where t_3 is the translation stored in the last row.
    """
    R = vm[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return pq.Quaternion(matrix=R_ortho)

def compute_quat_from_vectors(ref, target):
    """
    Given a reference vector and a target vector, compute the quaternion that
    rotates ref to target.
    
    Parameters:
      ref: reference vector (e.g. [0, 1, 0] for 3D or [1, 0] for 2D, extended appropriately)
      target: target vector
    Returns:
      A pyquaternion.Quaternion.
    """
    norm_target = np.linalg.norm(target)
    if norm_target < 1e-8:
        return pq.Quaternion()  # Identity
    target_norm = target / norm_target
    if np.allclose(target_norm, ref):
        return pq.Quaternion()  # No rotation needed
    axis = np.cross(ref, target_norm) if ref.ndim == 1 and target_norm.ndim == 1 and ref.size == 3 else np.array([0, 0, 1])
    norm_axis = np.linalg.norm(axis)
    if norm_axis < 1e-8:
        # If vectors are opposite, choose an arbitrary perpendicular axis.
        if ref.size == 3:
            axis = np.cross(ref, np.array([1, 0, 0]))
            if np.linalg.norm(axis) < 1e-8:
                axis = np.cross(ref, np.array([0, 1, 0]))
        else:
            axis = np.array([0, 0, 1])
        axis = axis / np.linalg.norm(axis)
        angle = np.pi
    else:
        axis = axis / norm_axis
        dot_val = np.clip(np.dot(ref, target_norm), -1.0, 1.0)
        angle = np.arccos(dot_val)
    return pq.Quaternion(axis=axis, angle=angle)

def compute_quat_from_2d(ref, parent_pt, child_pt):
    """
    Computes a quaternion representing a rotation about the Z-axis that
    aligns a reference 2D vector (ref) with the vector from parent_pt to child_pt.
    """
    vec = child_pt - parent_pt
    norm_vec = np.linalg.norm(vec)
    if norm_vec < 1e-8:
        return pq.Quaternion()
    vec_norm = vec / norm_vec
    angle_ref = np.arctan2(ref[1], ref[0])
    angle_vec = np.arctan2(vec_norm[1], vec_norm[0])
    angle = angle_vec - angle_ref
    return pq.Quaternion(axis=[0, 0, 1], angle=angle)

def compute_joint_quaternions_3d(kp3d, joint_order, parent_map, ref=np.array([0, 1, 0])):
    """
    Computes a per-joint quaternion (for 3D keypoints) by computing the bone vector
    from the parent joint to the current joint. If no parent is available, returns identity.
    
    Returns an array of shape (num_joints, 4).
    """
    num_joints = len(joint_order)
    quats = np.zeros((num_joints, 4), dtype=np.float32)
    for i, joint in enumerate(joint_order):
        if joint in parent_map:
            parent_joint = parent_map[joint]
            parent_idx = joint_order.index(parent_joint)
            vec = kp3d[i] - kp3d[parent_idx]
            q = compute_quat_from_vectors(ref, vec)
        else:
            q = pq.Quaternion()  # Identity for root joints
        quats[i] = q.elements  # [w, x, y, z]
    return quats

def compute_joint_quaternions_2d(kp2d, joint_order, parent_map, ref=np.array([1, 0])):
    """
    Computes a per-joint quaternion (for 2D keypoints) by computing the vector
    from the parent keypoint to the current keypoint, then computing a rotation
    about the Z-axis (since 2D rotations occur about Z).
    
    Returns an array of shape (num_joints, 4).
    """
    num_joints = len(joint_order)
    quats = np.zeros((num_joints, 4), dtype=np.float32)
    for i, joint in enumerate(joint_order):
        if joint in parent_map:
            parent_joint = parent_map[joint]
            parent_idx = joint_order.index(parent_joint)
            q = compute_quat_from_2d(ref, kp2d[parent_idx], kp2d[i])
        else:
            q = pq.Quaternion()
        quats[i] = q.elements
    return quats

def get_parent_map():
    """
    Build a parent mapping from the provided connection list.
    For each connection (parent, child), if a joint appears as a child, its parent is recorded.
    """
    conns = [
      ('Head', 'Neck'),
      ('Neck', 'Chest'),
      ('Chest', 'LeftShoulder'),
      ('LeftShoulder', 'LeftArm'),
      ('LeftArm', 'LeftForearm'),
      ('LeftForearm', 'LeftHand'),
      ('Chest', 'RightShoulder'),
      ('RightShoulder', 'RightArm'),
      ('RightArm', 'RightForearm'),
      ('RightForearm', 'RightHand'),
      ('Hips', 'LeftThigh'),
      ('LeftThigh', 'LeftLeg'),
      ('LeftLeg', 'LeftFoot'),
      ('Hips', 'RightThigh'),
      ('RightThigh', 'RightLeg'),
      ('RightLeg', 'RightFoot'),
      ('RightHand', 'RightFinger'),
      ('RightFinger', 'RightFingerEnd'),
      ('LeftHand', 'LeftFinger'),
      ('LeftFinger', 'LeftFingerEnd'),
      ('Head', 'HeadEnd'),
      ('RightFoot', 'RightHeel'),
      ('RightHeel', 'RightToe'),
      ('RightToe', 'RightToeEnd'),
      ('LeftFoot', 'LeftHeel'),
      ('LeftHeel', 'LeftToe'),
      ('LeftToe', 'LeftToeEnd'),
      ('SpineLow', 'Hips'),
      ('SpineMid', 'SpineLow'),
      ('Chest', 'SpineMid')
    ]
    parent_map = {}
    for parent, child in conns:
        if child not in parent_map:
            parent_map[child] = parent
    return parent_map
