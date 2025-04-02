import numpy as np
import pyquaternion as pq
import matplotlib.pyplot as plt
from dataloader import PoseDataLoader
import json 

data_path  = './data/train_data_20.json'
val_data = './data/val_data_20.json'


dl = PoseDataLoader(data_path,val_data)

vm = dl.__getitem__(1113)['vm']              # View matrix (4x4)
kp2d = dl.__getitem__(1113)['kp2d_camera']    # 2D keypoints (Nx2)
kp3d = dl.__getitem__(1113)['kp3d_world']       # 3D keypoints (Nx3)
joint_order = dl.joint_order              # List of joint names

conns = [('Head','Neck'), ('Neck','Chest'), ('Chest','LeftShoulder'), ('LeftShoulder','LeftArm'),
         ('LeftArm','LeftForearm'), ('LeftForearm','LeftHand'), ('Chest','RightShoulder'),
         ('RightShoulder','RightArm'), ('RightArm','RightForearm'), ('RightForearm','RightHand'),
         ('Hips','LeftThigh'), ('LeftThigh','LeftLeg'), ('LeftLeg','LeftFoot'),
         ('Hips','RightThigh'), ('RightThigh','RightLeg'), ('RightLeg','RightFoot'),
         ('RightHand','RightFinger'), ('RightFinger','RightFingerEnd'),
         ('LeftHand','LeftFinger'), ('LeftFinger','LeftFingerEnd'),
         ('Head','HeadEnd'), ('RightFoot','RightHeel'), ('RightHeel','RightToe'),
         ('RightToe','RightToeEnd'), ('LeftFoot','LeftHeel'), ('LeftHeel','LeftToe'),
         ('LeftToe','LeftToeEnd'), ('SpineLow','Hips'), ('SpineMid','SpineLow'),
         ('Chest','SpineMid')]

projection_matrix = np.array([2.790552,  0.,         0.,         0.,
                              0.,         1.5696855,  0.,         0.,
                              0.,         0.,        -1.0001999, -1.,
                              0.,         0.,        -0.20002,    0.], dtype=np.float32)
projection_matrix = projection_matrix.reshape((4, 4))

def rotation_quaternion_from_vectors(v_from, v_to):
    """
    Compute the quaternion that rotates vector v_from to vector v_to.
    Both vectors are assumed to be nonzero.
    """
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    dot = np.dot(v_from, v_to)
    
    if dot < -0.999999:
        orth = np.array([1, 0, 0])
        if np.abs(v_from[0]) > np.abs(v_from[1]):
            orth = np.array([0, 1, 0])
        axis = np.cross(v_from, orth)
        axis = axis / np.linalg.norm(axis)
        return pq.Quaternion(axis=axis, angle=np.pi)

    elif dot > 0.999999:
        return pq.Quaternion()
    else:
        axis = np.cross(v_from, v_to)
        angle = np.arccos(dot)
        return pq.Quaternion(axis=axis, angle=angle)

def project_points(points, view_matrix, proj_matrix):
    """
    Project 3D points (Nx3) to 2D using view and projection matrices.
    Converts points to homogeneous coordinates, applies the transforms,
    and then performs perspective division.
    """
    num_points = points.shape[0]
    points_h = np.hstack((points, np.ones((num_points, 1))))
    cam_points = points_h @ view_matrix
    proj_points = cam_points @ proj_matrix.T
    proj_points = proj_points / proj_points[:, 3:4]  # perspective division
    return proj_points[:, :2]

# ----- Preprocessing: Compute Bone Quaternions -----

# Use a canonical bone direction (rest pose) along the x-axis.
canonical_direction = np.array([1, 0, 0])

# Dictionaries to store computed bone quaternions.
quart_kp3d = {}  # Computed from kp3d (world coordinates)
quart_kp2d = {}  # Computed from projected 2D keypoints

# Project 3D keypoints to 2D.
proj_kp2d = project_points(kp3d, vm, projection_matrix)

# Compute bone quaternions for each connection.
for jointA, jointB in conns:
    try:
        idxA = joint_order.index(jointA)
        idxB = joint_order.index(jointB)
    except ValueError:
        print("Joint not found in joint_order:", jointA, jointB)
        continue

    # --- 3D Bone Quaternion ---
    bone_vec_3d = kp3d[idxB] - kp3d[idxA]
    if np.linalg.norm(bone_vec_3d) < 1e-6:
        q3d = pq.Quaternion()  # Identity if zero length.
    else:
        bone_dir_3d = bone_vec_3d / np.linalg.norm(bone_vec_3d)
        q3d = rotation_quaternion_from_vectors(canonical_direction, bone_dir_3d)
    quart_kp3d[(jointA, jointB)] = q3d

    # --- 2D Bone Quaternion ---
    bone_vec_2d = proj_kp2d[idxB] - proj_kp2d[idxA]
    if np.linalg.norm(bone_vec_2d) < 1e-6:
        q2d = pq.Quaternion()  # Identity if zero length.
    else:
        # Embed the 2D vector into 3D (z = 0).
        bone_vec_2d_3d = np.array([bone_vec_2d[0], bone_vec_2d[1], 0])
        bone_dir_2d = bone_vec_2d_3d / np.linalg.norm(bone_vec_2d_3d)
        q2d = rotation_quaternion_from_vectors(canonical_direction, bone_dir_2d)
    quart_kp2d[(jointA, jointB)] = q2d

# ----- Reconstruct Skeleton in Quaternion Space -----
# To ensure the quaternion skeleton has the same structure as the original,
# we reconstruct joint positions by integrating the bone displacements.
# Each bone displacement is computed as:
#   bone_disp = (bone_quaternion.rotate(canonical_direction)) * bone_length
#
# We'll define chains that mirror the original connections.

# Define chains using the original conns structure.
chains = {
    'head': [('Head','Neck'), ('Neck','Chest')],
    'left_arm': [('Chest','LeftShoulder'),
                 ('LeftShoulder','LeftArm'),
                 ('LeftArm','LeftForearm'),
                 ('LeftForearm','LeftHand'),
                 ('LeftHand','LeftFinger'),
                 ('LeftFinger','LeftFingerEnd')],
    'right_arm': [('Chest','RightShoulder'),
                  ('RightShoulder','RightArm'),
                  ('RightArm','RightForearm'),
                  ('RightForearm','RightHand'),
                  ('RightHand','RightFinger'),
                  ('RightFinger','RightFingerEnd')],
    'left_leg': [('Hips','LeftThigh'),
                 ('LeftThigh','LeftLeg'),
                 ('LeftLeg','LeftFoot'),
                 ('LeftFoot','LeftHeel'),
                 ('LeftHeel','LeftToe'),
                 ('LeftToe','LeftToeEnd')],
    'right_leg': [('Hips','RightThigh'),
                  ('RightThigh','RightLeg'),
                  ('RightLeg','RightFoot'),
                  ('RightFoot','RightHeel'),
                  ('RightHeel','RightToe'),
                  ('RightToe','RightToeEnd')],
    'spine': [('Chest','SpineMid'),
              ('SpineMid','SpineLow'),
              ('SpineLow','Hips')],
    'head_end': [('Head','HeadEnd')]
}

def reconstruct_chain(chain, kp3d, joint_order, quart_dict, canonical):
    """
    Reconstruct joint positions in quaternion space for a chain.
    Start from the root joint (first joint of the first connection in the chain)
    and integrate bone displacements computed from the bone quaternion and bone length.
    """
    # The starting joint is the first joint in the first connection.
    start_joint = chain[0][0]
    start_idx = joint_order.index(start_joint)
    positions = [kp3d[start_idx].copy()]
    
    for (jA, jB) in chain:
        idxA = joint_order.index(jA)
        idxB = joint_order.index(jB)
        # Bone length from the original skeleton.
        bone_length = np.linalg.norm(kp3d[idxB] - kp3d[idxA])
        # Rotated canonical vector gives the bone direction in quaternion space.
        bone_dir = quart_dict[(jA, jB)].rotate(canonical)
        new_pos = positions[-1] + bone_dir * bone_length
        positions.append(new_pos)
    return np.array(positions)

# Reconstruct each chain in quaternion space.
qskeleton = {}  # Dictionary to hold reconstructed joint positions per chain.
for chain_name, chain in chains.items():
    qskeleton[chain_name] = reconstruct_chain(chain, kp3d, joint_order, quart_kp3d, canonical_direction)

# ----- Visualization -----
fig = plt.figure(figsize=(16, 12))

# Subplot 1: Original 3D Skeleton (World Coordinates)
ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(kp3d[:, 0], kp3d[:, 1], kp3d[:, 2], color='k', s=50, label='Keypoints')
for j1, j2 in conns:
    if j1 in joint_order and j2 in joint_order:
        i1, i2 = joint_order.index(j1), joint_order.index(j2)
        ax1.plot([kp3d[i1, 0], kp3d[i2, 0]],
                 [kp3d[i1, 1], kp3d[i2, 1]],
                 [kp3d[i1, 2], kp3d[i2, 2]], color='blue')
ax1.set_title("3D Skeleton (World Coordinates)")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.legend()

# Subplot 2: Reconstructed Quaternion Skeleton (from 3D bone quaternions)
ax2 = fig.add_subplot(222, projection='3d')
colors = {'head': 'magenta', 'left_arm': 'red', 'right_arm': 'orange',
          'left_leg': 'green', 'right_leg': 'blue', 'spine': 'brown', 'head_end': 'purple'}
for chain_name, joints in qskeleton.items():
    ax2.plot(joints[:, 0], joints[:, 1], joints[:, 2], '-o',
             color=colors.get(chain_name, 'black'),
             label=chain_name)
ax2.set_title("Quaternion Skeleton (Reconstructed)")
ax2.set_xlabel("X'")
ax2.set_ylabel("Y'")
ax2.set_zlabel("Z'")
ax2.legend()

# Subplot 3: Quaternion Skeleton for 2D Bones (using 2D bone quaternions)
# We reconstruct a similar structure for the 2D projection.
def reconstruct_chain_2d(chain, kp3d, joint_order, quart_dict, canonical):
    # For 2D, use the projected kp2d positions embedded in 3D (with z=0) as starting points.
    start_joint = chain[0][0]
    start_idx = joint_order.index(start_joint)
    start_2d = np.array([proj_kp2d[start_idx][0], proj_kp2d[start_idx][1], 0])
    positions = [start_2d]
    for (jA, jB) in chain:
        idxA = joint_order.index(jA)
        idxB = joint_order.index(jB)
        # Use the original 3D bone length (or recompute from projected positions if desired).
        bone_length = np.linalg.norm(kp3d[idxB] - kp3d[idxA])
        bone_dir = quart_dict[(jA, jB)].rotate(canonical)
        new_pos = positions[-1] + bone_dir * bone_length
        positions.append(new_pos)
    return np.array(positions)

qskeleton_2d = {}
for chain_name, chain in chains.items():
    qskeleton_2d[chain_name] = reconstruct_chain_2d(chain, kp3d, joint_order, quart_kp2d, canonical_direction)

ax3 = fig.add_subplot(223, projection='3d')
for chain_name, joints in qskeleton_2d.items():
    ax3.plot(joints[:, 0], joints[:, 1], joints[:, 2], '-o',
             color=colors.get(chain_name, 'black'),
             label=chain_name)
ax3.set_title("Quaternion Skeleton (2D Projection Reconstructed)")
ax3.set_xlabel("X'")
ax3.set_ylabel("Y'")
ax3.set_zlabel("Z'")
ax3.legend()

plt.tight_layout()
plt.show()
