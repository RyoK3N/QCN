import numpy as np
import pyquaternion as pq
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ----------------------------
# Data Loading (Assumed to come from your data loader `dl`)
vm = dl.__getitem__(0)['vm']              # View matrix (4x4)
kp2d = dl.__getitem__(0)['kp2d_camera']    # 2D keypoints (Nx2)
kp3d = dl.__getitem__(0)['kp3d_world']       # 3D keypoints (Nx3)
joint_order = dl.joint_order              # List of joint names

# Define connectivity (each as (jointA, jointB))
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

# Projection matrix provided as a flat array; reshape it to (4,4)
projection_matrix = np.array([2.790552,  0.,         0.,         0.,
                              0.,         1.5696855,  0.,         0.,
                              0.,         0.,        -1.0001999, -1.,
                              0.,         0.,        -0.20002,    0.], dtype=np.float32)
projection_matrix = projection_matrix.reshape((4, 4))

# ----------------------------
# Helper Functions
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
        return pq.Quaternion()  # Identity quaternion
    else:
        axis = np.cross(v_from, v_to)
        angle = np.arccos(dot)
        return pq.Quaternion(axis=axis, angle=angle)

def project_points(points, view_matrix, proj_matrix):
    """
    Project 3D points (Nx3) to 2D using view and projection matrices.
    Converts points to homogeneous coordinates, applies the transforms,
    and performs perspective division.
    """
    num_points = points.shape[0]
    points_h = np.hstack((points, np.ones((num_points, 1))))
    cam_points = points_h @ view_matrix
    proj_points = cam_points @ proj_matrix.T
    proj_points = proj_points / proj_points[:, 3:4]
    return proj_points[:, :2]

# ----------------------------
# Preprocessing: Compute Bone Quaternions
# Use a canonical bone direction (rest pose) along the x-axis.
canonical_direction = np.array([1, 0, 0])

# Dictionaries to store bone quaternions.
quart_kp3d = {}  # From kp3d (world coordinates)
quart_kp2d = {}  # From projected 2D keypoints

# Project 3D keypoints to 2D.
proj_kp2d = project_points(kp3d, vm, projection_matrix)

# Compute quaternions for each connection.
for jointA, jointB in conns:
    try:
        idxA = joint_order.index(jointA)
        idxB = joint_order.index(jointB)
    except ValueError:
        print("Joint not found:", jointA, jointB)
        continue

    # 3D Bone Quaternion
    bone_vec_3d = kp3d[idxB] - kp3d[idxA]
    if np.linalg.norm(bone_vec_3d) < 1e-6:
        q3d = pq.Quaternion()
    else:
        bone_dir_3d = bone_vec_3d / np.linalg.norm(bone_vec_3d)
        q3d = rotation_quaternion_from_vectors(canonical_direction, bone_dir_3d)
    quart_kp3d[(jointA, jointB)] = q3d

    # 2D Bone Quaternion
    bone_vec_2d = proj_kp2d[idxB] - proj_kp2d[idxA]
    if np.linalg.norm(bone_vec_2d) < 1e-6:
        q2d = pq.Quaternion()
    else:
        bone_vec_2d_3d = np.array([bone_vec_2d[0], bone_vec_2d[1], 0])
        bone_dir_2d = bone_vec_2d_3d / np.linalg.norm(bone_vec_2d_3d)
        q2d = rotation_quaternion_from_vectors(canonical_direction, bone_dir_2d)
    quart_kp2d[(jointA, jointB)] = q2d

# ----------------------------
# Reconstruct Skeleton in Quaternion Space
# To preserve the exact structure, we reconstruct joint positions by integrating the bone displacements.
# Each bone displacement is: (bone_quaternion.rotate(canonical_direction)) * bone_length.
# Define chains mirroring the original connectivity.
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
    Reconstruct joint positions in quaternion space along a chain.
    Start at the first joint of the chain and integrate bone displacements.
    """
    start_joint = chain[0][0]
    start_idx = joint_order.index(start_joint)
    positions = [kp3d[start_idx].copy()]
    for (jA, jB) in chain:
        idxA = joint_order.index(jA)
        idxB = joint_order.index(jB)
        bone_length = np.linalg.norm(kp3d[idxB] - kp3d[idxA])
        bone_disp = quart_dict[(jA, jB)].rotate(canonical) * bone_length
        new_pos = positions[-1] + bone_disp
        positions.append(new_pos)
    return np.array(positions)

# Reconstruct the quaternion skeleton from 3D bone quaternions.
qskeleton = {}
for chain_name, chain in chains.items():
    qskeleton[chain_name] = reconstruct_chain(chain, kp3d, joint_order, quart_kp3d, canonical_direction)

# ----------------------------
# Compute Skeleton Bounds and Camera Parameters
margin = 20
x_min, x_max = np.min(kp3d[:,0]) - margin, np.max(kp3d[:,0]) + margin
y_min, y_max = np.min(kp3d[:,1]) - margin, np.max(kp3d[:,1]) + margin
z_min, z_max = np.min(kp3d[:,2]) - margin, np.max(kp3d[:,2]) + margin

center = np.mean(kp3d, axis=0)
default_cam = np.array([200, 100, 0])  # default camera offset

# ----------------------------
# Set up Figure and Axes for Animation
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')  # Original 3D Skeleton
ax2 = fig.add_subplot(122, projection='3d')  # Reconstructed Quaternion Skeleton

def plot_original(ax):
    """Plot the original 3D skeleton (world coordinates)."""
    ax.clear()
    ax.scatter(kp3d[:, 0], kp3d[:, 1], kp3d[:, 2], color='k', s=50)
    for j1, j2 in conns:
        if j1 in joint_order and j2 in joint_order:
            i1, i2 = joint_order.index(j1), joint_order.index(j2)
            ax.plot([kp3d[i1, 0], kp3d[i2, 0]],
                    [kp3d[i1, 1], kp3d[i2, 1]],
                    [kp3d[i1, 2], kp3d[i2, 2]], color='blue')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Skeleton (World Coordinates)")

def plot_quaternion(ax):
    """Plot the reconstructed quaternion skeleton."""
    ax.clear()
    colors = {'head': 'magenta', 'left_arm': 'red', 'right_arm': 'orange',
              'left_leg': 'green', 'right_leg': 'blue', 'spine': 'brown', 'head_end': 'purple'}
    for chain_name, joints in qskeleton.items():
        ax.plot(joints[:, 0], joints[:, 1], joints[:, 2], '-o',
                color=colors.get(chain_name, 'black'),
                label=chain_name)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X'")
    ax.set_ylabel("Y'")
    ax.set_zlabel("Z'")
    ax.set_title("Quaternion Skeleton (Reconstructed)")
    ax.legend()

# ----------------------------
# Update function for animation.
def update(frame):
    angle_rad = np.deg2rad(frame)
    # Create a rotation quaternion about the Y-axis.
    q_rot = pq.Quaternion(axis=[0, 1, 0], angle=angle_rad)
    cam_pos = center + q_rot.rotate(default_cam)
    rel = cam_pos - center
    r = np.linalg.norm(rel)
    elev = np.degrees(np.arcsin(rel[1] / r))
    azim = np.degrees(np.arctan2(rel[0], rel[2]))
    ax1.view_init(elev=elev, azim=azim)
    ax2.view_init(elev=elev, azim=azim)
    plot_original(ax1)
    plot_quaternion(ax2)
    return fig,

# Create animation: rotate the camera from 0 to 360 degrees.
ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), interval=50, blit=False)

# Save the animation as an MP4 file (requires ffmpeg installed).
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Your Name'), bitrate=1800)
ani.save("skeleton_animation.mp4", writer=writer)

plt.show()
