# ---------------------------
# Import required Julia packages
using PyCall
using LinearAlgebra
using Rotations
using PyPlot

# ---------------------------
# Import the Python dataloader and initialize it.
# (Make sure that the Python module "dataloader" is in your Python path.)
PoseDataLoader = pyimport("dataloader").PoseDataLoader

data_path = "./data/train_data_20.json"
val_data = "./data/val_data_20.json"

dl = PoseDataLoader(data_path, val_data)

# ---------------------------
# Extract data from the first sample.
sample = dl.__getitem__(0)
vm = sample["vm"]                  # view matrix (not used in this plotting code)
kp2d = sample["kp2d_camera"]         # 2D keypoints (not used here)
kp3d_py = sample["kp3d_world"]       # 3D keypoints from Python (as a numpy array)
kp3d = Array(kp3d_py)                # convert to a Julia Array{T,2}
joint_order_py = dl.joint_order      # Python list of joint names
joint_order = collect(joint_order_py)  # Convert to a Julia array of strings

# ---------------------------
# Define skeleton connectivity as a list of (joint1, joint2) pairs.
conns = [
    ("Head","Neck"), ("Neck","Chest"), ("Chest","LeftShoulder"), ("LeftShoulder","LeftArm"),
    ("LeftArm","LeftForearm"), ("LeftForearm","LeftHand"), ("Chest","RightShoulder"),
    ("RightShoulder","RightArm"), ("RightArm","RightForearm"), ("RightForearm","RightHand"),
    ("Hips","LeftThigh"), ("LeftThigh","LeftLeg"), ("LeftLeg","LeftFoot"),
    ("Hips","RightThigh"), ("RightThigh","RightLeg"), ("RightLeg","RightFoot"),
    ("RightHand","RightFinger"), ("RightFinger","RightFingerEnd"),
    ("LeftHand","LeftFinger"), ("LeftFinger","LeftFingerEnd"),
    ("Head","HeadEnd"), ("RightFoot","RightHeel"), ("RightHeel","RightToe"),
    ("RightToe","RightToeEnd"), ("LeftFoot","LeftHeel"), ("LeftHeel","LeftToe"),
    ("LeftToe","LeftToeEnd"), ("SpineLow","Hips"), ("SpineMid","SpineLow"),
    ("Chest","SpineMid")
]

# ---------------------------
# Set a default limb (canonical) direction.
default_vector = [0.0, 1.0, 0.0]

# ---------------------------
# Helper function: Compute a quaternion that rotates v0 to v1.
function quaternion_from_vectors(v0::Vector{<:Real}, v1::Vector{<:Real})
    v0_norm = v0 / norm(v0)
    v1_norm = v1 / norm(v1)
    dot_val = dot(v0_norm, v1_norm)
    if dot_val ≥ 0.9999
        return UnitQuaternion(1.0, 0.0, 0.0, 0.0)  # Identity quaternion
    elseif dot_val ≤ -0.9999
        # Choose an arbitrary perpendicular axis.
        orthog = [1.0, 0.0, 0.0]
        if abs(v0_norm[1]) > 0.9
            orthog = [0.0, 1.0, 0.0]
        end
        axis = cross(v0_norm, orthog)
        axis = axis / norm(axis)
        return UnitQuaternion(AngleAxis(pi, axis))
    else
        axis = cross(v0_norm, v1_norm)
        angle = acos(dot_val)
        return UnitQuaternion(AngleAxis(angle, axis))
    end
end

# ---------------------------
# Compute limb quaternions for each connection.
# Create a dictionary mapping (joint1, joint2) to the corresponding UnitQuaternion.
limb_quaternions = Dict{Tuple{String,String}, UnitQuaternion{Float64}}()
for (j1, j2) in conns
    if (j1 in joint_order) && (j2 in joint_order)
        i1 = findfirst(x -> x == j1, joint_order)
        i2 = findfirst(x -> x == j2, joint_order)
        limb_vector = kp3d[i2, :] - kp3d[i1, :]
        if norm(limb_vector) < 1e-6
            q = UnitQuaternion(1.0, 0.0, 0.0, 0.0)
        else
            q = quaternion_from_vectors(default_vector, limb_vector)
        end
        limb_quaternions[(j1, j2)] = q
    end
end

# ---------------------------
# Plotting: Create a figure with 4 subplots.

fig = figure(figsize=(16, 12))

# Subplot 1: 3D Skeleton with connections.
ax1 = fig.add_subplot(221, projection="3d")
# Plot keypoints (all rows, columns 1,2,3)
ax1.scatter(kp3d[:,1], kp3d[:,2], kp3d[:,3], color="k", s=50, label="Keypoints")
for (j1, j2) in conns
    if (j1 in joint_order) && (j2 in joint_order)
        i1 = findfirst(x -> x == j1, joint_order)
        i2 = findfirst(x -> x == j2, joint_order)
        ax1.plot([kp3d[i1,1], kp3d[i2,1]],
                 [kp3d[i1,2], kp3d[i2,2]],
                 [kp3d[i1,3], kp3d[i2,3]], color="blue")
    end
end
ax1.set_title("3D Skeleton")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.legend()

# Subplot 2: 3D Skeleton with limb orientation arrows.
ax2 = fig.add_subplot(222, projection="3d")
ax2.scatter(kp3d[:,1], kp3d[:,2], kp3d[:,3], color="k", s=50, label="Keypoints")
arrow_length = 5.0
for ((j1, j2), q) in limb_quaternions
    i1 = findfirst(x -> x == j1, joint_order)
    rotated_dir = q * default_vector  # rotate default_vector by the quaternion
    ax2.quiver(kp3d[i1,1], kp3d[i1,2], kp3d[i1,3],
               rotated_dir[1], rotated_dir[2], rotated_dir[3],
               length=arrow_length, color="red")
    i2 = findfirst(x -> x == j2, joint_order)
    ax2.plot([kp3d[i1,1], kp3d[i2,1]],
             [kp3d[i1,2], kp3d[i2,2]],
             [kp3d[i1,3], kp3d[i2,3]], color="gray", linestyle="--")
end
ax2.set_title("Limb Orientations (via Quaternions)")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")
ax2.legend()

# Subplot 3: Rotation angles of each limb quaternion.
ax3 = fig.add_subplot(223)
angles = Float64[]
labels = String[]
for ((j1, j2), q) in limb_quaternions
    push!(angles, angle(q))  # rotation angle in radians
    push!(labels, string(j1, "->", j2))
end
ax3.plot(1:length(angles), angles, marker="o", linestyle="-")
ax3.set_title("Limb Quaternion Rotation Angles")
ax3.set_xlabel("Connection Index")
ax3.set_ylabel("Angle (rad)")
ax3.set_xticks(1:length(angles))
ax3.set_xticklabels(labels, rotation=90, fontsize=8)

# Subplot 4: SLERP interpolation for the first connection.
first_conn = first(collect(keys(limb_quaternions)))
q_start = UnitQuaternion(1.0, 0.0, 0.0, 0.0)
q_end = limb_quaternions[first_conn]
n_interp = 10
interp_quats = [slerp(q_start, q_end, i/(n_interp-1)) for i in 0:(n_interp-1)]
i_base = findfirst(x -> x == first_conn[1], joint_order)
ax4 = fig.add_subplot(224, projection="3d")
ax4.scatter(kp3d[i_base,1], kp3d[i_base,2], kp3d[i_base,3], color="k", s=50, label="Base Joint")
for iq in interp_quats
    rotated = iq * default_vector
    ax4.quiver(kp3d[i_base,1], kp3d[i_base,2], kp3d[i_base,3],
               rotated[1], rotated[2], rotated[3],
               length=arrow_length, color="magenta")
end
ax4.set_title("SLERP Interpolation for $(first_conn[1])->$(first_conn[2])")
ax4.set_xlabel("X")
ax4.set_ylabel("Y")
ax4.set_zlabel("Z")
ax4.legend()

tight_layout()
show()
