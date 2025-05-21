import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion (qx, qy, qz, qw) to a 3x3 rotation matrix.
    The quaternion is assumed to be in the form [qx, qy, qz, qw].
    """
    qx, qy, qz, qw = q
    # Normalize the quaternion (if not already unit norm)
    norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx**2 + qy**2)]
    ])
    return R

# Load the pose data from a file
poses = []
filename = 'merged.txt'  # make sure this file contains your data lines

with open(filename, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  # skip empty lines
        parts = line.split(',')
        # Convert string parts to floats
        parts = [float(x) for x in parts]
        timestamp = parts[0]
        tx, ty, tz = parts[1:4]
        qx, qy, qz, qw = parts[4:8]
        poses.append({
            'timestamp': timestamp,
            't': np.array([tx, ty, tz]),
            'q': np.array([qx, qy, qz, qw])
        })

# Extract all translations for trajectory plotting
positions = np.array([pose['t'] for pose in poses])

# Create a 3D plot for the trajectory
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')


# Optionally, plot coordinate axes at each pose to show orientation
# (Adjust axis_length as needed for visibility)
axis_length = 0.005
for pose in poses:
    t = pose['t']
    q = pose['q']
    R = quaternion_to_rotation_matrix(q)
    
    # Define unit vectors in x, y, z directions and scale them
    x_axis = R @ np.array([1, 0, 0]) * axis_length
    y_axis = R @ np.array([0, 1, 0]) * axis_length
    z_axis = R @ np.array([0, 0, 1]) * axis_length
    
    # Plot arrows using quiver
    ax.quiver(t[0], t[1], t[2],
              x_axis[0], x_axis[1], x_axis[2],
              color='r', length=axis_length, normalize=True)
    ax.quiver(t[0], t[1], t[2],
              y_axis[0], y_axis[1], y_axis[2],
              color='g', length=axis_length, normalize=True)
    ax.quiver(t[0], t[1], t[2],
              z_axis[0], z_axis[1], z_axis[2],
              color='b', length=axis_length, normalize=True)

# Label axes and set title
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Map Initial Pose for different executions')
ax.legend()

plt.show()

