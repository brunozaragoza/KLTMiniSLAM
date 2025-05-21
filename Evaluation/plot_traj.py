import argparse
import numpy as np
import matplotlib.pyplot as plt

def read_trajectory(file_path):
    """ Reads a trajectory file and extracts positions (tx, ty). """
    data = np.loadtxt(file_path, delimiter=',')
    timestamps = data[:, 0]  # First column is timestamp
    positions = data[:, 1:3]  # tx, ty (second and third columns)
    return timestamps, positions

def plot_trajectory(timestamps, positions, output_file=None):
    """ Plots the 2D trajectory (tx, ty). """
    plt.figure(figsize=(8, 6))
    plt.plot(positions[:, 0], positions[:, 1], 'b-', label='Trajectory')
    plt.scatter(positions[0, 0], positions[0, 1], c='g', marker='o', label='Start')
    plt.scatter(positions[-1, 0], positions[-1, 1], c='r', marker='x', label='End')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('2D Trajectory Plot')
    plt.legend()
    plt.grid()
    
    if output_file:
        plt.savefig(output_file, dpi=300)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot a 2D trajectory from a file.")
    parser.add_argument('trajectory_file', help="Path to the trajectory file")
    parser.add_argument('--output', help="Output file name (e.g., output.png)", default=None)
    args = parser.parse_args()

    timestamps, positions = read_trajectory(args.trajectory_file)
    plot_trajectory(timestamps, positions, args.output)