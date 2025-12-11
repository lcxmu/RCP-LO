import numpy as np
import os



def load_kitti_poses(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)
            poses.append(pose)
    return np.array(poses)

def compute_relative_poses(poses):
    relative_poses = []
    for i in range(len(poses)):
        if i == 0:
            relative_poses.append(poses[i])
        else:

            T_prev = np.vstack([poses[i - 1], [0, 0, 0, 1]]) 
            T_curr = np.vstack([poses[i], [0, 0, 0, 1]])    
            T_relative = np.matmul(np.linalg.inv(T_prev), T_curr)
            relative_poses.append(T_relative[:3, :])  
    return np.array(relative_poses)


kitti_pose_file = './21.txt'  
output_file = './Pose_GT_Diff/21_diff.npy'
poses = load_kitti_poses(kitti_pose_file)
relative_poses = compute_relative_poses(poses)

np.save(output_file, relative_poses)
print(f"Relative poses saved to {output_file} in binary format.")

#np.savetxt(output_file, relative_poses.reshape(len(relative_poses), -1), fmt='%.6f')
#print(f"Relative poses saved to {output_file}")
