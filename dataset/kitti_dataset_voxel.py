import os
import os.path
import yaml
import math
import numpy as np
from numpy.core.defchararray import join

import json
import sys
import pickle
import glob
import time
import scipy.misc
import matplotlib.pyplot as plt
import open3d as o3d
from open3d import geometry, utility

CMAP = 'plasma'
np.set_printoptions(threshold=1e10)


def aug_matrix():
    anglex = np.clip(0.01 * np.random.randn(), -0.02, 0.02).astype(np.float32) * np.pi / 4.0
    angley = np.clip(0.01 * np.random.randn(), -0.02, 0.02).astype(np.float32) * np.pi / 4.0
    anglez = np.clip(0.05 * np.random.randn(), -0.1, 0.1).astype(np.float32) * np.pi / 4.0

    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)

    Rx = np.array([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]])

    scale = np.diag(np.random.uniform(1.00, 1.00, 3).astype(np.float32))
    R_trans = Rx.dot(Ry).dot(Rz).dot(scale.T)
    # R_trans = Rx.dot(Ry).dot(Rz)

    xx = np.clip(0.5 * np.random.randn(), -1.0, 1.0).astype(np.float32)
    yy = np.clip(0.1 * np.random.randn(), -0.2, 0.2).astype(np.float32)
    zz = np.clip(0.05 * np.random.randn(), -0.15, 0.15).astype(np.float32)

    add_xyz = np.array([[xx], [yy], [zz]])

    T_trans = np.concatenate([R_trans, add_xyz], axis=-1)
    filler = np.array([0.0, 0.0, 0.0, 1.0])
    filler = np.expand_dims(filler, axis=0)  ##1*4
    T_trans = np.concatenate([T_trans, filler], axis=0)  # 4*4

    return T_trans


def cartesian(points, normal=None, nvoxel=8192, priority='ucdduz'):
    ximin, xrate, ximax = -60, 0.2, 60
    yimin, yrate, yimax = -40, 0.2, 40
    zimin, zrate, zimax = -2, 0.2, 4

    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]
    depth = np.linalg.norm(points[:, :2], 2, axis=1)
    xmax = int((ximax - ximin) / xrate)
    ymax = int((yimax - yimin) / yrate)
    zmax = int((zimax - zimin) / zrate)
    proj_x = np.floor((scan_x - ximin) / xrate).astype(np.int32)
    proj_y = np.floor((scan_y - yimin) / yrate).astype(np.int32)
    proj_z = np.floor((scan_z - zimin) / zrate).astype(np.int32)
    xcheck = np.logical_and(0 <= proj_x, proj_x < xmax)
    ycheck = np.logical_and(0 <= proj_y, proj_y < ymax)
    zcheck = np.logical_and(0 <= proj_z, proj_z < zmax)
    check = np.logical_and(np.logical_and(xcheck, ycheck), zcheck)
    proj_x = proj_x[check]
    proj_y = proj_y[check]
    proj_z = proj_z[check]
    depth = depth[check]  #
    points = points[check]
    normal = normal[check]

    voxelall = proj_z * 1000000 + proj_y * 1000 + proj_x
    pointsid = np.lexsort((depth, voxelall))
    sortdata = np.concatenate((points, normal, voxelall[:, None]), axis=1)[pointsid]
    id = np.where(np.diff(sortdata[:, -1]))[0]
    sortlist = np.split(sortdata, id + 1)

    value, firstindex, count = np.unique(voxelall[pointsid], return_index=True, return_counts=True)

    voxel_z = proj_z[firstindex]
    depthchoose = np.round(depth[pointsid][firstindex]).astype(np.int32)
    if priority == "ucdduz":  # 1
        voxelchoose = np.argsort(count * 10000 + (100 - depthchoose) * 100 + voxel_z)[::-1][:nvoxel]
    elif priority == "ucuzdd":  # 2
        voxelchoose = np.argsort(count * 10000 + voxel_z * 100 + (100 - depthchoose))[::-1][:nvoxel]
    else:
        raise NotImplementedError()

    voxelmsg = []
    for i in range(voxelchoose.shape[0]):
        nowpointsmsg = sortlist[voxelchoose[i]]
        cendata = np.mean(nowpointsmsg[:, :6], axis=0)
        vercen = cendata[:3]
        normal = cendata[3:6]
        voxelmsg.append(np.concatenate([vercen, normal], axis=0))
    voxelmsg = np.array(voxelmsg)
    voxelsize = voxelmsg.shape[0]
    # print(voxelsize)

    if voxelsize < nvoxel:
        need = nvoxel - voxelsize
        repeat = voxelmsg[np.random.choice(voxelsize, need, replace=True)]
        voxelmsg = np.vstack([voxelmsg, repeat])
        pass

    return voxelmsg, voxelsize


class OdometryDataset():
    def __init__(self, is_training=0, data_root='/tmp/data_odometry_velodyne/dataset', num_points=8192,
                 data_dir_list: list = [0, 1], remove_ground=True):

        self.is_training = is_training
        self.npoints = num_points
        self.datapath = data_root
        self.remove_ground = remove_ground

        data_dir_list.sort()
        self.data_list = data_dir_list
        self.data_len_sequence = [4541, 1101, 4661, 801, 271, 2761, 1101, 1101, 4071, 1591, 1201, 921, 1061, 3281, 631,
                                  1901, 1731, 491, 1801, 4981, 831, 2721]

        # self.len_list = [0, 4541, 5642, 10303, 11104, 11375, 14136, 15237, 16338, 20409, 22000, 23201, \
        #     24122, 25183, 28464, 29095, 30996, 32727, 33218, 35019, 40000, 40831, 43552]           
        # self.file_map = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', \
        #     '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']

        Tr_tmp = []
        data_sum = [0]
        vel_to_cam_Tr = []

        with open('ground_truth_pose/calib.yaml', "r") as f:
            con = yaml.load(f, Loader=yaml.FullLoader)
        for i in range(22):
            vel_to_cam_Tr.append(np.array(con['Tr{}'.format(i)]))
        for i in self.data_list:
            data_sum.append(data_sum[-1] + self.data_len_sequence[i])
            Tr_tmp.append(vel_to_cam_Tr[i])

        self.Tr_list = Tr_tmp
        self.data_sum = data_sum

    def __len__(self):
        return self.data_sum[-1]

    def __getitem__(self, index):

        sequence_str_list = []

        for item in self.data_list:
            sequence_str_list.append('{:02d}'.format(item))

        if index in self.data_sum:
            index_index = self.data_sum.index(index)
            index_ = 0
            fn1 = index_
            fn2 = index_
            sample_id = index_

        else:
            index_index, data_begin, data_end = self.get_index(index, self.data_sum)
            index_ = index - data_begin
            fn1 = index_ - 1
            fn2 = index_
            sample_id = index_

        cur_lidar_dir = os.path.join(self.datapath, sequence_str_list[index_index])
        pc1_bin = os.path.join(cur_lidar_dir, 'velodyne/' + '{:06d}.bin'.format(fn1))
        pc2_bin = os.path.join(cur_lidar_dir, 'velodyne/' + '{:06d}.bin'.format(fn2))

        pose = np.load('ground_truth_pose/Pose_GT_loop/Pose_GT_Diff/' + sequence_str_list[index_index] + '_diff.npy')
        T_diff = pose[index_:index_ + 1, :]
        T_diff = T_diff.reshape(3, 4)
        filler = np.array([0.0, 0.0, 0.0, 1.0])
        filler = np.expand_dims(filler, axis=0)
        T_diff = np.concatenate([T_diff, filler], axis=0)
        # print(T_diff)

        Tr = self.Tr_list[index_index]
        Tr_inv = np.linalg.inv(Tr)
        T_gt = np.matmul(Tr_inv, T_diff)
        T_gt = np.matmul(T_gt, Tr)
        # print(T_gt)

        point1 = np.fromfile(pc1_bin, dtype=np.float32).reshape(-1, 4)
        point2 = np.fromfile(pc2_bin, dtype=np.float32).reshape(-1, 4)
        pos1_raw = point1[:, :3].astype(np.float32)
        pos2_raw = point2[:, :3].astype(np.float32)

        if self.remove_ground:
            pos1_raw = pos1_raw[pos1_raw[:, 2] > -1.3]  # ground 1.4
            pos2_raw = pos2_raw[pos2_raw[:, 2] > -1.3]

        voxelpc1, voxellen1 = cartesian(pos1_raw, normal=pos1_raw, nvoxel=self.npoints)
        voxelpc2, voxellen2 = cartesian(pos2_raw, normal=pos2_raw, nvoxel=self.npoints)

        voxel_xyz1, voxel_nor1 = voxelpc1[:, :3], voxelpc1[:, 3:]
        voxel_xyz2, voxel_nor2 = voxelpc2[:, :3], voxelpc2[:, 3:]

        if self.is_training:
            T_trans = aug_matrix()
        else:
            T_trans = np.eye(4).astype(np.float32)

        T_trans_inv = np.linalg.inv(T_trans)

        if self.is_training:
            add_T = np.ones((self.npoints, 1))
            pos2_trans = np.concatenate([voxel_xyz2, add_T], axis=-1)
            pos2_trans = np.matmul(T_trans, pos2_trans.T)
            voxel_xyz2 = pos2_trans.T[:, :3]

            normal_trans = np.matmul(T_trans[:3, :3], voxel_nor2.T)
            voxel_nor2 = normal_trans.T
            T_gt = np.matmul(T_gt, T_trans_inv)

        R_gt = T_gt[:3, :3]
        q_gt = rot2quat(R_gt)
        t_gt = T_gt[:3, 3:]

        return voxel_xyz2, voxel_nor2, voxel_xyz1, voxel_nor1, T_gt, q_gt, t_gt, Tr

    def get_index(self, value, mylist):
        mylist.sort()
        for i, num in enumerate(mylist):
            if num > value:
                return i - 1, mylist[i - 1], num


def rot2quat(R):
    assert R.shape == (3, 3)
    trace = np.trace(R)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return np.array([w, x, y, z])


if __name__ == '__main__':
    root_dir = './KITTI/sequences'
    dataset = OdometryDataset(data_root=root_dir, data_dir_list=[7], is_training=1)
    print(len(dataset))
