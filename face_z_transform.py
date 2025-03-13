'''
This file is used to transform all smpl to z+ direction.
'''

import numpy as np
from utils.face_z_align_util import *
import os
import torch
from tqdm import tqdm
import argparse

def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path


def my_quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

    
def calc_heading(q):
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 2] = 1
    rot_dir = my_quat_rotate(q, ref_dir)
    heading = torch.atan2(rot_dir[..., 0], rot_dir[..., 2])
    return heading


def calc_heading_quat_inv(q):
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 1] = 1
    return -heading, axis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--filedir', type=str, required=True, help='Input directory path')
    args = parser.parse_args()

    bad_cnt = 0
    for data_path in tqdm(findAllFile(os.path.join(args.filedir, 'smpl_85'))):
        smpl_data = np.load(data_path)
        seq_len = smpl_data.shape[0]
        if seq_len > 0:
            pose_body = smpl_data[:, :72].reshape(seq_len, -1, 3)
        else:  
            bad_cnt += 1
            continue

        trans= smpl_data[:, 72:75]
        beta = smpl_data[:, 75:]
        root_first_frame_root_orient = pose_body[0,0]
        root_first_frame_root_orient_quat = expmap_to_quaternion(root_first_frame_root_orient)
        root_first_frame_root_orient_quat_xyzw = root_first_frame_root_orient_quat[[1, 2, 3, 0]]
        root_first_frame_root_orient_quat_xyzw = torch.from_numpy(root_first_frame_root_orient_quat_xyzw).float().unsqueeze(0)
        heading_inv, axis = calc_heading_quat_inv(root_first_frame_root_orient_quat_xyzw)
        heading_inv_axis_angle = heading_inv * axis
        heading_inv_axis_angle = heading_inv_axis_angle.numpy()
        q_diff = expmap_to_quaternion(heading_inv_axis_angle)
        result_root_orient_quaternion = qmul_np(q_diff.reshape(1, -1).repeat(seq_len, axis=0), expmap_to_quaternion(pose_body[:,0]))
        result_root_orient_axis_angle = quaternion_to_axis_angle(torch.from_numpy(result_root_orient_quaternion)).numpy()

        trans = qrot_np(q_diff.reshape(1, -1).repeat(seq_len, axis=0), trans)
        result_pose_body = np.concatenate([result_root_orient_axis_angle, pose_body[:,1:].reshape(seq_len, -1), trans, beta], axis=-1)
        output_path = data_path.replace('smpl_85', 'smpl_85_face_z_transform')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, result_pose_body)
    print(f"bad_cnt: {bad_cnt}")
    print(f"Processed files are saved in {args.filedir}/smpl_85_face_z_transform")
    


