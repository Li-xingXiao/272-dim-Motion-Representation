# representation: 272 dim
# :2 local xz velocities of root, no heading, can recover translation
# 2:8  heading angular velocities, 6d rotation, can recover heading
# 8:8+3*njoint local position, no heading, all at xz origin
# 8+3*njoint:8+6*njoint local velocities, no heading, all at xz origin, can recover local postion
# 8+6*njoint:8+12*njoint local rotations, 6d rotation, no heading, all frames z+

import numpy as np
from utils.face_z_align_util import expmap_to_quaternion, quaternion_to_matrix, quaternion_to_matrix_np, matrix_to_rotation_6d, qrot_np, rotation_6d_to_matrix, matrix_to_axis_angle
import copy
import torch
import scipy.ndimage as ndimage
from tqdm import tqdm
import os
import argparse

def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path

def rot_yaw(yaw):
    cs = np.cos(yaw)
    sn = np.sin(yaw)
    return np.array([[cs,0,sn],[0,1,0],[-sn,0,cs]])

def foot_detect(global_positions, thres):
    """
        derived from https://github.com/orangeduck/Motion-Matching/blob/37df18afc44e8acca3af5e85dff96effa6a34b03/resources/generate_database.py#L160
    """
    left_foot = 10
    right_foot = 11
    global_velocities = global_positions[1:] - global_positions[:-1]
    contact_velocities = np.sqrt(np.sum(global_velocities[:, np.array([left_foot, right_foot])]**2, axis=-1))
    contacts = contact_velocities < thres
    # Median filter here acts as a kind of "majority vote", and removes
    # small regions  where contact is either active or inactive
    for ci in range(contacts.shape[1]):
        contacts[:,ci] = ndimage.median_filter(
            contacts[:,ci], 
            size=6, 
            mode='nearest')
    return contacts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--filedir', type=str, required=True, help='Input directory path')
    args = parser.parse_args()

    bad_cnt = 0
    for file in tqdm(findAllFile(os.path.join(args.filedir, 'smpl_85_face_z_transform_joints'))):
        output_file = file.replace('smpl_85_face_z_transform_joints', 'Representation_272')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        root_idx = 0
        # get joint positions
        position_data = np.load(file)
        position_data = position_data[:, :22, :3]
        nfrm, njoint, _ = position_data.shape
        # get smpl rotations
        rotation_smpl_axis_angle = np.load(file.replace('smpl_85_face_z_transform_joints', 'smpl_85_face_z_transform'))
        rotations_wxyz = expmap_to_quaternion(rotation_smpl_axis_angle[:, :66].reshape(nfrm, njoint, 3))
        
        rotations_matrix = quaternion_to_matrix_np(rotations_wxyz)  # nframe, njoint, 3, 3

        # put on floor and put root on origin for the first frame
        ori = copy.deepcopy(position_data[0,root_idx]) # first frame root position
        y_min = np.min(position_data[:,:,1])
        ori[1] = y_min
        position_data = position_data - ori
        velocities_root = position_data[1:,root_idx,:] - position_data[:-1,root_idx,:]

        # smpl unit is m and 0.15 is given as cm, may need to change depending on the datasets
        contacts = foot_detect(position_data, 0.15/100)
        
        # calculate local position, all frames on xz origin
        position_data[:,:,0] -= position_data[:,0:1,0]
        position_data[:,:,2] -= position_data[:,0:1,2]

        # calculate heading
        global_heading = - np.arctan2(rotations_matrix[:,root_idx,0,2], rotations_matrix[:, root_idx, 2,2])
        global_heading_rot = np.array([rot_yaw(x) for x in global_heading])
        global_heading_diff = global_heading[1:] - global_heading[:-1]
        global_heading_diff_rot = np.array([rot_yaw(x) for x in global_heading_diff])

        # calculate positions no heading
        positions_no_heading = np.matmul(np.repeat(global_heading_rot[:, None,:, :], njoint, axis=1), position_data[...,None]).squeeze(-1)

        # calculate velocity no heading
        velocities_no_heading = positions_no_heading[1:] - positions_no_heading[:-1]

        # calculate root velocity_xz_no_heading
        velocities_root_xy_no_heading = np.matmul(global_heading_rot[:-1], velocities_root[:, :, None]).squeeze()[...,[0,2]]

        # calculate rotations no heading
        rotations_matrix[:,0,...] = np.matmul(global_heading_rot, rotations_matrix[:,0,...]) 

        # concat all
        size_frame = 8+njoint*3+njoint*3+njoint*6
        final_x = np.zeros((nfrm, size_frame))

        # set the first frame of the root rotation to identity
        final_x[0, 2] = 1
        final_x[0, 6] = 1
        try:
            final_x[1:,2:8] = matrix_to_rotation_6d(torch.from_numpy(global_heading_diff_rot)).numpy() # take 6D rotation
        except:
            bad_cnt += 1
            continue
        final_x[1:,:2] = velocities_root_xy_no_heading 
        final_x[:,8:8+3*njoint] = np.reshape(positions_no_heading, (nfrm,-1))
        final_x[1:,8+3*njoint:8+6*njoint] = np.reshape(velocities_no_heading, (nfrm-1,-1))
        final_x[:,8+6*njoint:8+12*njoint] = np.reshape(rotations_matrix[..., :, :2, :], (nfrm,-1)) # take 6D rotation
        np.save(output_file, final_x)
    print(f"bad_cnt: {bad_cnt}")
    print(f"Processed files are saved in {args.filedir}/Representation_272")
    

