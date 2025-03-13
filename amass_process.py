import codecs as cs
import pandas as pd
import numpy as np
from tqdm import tqdm
from os.path import join as pjoin
import math
import torch
from rotation_conversions import *
import copy
from human_body_prior.body_model.body_model import BodyModel
import os
import argparse



orig_flip_pairs = \
( (1,2), (4,5), (7,8), (10,11), (13,14), (16,17), (18,19), (20,21), # body joints
#(22,37), (23,38), (24,39), (25,40), (26,41), (27,42), (28,43), (29,44), (30,45), (31,46), (32,47), (33,48), (34,49), (35,50), (36,51) # hand joints
)

left_chain = []
right_chain = []

for pair in orig_flip_pairs:
    left_chain.append(pair[0])
    right_chain.append(pair[1])

# smplx_male_model_path = './smplx/SMPLX_MALE.npz'
# smplx_female_model_path = './smplx/SMPLX_FEMALE.npz'


def swap_left_right(data):
    pose = data[..., :3+51 *3].reshape(data.shape[0], 52, 3)

    tmp = pose[:, right_chain, :]
    pose[:, right_chain, :] = pose[:, left_chain, :]
    pose[:, left_chain, :] = tmp

    pose[:, :, 1:3] *= -1
    # change translation
    trans = copy.deepcopy(data[..., 309:312])
    trans[..., 0] *= -1

    data[..., :3+51 *3] = pose.reshape(data.shape[0], -1)
    data[..., 309:312] = trans

    return data

def swap_left_right_smplx_85(data):
    pose = data[..., :24 *3].reshape(data.shape[0], 24, 3)
    tmp = pose[:, right_chain, :]
    pose[:, right_chain, :] = pose[:, left_chain, :]
    pose[:, left_chain, :] = tmp
    pose[:, :, 1:3] *= -1
    trans = copy.deepcopy(data[..., 24*3:24*3+3])
    trans[..., 0] *= -1
    data[..., :24 *3] = pose.reshape(data.shape[0], -1)
    data[..., 24*3:24*3+3] = trans
    return data


def rotate_motion(root_global_orient):
    trans_matrix = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    motion = np.dot(root_global_orient, trans_matrix)  # exchange the y and z axis

    return motion

def compute_canonical_transform(global_orient):
    rotation_matrix = torch.tensor([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=global_orient.dtype)
    global_orient_matrix = axis_angle_to_matrix(global_orient)
    global_orient_matrix = torch.matmul(rotation_matrix, global_orient_matrix)
    global_orient = matrix_to_axis_angle(global_orient_matrix)
    return global_orient

def transform_translation(trans):
    trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
    trans[:, 2] = trans[:, 2] * (-1)
    return trans


def get_smplx_322(data, ex_fps):
    fps = 0


    if 'mocap_frame_rate' in data:
        fps = data['mocap_frame_rate']
        print(fps)
        down_sample = int(fps / ex_fps)
        
    elif 'mocap_framerate' in data:
        fps = data['mocap_framerate']
        print(fps)
        down_sample = int(fps / ex_fps)
    else:
        # down_sample = 1
        return None

    frame_number = data['trans'].shape[0]
    


    fId = 0 # frame id of the mocap sequence
    pose_seq = []

    

    for fId in range(0, frame_number, down_sample):
        pose_root = data['root_orient'][fId:fId+1]
        pose_body = data['pose_body'][fId:fId+1]
        pose_hand = data['pose_hand'][fId:fId+1]
        pose_jaw = data['pose_jaw'][fId:fId+1]
        pose_expression = np.zeros((1, 50))
        pose_face_shape = np.zeros((1, 100))
        pose_trans = data['trans'][fId:fId+1]
        pose_body_shape = data['betas'][:10][None, :]
        pose = np.concatenate((pose_root, pose_body, pose_hand, pose_jaw, pose_expression, pose_face_shape, pose_trans, pose_body_shape), axis=1)
        pose_seq.append(pose)

    pose_seq = np.concatenate(pose_seq, axis=0)
    
    return pose_seq

def get_smplx_85(data, ex_fps):
    fps = 0
    if 'mocap_frame_rate' in data:
        fps = data['mocap_frame_rate']
        #print(fps)
        down_sample = int(fps / ex_fps)
        
    elif 'mocap_framerate' in data:
        fps = data['mocap_framerate']
        #print(fps)
        down_sample = int(fps / ex_fps)
    else:
        # down_sample = 1
        raise ValueError('fps not found')

    frame_number = data['trans'].shape[0]
    
    fId = 0 # frame id of the mocap sequence
    pose_seq = []

    

    for fId in range(0, frame_number, down_sample):
        # import ipdb; ipdb.set_trace()
        pose_body = data['poses'][fId:fId+1, :22*3]
        # pose_root = data['root_orient'][fId:fId+1]
        pose_wrist = np.zeros((1, 6))
        # pose_hand = data['pose_hand'][fId:fId+1]
        # pose_jaw = data['pose_jaw'][fId:fId+1]
        # pose_expression = np.zeros((, 50))
        # pose_face_shape = np.zeros((1, 100))
        pose_trans = data['trans'][fId:fId+1]
        pose_body_shape = data['betas'][:10][None, :]
        pose = np.concatenate((pose_body, pose_wrist, pose_trans, pose_body_shape), axis=1)
        pose_seq.append(pose)

    pose_seq = np.concatenate(pose_seq, axis=0)
    

    return pose_seq


def process_pose(pose):
    pose_root = pose[:, :3]
    pose_root = compute_canonical_transform(torch.from_numpy(pose_root)).detach().cpu().numpy()
    pose[:, :3] = pose_root
    pose_trans = pose[:, 24*3:24*3+3]
    pose_trans = transform_translation(pose_trans)
    pose[:, 24*3:24*3+3] = pose_trans

    # return pose
    return np.float32(pose)



def face_z_align(pose, smplx_model):
    pose = torch.from_numpy(pose).float().cuda()

    param = {
        'root_orient': pose[:, :3],  # controls the global root orientation
        'pose_body': pose[:, 3:3+63],  # controls the body
        'pose_hand': pose[:, 66:66+90],  # controls the finger articulation
        'pose_jaw': pose[:, 66+90:66+93],  # controls the yaw pose
        'face_expr': pose[:, 159:159+50],  # controls the face expression
        'face_shape': pose[:, 209:209+100],  # controls the face shape
        'trans': pose[:, 309:309+3],  # controls the global body position
        'betas': pose[:, 312:],  # controls the body shape. Body shape is static
    }

    batch_size = param['face_expr'].shape[0]
    zero_pose = torch.zeros((batch_size, 3)).float().cuda()

    smplx_output = smplx_model(betas=param['betas'], body_pose=param['pose_body'],
                               global_orient=param['root_orient'], pose2rot=True, jaw_pose=zero_pose, leye_pose=zero_pose, reye_pose=zero_pose,
                               left_hand_pose=param['pose_hand'][:, :45], right_hand_pose=param['pose_hand'][:, 45:],
                               expression=param['face_expr'][:, :10], transl=param['trans'])
                        
    vertices = smplx_output.vertices
    joints = smplx_output.joints
    joints = joints[:, joint_idx, :]
    param['root_orient'], param['trans'] = face_z_transform(joints.cpu().numpy(), param['root_orient'], param['trans'])

    pose_list = []
    for k in ['root_orient', 'pose_body', 'pose_hand', 'pose_jaw', 'face_expr', 'face_shape', 'trans', 'betas']:
        pose_list.append(param[k])
    pose_list = torch.cat(pose_list, dim=-1).cpu().numpy()

    return pose_list


from concurrent.futures import ThreadPoolExecutor
import threading

def process_single_file(args):
    i, row, male_bm, female_bm, save_dir, ex_fps = args
    try:
        if i % 1000 == 0:
            print(f"processing {i} file")
        
        source_path = row['source_path']
        source_path = source_path.replace('./datasets/pose_data', './datasets/amass_data')
        source_path = source_path.replace('.npy', '.npz')
        
        data = np.load(source_path)
        gender = data['gender'].item()
        
        bm = male_bm if gender == 'male' else female_bm
        
        new_name = row['new_name']

        start_frame = row['start_frame']  
        end_frame = row['end_frame']

        pose = get_smplx_85(data, ex_fps)
        pose = pose[int(start_frame):int(end_frame)]
        pose = process_pose(pose)
        
        if pose is not None:
            np.save(pjoin(save_dir, new_name), pose)
            pose_mirror = swap_left_right_smplx_85(pose)
            np.save(pjoin(save_dir, 'M'+new_name), pose_mirror)
            return 0
        return 1
    except Exception as e:
        print(f"error: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--index_path', type=str, required=True, help='Path to the index CSV file')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save processed data')
    
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    
    index_path = args.index_path
    save_dir = args.save_dir

    index_file = pd.read_csv(index_path)
    total_amount = index_file.shape[0]
    ex_fps = 60   # resample to 60 FPS

    male_bm_path = './body_models/smplh/male/model.npz'
    male_dmpl_path = './body_models/dmpls/male/model.npz'
    female_bm_path = './body_models/smplh/female/model.npz'
    female_dmpl_path = './body_models/dmpls/female/model.npz'

    num_betas = 10
    num_dmpls = 8

    male_bm = BodyModel(bm_fname=male_bm_path, num_betas=num_betas,
                       num_dmpls=num_dmpls, dmpl_fname=male_dmpl_path)
    female_bm = BodyModel(bm_fname=female_bm_path, num_betas=num_betas,
                         num_dmpls=num_dmpls, dmpl_fname=female_dmpl_path)

    args_list = [(i, index_file.iloc[i], male_bm, female_bm, save_dir, ex_fps) 
                 for i in range(total_amount)]
    
    
    max_workers = 97  # Number of threads
    bad_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_single_file, args_list), 
                          total=total_amount, 
                          desc="Processing file progress"))
    
    bad_count = sum(results)
    print(f"Number of failed processing: {bad_count}")

if __name__ == '__main__':
    main()