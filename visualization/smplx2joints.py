import torch
from einops import rearrange
from utils.rotation_conversions import rotation_6d_to_matrix, matrix_to_axis_angle
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import copy
from utils.rotation_conversions import axis_angle_to_matrix
import math
import torch.nn.functional as F

# sys.path.insert(0, '/home/linjing/code/Hand4Whole_RELEASE-Fitting/main')
# sys.path.insert(0, '/home/linjing/code/Hand4Whole_RELEASE-Fitting/data')
# sys.path.insert(0, '/home/linjing/code/Hand4Whole_RELEASE-Fitting/common')
from utils.config_3 import cfg
cfg.set_additional_args(use_flame=True, modify_root_joint=False)
from utils.human_models import smpl_x, mano
cudnn.benchmark = True
# define smplx model
smplx_layer = copy.deepcopy(smpl_x.layer['neutral']).cuda()
lhand_mano_layer = copy.deepcopy(mano.layer['left']).cuda()
rhand_mano_layer = copy.deepcopy(mano.layer['right']).cuda()

import torchgeometry as tgm
# import eulerangles


def extract(features):
    root_y = features[..., 0]
    vel_trajectory = features[..., 1:3]
    poses_features = features[..., 3:]
    poses = rearrange(poses_features,
                      "... (joints rot) -> ... joints rot", rot=6)
    return root_y, vel_trajectory, poses

def inverse(features):
    root_y, vel_trajectory, poses = extract(features)

    # integrate the trajectory
    trajectory = torch.cumsum(vel_trajectory, dim=-2)
    # First frame should be 0, but if infered it is better to ensure it
    trajectory = trajectory - trajectory[..., [0], :]

    # Get back the translation
    trans = torch.cat([trajectory, root_y[..., None]], dim=-1)
    matrix_poses = rotation_6d_to_matrix(poses)
    axis_angle_poses = matrix_to_axis_angle(matrix_poses).flatten(-2)

    return trans, axis_angle_poses

def feature2smplx(path=None):
    # path = '/home/linjing/data/Motion_Generation/AMASS/SMPLX_G/joints_combined_smplx6d/000023.npy'
    data = np.load(path)
    trans, poses = inverse(torch.from_numpy(data[..., 263:-10]))
    # print(trans)
    trans, poses = trans.numpy(), poses.numpy()
    betas = data[..., -10:]
    smplx_param = np.concatenate((poses, trans, betas), axis=-1)
    return smplx_param

# keep x orientation;
# def compute_canonical_transform(global_orient, trans):
#     device = global_orient.device
#     dtype = global_orient.dtype
#     R = tgm.angle_axis_to_rotation_matrix(global_orient)  # [:, :3, :3].detach().cpu().numpy().squeeze()
#     # R_inv = R[:, :3, :3].reshape(3, 3).t()
#     R_inv = R[:, :3, :3].transpose(1, 2)
#
#     x, y, z = -(90/180) * math.pi, 0, 0
#
#     # x, z, y = eulerangles.mat2euler(R[:, :3, :3].detach().cpu().numpy().squeeze(), 'sxzy')
#     # y = 0
#     # z = 0
#     #  The solution is not unique in most cases.
#     #  Using the code in the previous section you can verify that rotation matrices corresponding to
#     #  Euler angles [0.1920, 2.3736, 1.1170] ( or [[11, 136, 64] in degrees) and
#     # [-2.9496, 0.7679, -2.0246] ( or [-169, 44, -116] in degrees)
#     # are actually the same even though the Euler angles look very different.
#     # print(x, y, z)
#     num_frame = global_orient.shape[0]
#     R_new = torch.tensor(eulerangles.euler2mat(x, z, y, 'sxzy'), dtype=dtype, device=device)
#     global_orient_canonical = torch.matmul(R_new, R_inv)
#     global_orient_canonical = matrix_to_axis_angle(global_orient_canonical)
#
#     trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
#     trans_matrix = torch.from_numpy(trans_matrix).float().cuda()
#     trans = torch.matmul(trans, trans_matrix)
#     trans[:, 2] = trans[:, 2] * (-1)
#
#     return global_orient_canonical, trans

def compute_canonical_transform(global_orient):
    rotation_matrix = torch.tensor([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=global_orient.dtype).cuda()
    global_orient_matrix = axis_angle_to_matrix(global_orient)
    global_orient_matrix = torch.matmul(rotation_matrix, global_orient_matrix)
    global_orient = matrix_to_axis_angle(global_orient_matrix)
    return global_orient


def updateFaceMotion(motion=None, motion_file=None, emotion=None, face_motion_file=None):
    if motion is None:
        motion = np.load(motion_file)
    motion_length = motion.shape[0]
    if face_motion_file is None:
        face_motion_file = random.choice(face_motion_files)
    face_motion = np.load(face_motion_file)
    emotion = face_motion_file.split('/')[-3]
    if face_motion.shape[0] != motion_length:
        face_motion = torch.from_numpy(face_motion)
        face_motion = face_motion[None].permute(0, 2, 1) # [1, num_feats, num_frames]
        face_motion = F.interpolate(face_motion, size=motion_length, mode='linear')   # motion interpolate
        face_motion = face_motion.permute(0, 2, 1)[0].numpy()   # [num_frames, num_feats]
    motion[:, 159:159+150] = face_motion    # update face motion
    # motion[:, 159:159+50] = face_motion[:, :50]    # update face expression
    # motion[:, 66+90:66+93] = face_motion[:, 50:]    # update yaw pose
    return motion, emotion

def process(input_path=None, face_replace_path=None, face_carnical=False, pose=None, norm_global_orient=False, transform=False):
    # generate mesh
    motion_representation = 'rot3d'
    if pose is None:
        if motion_representation=='rot6d':
            pose = torch.tensor(feature2smplx(input_path), dtype=torch.float32)
        else:
            pose = torch.tensor(np.load(input_path), dtype=torch.float32)
    
    if face_replace_path is not None:
        face_data = torch.tensor(np.load(face_replace_path), dtype=torch.float32)
        if face_data.shape[0] != pose.shape[0]:
            # face_motion = torch.from_numpy(face_motion)
            # import pdb; pdb.set_trace()
            face_data = face_data[None].permute(0, 2, 1) # [1, num_feats, num_frames]
            face_data = F.interpolate(face_data, size=pose.shape[0], mode='linear')   # motion interpolate
            face_data = face_data.permute(0, 2, 1)[0]   # [num_frames, num_feats]
        assert face_data.shape[1] == 50
        pose[:, 159:209] = face_data

    use_flame = (pose.shape[1] == 322)
    # pose = torch.zeros_like(pose)
    cfg.set_additional_args(use_flame=use_flame, modify_root_joint=False)
    num_frames = pose.shape[0]
    zero_pose = torch.zeros((1, 3)).float().cuda().repeat(num_frames, 1)  # eye poses
    zeor_expr = torch.zeros((1, 10)).float().cuda().repeat(num_frames, 1)  # smplx face expression
    if face_carnical:
        neck_idx = 12
        pose[:, neck_idx*3:(neck_idx+1)*3] = zero_pose
        pose[:, :3] = 0

    if use_flame:
        # 322
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation
            'pose_body': pose[:, 3:3+63].to(comp_device),  # controls the body
            'pose_hand': pose[:, 66:66+90].to(comp_device),  # controls the finger articulation
            'pose_jaw': pose[:, 66+90:66+93].to(comp_device),  # controls the yaw pose
            'face_expr': pose[:, 159:159+50].to(comp_device),  # controls the face expression
            'face_shape': pose[:, 209:209+100].to(comp_device),  # controls the face shape
            'trans': pose[:, 309:309+3].to(comp_device),  # controls the global body position
            'betas': pose[:, 312:].to(comp_device),  # controls the body shape. Body shape is static
        }
    else:
        #172
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation 3
            'pose_body': pose[:, 3:3 + 63].to(comp_device),  # controls the body 63
            'pose_hand': pose[:, 66:66 + 90].to(comp_device),  # controls the finger articulation 90
            'pose_jaw': pose[:, 66 + 90:66 + 93].to(comp_device),  # controls the yaw pose 3
            'trans': pose[:, 159:159 + 3].to(comp_device),  # controls the global body position 3
            'betas': pose[:, 162:].to(comp_device),  # controls the body shape. Body shape is static 10
        }
        
    if norm_global_orient:
        for i in range(num_frames):
            body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1] = compute_canonical_transform(body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1])

    if transform:
        trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        trans = body_parms['trans'].cpu().numpy()
        trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
        trans[:, 1] = trans[:, 1] * (-1)
        body_parms['trans'] = torch.tensor(trans, dtype=torch.float32).to(comp_device)
        body_parms['root_orient'] = compute_canonical_transform(body_parms['root_orient'])

    if use_flame:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, transl=body_parms['trans'],
                             expression=zeor_expr, flame_betas=body_parms['face_shape'], flame_expression=body_parms['face_expr'])
    else:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, expression=zeor_expr,
                             transl=body_parms['trans'])
    vertices = output.vertices
    # import pdb; pdb.set_trace()
    joints = output.joints[:, smpl_x.joint_idx, :]
    faces = output.faces

    vertices_mesh=vertices.detach().cpu().numpy()
    # import trimesh
    # for i in range(len(vertices_mesh)):
        
    #     mesh = trimesh.Trimesh(vertices=vertices_mesh[i], faces=faces)
    #     mesh.export(f'/comp_robot/lushunlin/visualization/visualization/humanml_077/humanml_077_{i}.obj')
    # import pdb; pdb.set_trace()

    return vertices, joints, pose, faces


def process_replace_face_hand(input_path=None, face_replace_path=None, hand_replace_path=None, face_carnical=False, pose=None, norm_global_orient=False, transform=False):
    # generate mesh
    motion_representation = 'rot3d'
    if pose is None:
        if motion_representation=='rot6d':
            pose = torch.tensor(feature2smplx(input_path), dtype=torch.float32)
        else:
            pose = torch.tensor(np.load(input_path), dtype=torch.float32)
    
    if face_replace_path is not None:
        face_data = torch.tensor(np.load(face_replace_path), dtype=torch.float32)
        if face_data.shape[0] != pose.shape[0]:
            # face_motion = torch.from_numpy(face_motion)
            # import pdb; pdb.set_trace()
            face_data = face_data[None].permute(0, 2, 1) # [1, num_feats, num_frames]
            face_data = F.interpolate(face_data, size=pose.shape[0], mode='linear')   # motion interpolate
            face_data = face_data.permute(0, 2, 1)[0]   # [num_frames, num_feats]
        assert face_data.shape[1] == 53
        pose[:, 156:209] = face_data

    if hand_replace_path is not None:
        hand_motion_data = torch.tensor(np.load(hand_replace_path), dtype=torch.float32)[:, 66:156]
        if hand_motion_data.shape[0] != pose.shape[0]:
            # face_motion = torch.from_numpy(face_motion)
            # import pdb; pdb.set_trace()
            hand_motion_data = hand_motion_data[None].permute(0, 2, 1)
            hand_motion_data = F.interpolate(hand_motion_data, size=pose.shape[0], mode='linear')   # motion interpolate
            hand_motion_data = hand_motion_data.permute(0, 2, 1)[0]   # [num_frames, num_feats]
        assert hand_motion_data.shape[1] == 90
        pose[:, 66:156] = hand_motion_data

    pose[:, 309:312] = torch.zeros((pose.shape[0], 3)).float().cuda()  # zero translation

    use_flame = (pose.shape[1] == 322)
    # pose = torch.zeros_like(pose)
    cfg.set_additional_args(use_flame=use_flame, modify_root_joint=False)
    num_frames = pose.shape[0]
    zero_pose = torch.zeros((1, 3)).float().cuda().repeat(num_frames, 1)  # eye poses
    zeor_expr = torch.zeros((1, 10)).float().cuda().repeat(num_frames, 1)  # smplx face expression
    if face_carnical:
        neck_idx = 12
        pose[:, neck_idx*3:(neck_idx+1)*3] = zero_pose
        pose[:, :3] = 0

    if use_flame:
        # 322
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation
            'pose_body': pose[:, 3:3+63].to(comp_device),  # controls the body
            'pose_hand': pose[:, 66:66+90].to(comp_device),  # controls the finger articulation
            'pose_jaw': pose[:, 66+90:66+93].to(comp_device),  # controls the yaw pose
            'face_expr': pose[:, 159:159+50].to(comp_device),  # controls the face expression
            'face_shape': pose[:, 209:209+100].to(comp_device),  # controls the face shape
            'trans': pose[:, 309:309+3].to(comp_device),  # controls the global body position
            'betas': pose[:, 312:].to(comp_device),  # controls the body shape. Body shape is static
        }
    else:
        #172
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation 3
            'pose_body': pose[:, 3:3 + 63].to(comp_device),  # controls the body 63
            'pose_hand': pose[:, 66:66 + 90].to(comp_device),  # controls the finger articulation 90
            'pose_jaw': pose[:, 66 + 90:66 + 93].to(comp_device),  # controls the yaw pose 3
            'trans': pose[:, 159:159 + 3].to(comp_device),  # controls the global body position 3
            'betas': pose[:, 162:].to(comp_device),  # controls the body shape. Body shape is static 10
        }
        
    if norm_global_orient:
        for i in range(num_frames):
            body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1] = compute_canonical_transform(body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1])

    if transform:
        trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        trans = body_parms['trans'].cpu().numpy()
        trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
        trans[:, 1] = trans[:, 1] * (-1)
        body_parms['trans'] = torch.tensor(trans, dtype=torch.float32).to(comp_device)
        body_parms['root_orient'] = compute_canonical_transform(body_parms['root_orient'])

    if use_flame:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, transl=body_parms['trans'],
                             expression=zeor_expr, flame_betas=body_parms['face_shape'], flame_expression=body_parms['face_expr'])
    else:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, expression=zeor_expr,
                             transl=body_parms['trans'])
    vertices = output.vertices
    # import pdb; pdb.set_trace()
    joints = output.joints[:, smpl_x.joint_idx, :]
    faces = output.faces

    vertices_mesh=vertices.detach().cpu().numpy()
    # import trimesh
    # for i in range(len(vertices_mesh)):
        
    #     mesh = trimesh.Trimesh(vertices=vertices_mesh[i], faces=faces)
    #     mesh.export(f'/comp_robot/lushunlin/visualization/visualization/humanml_077/humanml_077_{i}.obj')
    # import pdb; pdb.set_trace()

    return vertices, joints, pose, faces

def process_head_zero_trans(input_path=None, face_replace_path=None, face_carnical=False, pose=None, norm_global_orient=False, transform=False):
    # generate mesh
    motion_representation = 'rot3d'
    if pose is None:
        if motion_representation=='rot6d':
            pose = torch.tensor(feature2smplx(input_path), dtype=torch.float32)
        else:
            pose = torch.tensor(np.load(input_path), dtype=torch.float32)
    
    if face_replace_path is not None:
        face_data = torch.tensor(np.load(face_replace_path), dtype=torch.float32)
        if face_data.shape[0] != pose.shape[0]:
            # face_motion = torch.from_numpy(face_motion)
            # import pdb; pdb.set_trace()
            face_data = face_data[None].permute(0, 2, 1) # [1, num_feats, num_frames]
            face_data = F.interpolate(face_data, size=pose.shape[0], mode='linear')   # motion interpolate
            face_data = face_data.permute(0, 2, 1)[0]   # [num_frames, num_feats]
        assert face_data.shape[1] == 50
        pose[:, 159:209] = face_data

    use_flame = (pose.shape[1] == 322)
    # pose = torch.zeros_like(pose)
    cfg.set_additional_args(use_flame=use_flame, modify_root_joint=False)
    num_frames = pose.shape[0]
    zero_pose = torch.zeros((1, 3)).float().cuda().repeat(num_frames, 1)  # eye poses
    zeor_expr = torch.zeros((1, 10)).float().cuda().repeat(num_frames, 1)  # smplx face expression
    if face_carnical:
        neck_idx = 12
        pose[:, neck_idx*3:(neck_idx+1)*3] = zero_pose
        pose[:, :3] = 0

    if use_flame:
        # 322
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation
            'pose_body': pose[:, 3:3+63].to(comp_device),  # controls the body
            'pose_hand': pose[:, 66:66+90].to(comp_device),  # controls the finger articulation
            'pose_jaw': pose[:, 66+90:66+93].to(comp_device),  # controls the yaw pose
            'face_expr': pose[:, 159:159+50].to(comp_device),  # controls the face expression
            'face_shape': pose[:, 209:209+100].to(comp_device),  # controls the face shape
            'trans': pose[:, 309:309+3].to(comp_device),  # controls the global body position
            'betas': pose[:, 312:].to(comp_device),  # controls the body shape. Body shape is static
        }
    else:
        #172
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation 3
            'pose_body': pose[:, 3:3 + 63].to(comp_device),  # controls the body 63
            'pose_hand': pose[:, 66:66 + 90].to(comp_device),  # controls the finger articulation 90
            'pose_jaw': pose[:, 66 + 90:66 + 93].to(comp_device),  # controls the yaw pose 3
            'trans': pose[:, 159:159 + 3].to(comp_device),  # controls the global body position 3
            'betas': pose[:, 162:].to(comp_device),  # controls the body shape. Body shape is static 10
        }
        
    if norm_global_orient:
        for i in range(num_frames):
            body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1] = compute_canonical_transform(body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1])

    if transform:
        trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        trans = body_parms['trans'].cpu().numpy()
        trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
        trans[:, 1] = trans[:, 1] * (-1)
        body_parms['trans'] = torch.tensor(trans, dtype=torch.float32).to(comp_device)
        body_parms['root_orient'] = compute_canonical_transform(body_parms['root_orient'])

    if use_flame:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, transl=body_parms['trans'],
                             expression=zeor_expr, flame_betas=body_parms['face_shape'], flame_expression=body_parms['face_expr'])
    else:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, expression=zeor_expr,
                             transl=body_parms['trans'])
    vertices = output.vertices
    # import pdb; pdb.set_trace()
    # joints = output.joints[:, smpl_x.joint_idx, :]
    
    faces = output.faces


    # import pdb; pdb.set_trace()
    
    vertices_mesh=vertices.detach().cpu().numpy()
    # import trimesh
    # for i in range(len(vertices_mesh)):
        
    #     mesh = trimesh.Trimesh(vertices=vertices_mesh[i], faces=faces)
    #     mesh.export(f'/comp_robot/lushunlin/visualization/visualization/humanml_077/humanml_077_{i}.obj')
    # import pdb; pdb.set_trace()

    return vertices, output.joints, pose, faces


def process_hand_zero_trans(input_path=None, face_replace_path=None, face_carnical=False, pose=None, norm_global_orient=False, transform=False):
    # generate mesh
    motion_representation = 'rot3d'
    if pose is None:
        if motion_representation=='rot6d':
            pose = torch.tensor(feature2smplx(input_path), dtype=torch.float32)
        else:
            pose = torch.tensor(np.load(input_path), dtype=torch.float32)
    
    if face_replace_path is not None:
        face_data = torch.tensor(np.load(face_replace_path), dtype=torch.float32)
        if face_data.shape[0] != pose.shape[0]:
            # face_motion = torch.from_numpy(face_motion)
            # import pdb; pdb.set_trace()
            face_data = face_data[None].permute(0, 2, 1) # [1, num_feats, num_frames]
            face_data = F.interpolate(face_data, size=pose.shape[0], mode='linear')   # motion interpolate
            face_data = face_data.permute(0, 2, 1)[0]   # [num_frames, num_feats]
        assert face_data.shape[1] == 50
        pose[:, 159:209] = face_data

    use_flame = (pose.shape[1] == 322)
    # pose = torch.zeros_like(pose)
    cfg.set_additional_args(use_flame=use_flame, modify_root_joint=False)
    num_frames = pose.shape[0]
    zero_pose = torch.zeros((1, 3)).float().cuda().repeat(num_frames, 1)  # eye poses
    zeor_expr = torch.zeros((1, 10)).float().cuda().repeat(num_frames, 1)  # smplx face expression
    if face_carnical:
        neck_idx = 12
        pose[:, neck_idx*3:(neck_idx+1)*3] = zero_pose
        pose[:, :3] = 0

    if use_flame:
        # 322
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation
            'pose_body': pose[:, 3:3+63].to(comp_device),  # controls the body
            'pose_hand': pose[:, 66:66+90].to(comp_device),  # controls the finger articulation
            'pose_jaw': pose[:, 66+90:66+93].to(comp_device),  # controls the yaw pose
            'face_expr': pose[:, 159:159+50].to(comp_device),  # controls the face expression
            'face_shape': pose[:, 209:209+100].to(comp_device),  # controls the face shape
            'trans': pose[:, 309:309+3].to(comp_device),  # controls the global body position
            'betas': pose[:, 312:].to(comp_device),  # controls the body shape. Body shape is static
        }
    else:
        #172
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation 3
            'pose_body': pose[:, 3:3 + 63].to(comp_device),  # controls the body 63
            'pose_hand': pose[:, 66:66 + 90].to(comp_device),  # controls the finger articulation 90
            'pose_jaw': pose[:, 66 + 90:66 + 93].to(comp_device),  # controls the yaw pose 3
            'trans': pose[:, 159:159 + 3].to(comp_device),  # controls the global body position 3
            'betas': pose[:, 162:].to(comp_device),  # controls the body shape. Body shape is static 10
        }
        
    if norm_global_orient:
        for i in range(num_frames):
            body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1] = compute_canonical_transform(body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1])

    if transform:
        trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        trans = body_parms['trans'].cpu().numpy()
        trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
        trans[:, 1] = trans[:, 1] * (-1)
        body_parms['trans'] = torch.tensor(trans, dtype=torch.float32).to(comp_device)
        body_parms['root_orient'] = compute_canonical_transform(body_parms['root_orient'])

    if use_flame:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, transl=body_parms['trans'],
                             expression=zeor_expr, flame_betas=body_parms['face_shape'], flame_expression=body_parms['face_expr'])
    else:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, expression=zeor_expr,
                             transl=body_parms['trans'])
    vertices = output.vertices
    # import pdb; pdb.set_trace()
    # joints = output.joints[:, smpl_x.joint_idx, :]
    
    faces = output.faces


    # import pdb; pdb.set_trace()
    
    vertices_mesh=vertices.detach().cpu().numpy()
    # import trimesh
    # for i in range(len(vertices_mesh)):
        
    #     mesh = trimesh.Trimesh(vertices=vertices_mesh[i], faces=faces)
    #     mesh.export(f'/comp_robot/lushunlin/visualization/visualization/humanml_077/humanml_077_{i}.obj')
    # import pdb; pdb.set_trace()

    return vertices, output.joints, pose, faces

def process_input_data(input_data=None, face_replace_path=None, face_carnical=False, pose=None, norm_global_orient=False, transform=False):
    # generate mesh
    motion_representation = 'rot3d'
    assert input_data is not None
    # if pose is None:
    #     if motion_representation=='rot6d':
    #         pose = torch.tensor(feature2smplx(input_path), dtype=torch.float32)
    #     else:
    #         pose = torch.tensor(np.load(input_path), dtype=torch.float32)
    
    pose = torch.tensor(input_data)
    if face_replace_path is not None:
        face_data = torch.tensor(np.load(face_replace_path), dtype=torch.float32)
        if face_data.shape[0] != pose.shape[0]:
            # face_motion = torch.from_numpy(face_motion)
            # import pdb; pdb.set_trace()
            face_data = face_data[None].permute(0, 2, 1) # [1, num_feats, num_frames]
            face_data = F.interpolate(face_data, size=pose.shape[0], mode='linear')   # motion interpolate
            face_data = face_data.permute(0, 2, 1)[0]   # [num_frames, num_feats]
        assert face_data.shape[1] == 50
        pose[:, 159:209] = face_data

    use_flame = (pose.shape[1] == 322)
    cfg.set_additional_args(use_flame=use_flame, modify_root_joint=False)
    num_frames = pose.shape[0]
    zero_pose = torch.zeros((1, 3)).float().cuda().repeat(num_frames, 1)  # eye poses
    zeor_expr = torch.zeros((1, 10)).float().cuda().repeat(num_frames, 1)  # smplx face expression
    if face_carnical:
        neck_idx = 12
        pose[:, neck_idx*3:(neck_idx+1)*3] = zero_pose
        pose[:, :3] = 0

    if use_flame:
        # 322
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation
            'pose_body': pose[:, 3:3+63].to(comp_device),  # controls the body
            'pose_hand': pose[:, 66:66+90].to(comp_device),  # controls the finger articulation
            'pose_jaw': pose[:, 66+90:66+93].to(comp_device),  # controls the yaw pose
            'face_expr': pose[:, 159:159+50].to(comp_device),  # controls the face expression
            'face_shape': pose[:, 209:209+100].to(comp_device),  # controls the face shape
            'trans': pose[:, 309:309+3].to(comp_device),  # controls the global body position
            'betas': pose[:, 312:].to(comp_device),  # controls the body shape. Body shape is static
        }
    else:
        #172
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation 3
            'pose_body': pose[:, 3:3 + 63].to(comp_device),  # controls the body 63
            'pose_hand': pose[:, 66:66 + 90].to(comp_device),  # controls the finger articulation 90
            'pose_jaw': pose[:, 66 + 90:66 + 93].to(comp_device),  # controls the yaw pose 3
            'trans': pose[:, 159:159 + 3].to(comp_device),  # controls the global body position 3
            'betas': pose[:, 162:].to(comp_device),  # controls the body shape. Body shape is static 10
        }
        
    if norm_global_orient:
        for i in range(num_frames):
            body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1] = compute_canonical_transform(body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1])

    if transform:
        trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        trans = body_parms['trans'].cpu().numpy()
        trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
        trans[:, 1] = trans[:, 1] * (-1)
        body_parms['trans'] = torch.tensor(trans, dtype=torch.float32).to(comp_device)
        body_parms['root_orient'] = compute_canonical_transform(body_parms['root_orient'])

    if use_flame:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, transl=body_parms['trans'],
                             expression=zeor_expr, flame_betas=body_parms['face_shape'], flame_expression=body_parms['face_expr'])
    else:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, expression=zeor_expr,
                             transl=body_parms['trans'])
    vertices = output.vertices
    # import pdb; pdb.set_trace()
    joints = output.joints[:, smpl_x.joint_idx, :]
    faces = output.faces

    vertices_mesh=vertices.detach().cpu().numpy()
    # import trimesh
    # for i in range(len(vertices_mesh)):
        
    #     mesh = trimesh.Trimesh(vertices=vertices_mesh[i], faces=faces)
    #     mesh.export(f'/comp_robot/lushunlin/visualization/visualization/humanml_077/humanml_077_{i}.obj')
    # import pdb; pdb.set_trace()

    return vertices, joints, pose, faces

def process_teaser(input_path=None, face_replace_path=None, face_carnical=False, pose=None, norm_global_orient=False, transform=False):
    # generate mesh
    motion_representation = 'rot3d'
    if pose is None:
        if motion_representation=='rot6d':
            pose = torch.tensor(feature2smplx(input_path), dtype=torch.float32)
        else:
            pose = torch.tensor(np.load(input_path), dtype=torch.float32)
    
    # pose = torch.zeros_like(pose)

    if face_replace_path is not None:
        face_data = torch.tensor(np.load(face_replace_path), dtype=torch.float32)
        if face_data.shape[0] != pose.shape[0]:
            # face_motion = torch.from_numpy(face_motion)
            # import pdb; pdb.set_trace()
            face_data = face_data[None].permute(0, 2, 1) # [1, num_feats, num_frames]
            face_data = F.interpolate(face_data, size=pose.shape[0], mode='linear')   # motion interpolate
            face_data = face_data.permute(0, 2, 1)[0]   # [num_frames, num_feats]
        assert face_data.shape[1] == 50
        pose[:, 159:209] = face_data
    
    # selected_index = torch.tensor([0, 1, 9, 132, 186, 188, 190, 195])
    # selected_index = torch.tensor([0, 1, 9, 132, 186, 188, 190, 195])
    selected_index = torch.tensor([0, 9, 186, 195])
    pose = torch.index_select(pose, 0, selected_index)

    pose[..., 309:309+3] = 0
    for i in range(len(pose)):
        pose[i, 309] = -0.45 * i

        

    # import pdb; pdb.set_trace()

    # pose = pose[]
    use_flame = (pose.shape[1] == 322)
    cfg.set_additional_args(use_flame=use_flame, modify_root_joint=False)
    num_frames = pose.shape[0]
    zero_pose = torch.zeros((1, 3)).float().cuda().repeat(num_frames, 1)  # eye poses
    zeor_expr = torch.zeros((1, 10)).float().cuda().repeat(num_frames, 1)  # smplx face expression
    if face_carnical:
        neck_idx = 12
        pose[:, neck_idx*3:(neck_idx+1)*3] = zero_pose
        pose[:, :3] = 0

    if use_flame:
        # 322
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation
            'pose_body': pose[:, 3:3+63].to(comp_device),  # controls the body
            'pose_hand': pose[:, 66:66+90].to(comp_device),  # controls the finger articulation
            'pose_jaw': pose[:, 66+90:66+93].to(comp_device),  # controls the yaw pose
            'face_expr': pose[:, 159:159+50].to(comp_device),  # controls the face expression
            'face_shape': pose[:, 209:209+100].to(comp_device),  # controls the face shape
            'trans': pose[:, 309:309+3].to(comp_device),  # controls the global body position
            'betas': pose[:, 312:].to(comp_device),  # controls the body shape. Body shape is static
        }
    else:
        #172
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation 3
            'pose_body': pose[:, 3:3 + 63].to(comp_device),  # controls the body 63
            'pose_hand': pose[:, 66:66 + 90].to(comp_device),  # controls the finger articulation 90
            'pose_jaw': pose[:, 66 + 90:66 + 93].to(comp_device),  # controls the yaw pose 3
            'trans': pose[:, 159:159 + 3].to(comp_device),  # controls the global body position 3
            'betas': pose[:, 162:].to(comp_device),  # controls the body shape. Body shape is static 10
        }
        
    if norm_global_orient:
        for i in range(num_frames):
            body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1] = compute_canonical_transform(body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1])

    if transform:
        trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        trans = body_parms['trans'].cpu().numpy()
        trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
        trans[:, 1] = trans[:, 1] * (-1)
        body_parms['trans'] = torch.tensor(trans, dtype=torch.float32).to(comp_device)
        body_parms['root_orient'] = compute_canonical_transform(body_parms['root_orient'])

    if use_flame:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, transl=body_parms['trans'],
                             expression=zeor_expr, flame_betas=body_parms['face_shape'], flame_expression=body_parms['face_expr'])
    else:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, expression=zeor_expr,
                             transl=body_parms['trans'])
    vertices = output.vertices
    # import pdb; pdb.set_trace()
    joints = output.joints[:, smpl_x.joint_idx, :]
    faces = output.faces

    vertices_mesh=vertices.detach().cpu().numpy()
    # import trimesh
    # for i in range(len(vertices_mesh)):
        
    #     mesh = trimesh.Trimesh(vertices=vertices_mesh[i], faces=faces)
    #     mesh.export(f'/comp_robot/lushunlin/visualization/visualization/humanml_077/humanml_077_{i}.obj')
    # import pdb; pdb.set_trace()

    return vertices, joints, pose, faces




def process_teaser_2(input_path=None, face_replace_path=None, face_carnical=False, pose=None, norm_global_orient=False, transform=False):
    # generate mesh
    
    motion_representation = 'rot3d'
    if pose is None:
        if motion_representation=='rot6d':
            pose = torch.tensor(feature2smplx(input_path), dtype=torch.float32)
        else:
            pose = torch.tensor(np.load(input_path), dtype=torch.float32)

    # pose_jaw = pose[:, 66+90:66+93]
    # pose = torch.zeros_like(pose)
    # pose[:, 66+90:66+93] = pose_jaw
    
    if face_replace_path is not None:
        face_data = torch.tensor(np.load(face_replace_path), dtype=torch.float32)
        # face_data = face_data[:, 0:50]
        if face_data.shape[0] != pose.shape[0]:
            # face_motion = torch.from_numpy(face_motion)
            # import pdb; pdb.set_trace()
            face_data = face_data[None].permute(0, 2, 1) # [1, num_feats, num_frames]
            face_data = F.interpolate(face_data, size=pose.shape[0], mode='linear')   # motion interpolate
            face_data = face_data.permute(0, 2, 1)[0]   # [num_frames, num_feats]
        assert face_data.shape[1] == 53
        pose[:, 156:209] = face_data


    
    
    selected_index = torch.tensor([0, 31, 62, 93])
    # selected_index = torch.tensor([0, 19, 39, 58, 77, 96, 116, 135])
    # selected_index = torch.tensor([0, 39,  58, 77, 116, 135])
    pose = torch.index_select(pose, 0, selected_index)

    pose[..., 309:309+3] = 0
    for i in range(len(pose)):
        pose[i, 309] = 0.55 * i
        if i == 3:
            pose[i, 309] = 0.6 * i
        # pose[i, 311] = -0.25 * i

    # import pdb; pdb.set_trace()

    # pose = pose[]
    use_flame = (pose.shape[1] == 322)
    cfg.set_additional_args(use_flame=use_flame, modify_root_joint=False)
    num_frames = pose.shape[0]
    zero_pose = torch.zeros((1, 3)).float().cuda().repeat(num_frames, 1)  # eye poses
    zeor_expr = torch.zeros((1, 10)).float().cuda().repeat(num_frames, 1)  # smplx face expression
    if face_carnical:
        neck_idx = 12
        pose[:, neck_idx*3:(neck_idx+1)*3] = zero_pose
        pose[:, :3] = 0

    if use_flame:
        # 322
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation
            'pose_body': pose[:, 3:3+63].to(comp_device),  # controls the body
            'pose_hand': pose[:, 66:66+90].to(comp_device),  # controls the finger articulation
            'pose_jaw': pose[:, 66+90:66+93].to(comp_device),  # controls the yaw pose
            'face_expr': pose[:, 159:159+50].to(comp_device),  # controls the face expression
            'face_shape': pose[:, 209:209+100].to(comp_device),  # controls the face shape
            'trans': pose[:, 309:309+3].to(comp_device),  # controls the global body position
            'betas': pose[:, 312:].to(comp_device),  # controls the body shape. Body shape is static
        }
    else:
        #172
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation 3
            'pose_body': pose[:, 3:3 + 63].to(comp_device),  # controls the body 63
            'pose_hand': pose[:, 66:66 + 90].to(comp_device),  # controls the finger articulation 90
            'pose_jaw': pose[:, 66 + 90:66 + 93].to(comp_device),  # controls the yaw pose 3
            'trans': pose[:, 159:159 + 3].to(comp_device),  # controls the global body position 3
            'betas': pose[:, 162:].to(comp_device),  # controls the body shape. Body shape is static 10
        }
        
    if norm_global_orient:
        for i in range(num_frames):
            body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1] = compute_canonical_transform(body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1])

    if transform:
        trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        trans = body_parms['trans'].cpu().numpy()
        trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
        trans[:, 1] = trans[:, 1] * (-1)
        body_parms['trans'] = torch.tensor(trans, dtype=torch.float32).to(comp_device)
        body_parms['root_orient'] = compute_canonical_transform(body_parms['root_orient'])

    if use_flame:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, transl=body_parms['trans'],
                             expression=zeor_expr, flame_betas=body_parms['face_shape'], flame_expression=body_parms['face_expr'])
    else:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, expression=zeor_expr,
                             transl=body_parms['trans'])
    vertices = output.vertices
    # import pdb; pdb.set_trace()
    joints = output.joints[:, smpl_x.joint_idx, :]
    faces = output.faces

    vertices_mesh=vertices.detach().cpu().numpy()
    # import trimesh
    # for i in range(len(vertices_mesh)):
        
    #     mesh = trimesh.Trimesh(vertices=vertices_mesh[i], faces=faces)
    #     mesh.export(f'/comp_robot/lushunlin/visualization/visualization/humanml_077/humanml_077_{i}.obj')
    # import pdb; pdb.set_trace()

    return vertices, joints, pose, faces




def process_teaser_kicking(input_path=None, face_replace_path=None, face_carnical=False, pose=None, norm_global_orient=False, transform=False):
    # generate mesh
    
    motion_representation = 'rot3d'
    if pose is None:
        if motion_representation=='rot6d':
            pose = torch.tensor(feature2smplx(input_path), dtype=torch.float32)
        else:
            pose = torch.tensor(np.load(input_path), dtype=torch.float32)

    hand_pose = pose[:, 66:66+90]
    # import pdb; pdb.set_trace()
    pose = torch.zeros_like(pose)
    pose[:, 66:66+90] = hand_pose
    
    if face_replace_path is not None:
        face_data = torch.tensor(np.load(face_replace_path), dtype=torch.float32)
        if face_data.shape[0] != pose.shape[0]:
            # face_motion = torch.from_numpy(face_motion)
            # import pdb; pdb.set_trace()
            face_data = face_data[None].permute(0, 2, 1) # [1, num_feats, num_frames]
            face_data = F.interpolate(face_data, size=pose.shape[0], mode='linear')   # motion interpolate
            face_data = face_data.permute(0, 2, 1)[0]   # [num_frames, num_feats]
        assert face_data.shape[1] == 50
        pose[:, 159:209] = face_data


    
    
    # selected_index = torch.tensor([2, 8, 16, 19, 38, 47, 50, 52])
    # selected_index = torch.tensor([0, 19, 39, 58, 77, 96, 116, 135])
    # selected_index = torch.tensor([0, 39,  58, 77, 116, 135])
    # pose = torch.index_select(pose, 0, selected_index)

    # pose[..., 309:309+3] = 0
    # for i in range(len(pose)):
    #     pose[i, 309] = 0.25 * i
        # pose[i, 311] = -0.25 * i

    # import pdb; pdb.set_trace()

    # pose = pose[]
    use_flame = (pose.shape[1] == 322)
    cfg.set_additional_args(use_flame=use_flame, modify_root_joint=False)
    num_frames = pose.shape[0]
    zero_pose = torch.zeros((1, 3)).float().cuda().repeat(num_frames, 1)  # eye poses
    zeor_expr = torch.zeros((1, 10)).float().cuda().repeat(num_frames, 1)  # smplx face expression
    if face_carnical:
        neck_idx = 12
        pose[:, neck_idx*3:(neck_idx+1)*3] = zero_pose
        pose[:, :3] = 0

    if use_flame:
        # 322
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation
            'pose_body': pose[:, 3:3+63].to(comp_device),  # controls the body
            'pose_hand': pose[:, 66:66+90].to(comp_device),  # controls the finger articulation
            'pose_jaw': pose[:, 66+90:66+93].to(comp_device),  # controls the yaw pose
            'face_expr': pose[:, 159:159+50].to(comp_device),  # controls the face expression
            'face_shape': pose[:, 209:209+100].to(comp_device),  # controls the face shape
            'trans': pose[:, 309:309+3].to(comp_device),  # controls the global body position
            'betas': pose[:, 312:].to(comp_device),  # controls the body shape. Body shape is static
        }
    else:
        #172
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation 3
            'pose_body': pose[:, 3:3 + 63].to(comp_device),  # controls the body 63
            'pose_hand': pose[:, 66:66 + 90].to(comp_device),  # controls the finger articulation 90
            'pose_jaw': pose[:, 66 + 90:66 + 93].to(comp_device),  # controls the yaw pose 3
            'trans': pose[:, 159:159 + 3].to(comp_device),  # controls the global body position 3
            'betas': pose[:, 162:].to(comp_device),  # controls the body shape. Body shape is static 10
        }
        
    if norm_global_orient:
        for i in range(num_frames):
            body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1] = compute_canonical_transform(body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1])

    if transform:
        trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        trans = body_parms['trans'].cpu().numpy()
        trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
        trans[:, 1] = trans[:, 1] * (-1)
        body_parms['trans'] = torch.tensor(trans, dtype=torch.float32).to(comp_device)
        body_parms['root_orient'] = compute_canonical_transform(body_parms['root_orient'])

    if use_flame:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, transl=body_parms['trans'],
                             expression=zeor_expr, flame_betas=body_parms['face_shape'], flame_expression=body_parms['face_expr'])
    else:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, expression=zeor_expr,
                             transl=body_parms['trans'])
    vertices = output.vertices
    # import pdb; pdb.set_trace()
    joints = output.joints[:, smpl_x.joint_idx, :]
    faces = output.faces

    vertices_mesh=vertices.detach().cpu().numpy()
    # import trimesh
    # for i in range(len(vertices_mesh)):
        
    #     mesh = trimesh.Trimesh(vertices=vertices_mesh[i], faces=faces)
    #     mesh.export(f'/comp_robot/lushunlin/visualization/visualization/humanml_077/humanml_077_{i}.obj')
    # import pdb; pdb.set_trace()

    return vertices, joints, pose, faces



def process_teaser_walking(input_path=None, face_replace_path=None, face_carnical=False, pose=None, norm_global_orient=False, transform=False):
    # generate mesh
    
    motion_representation = 'rot3d'
    if pose is None:
        if motion_representation=='rot6d':
            pose = torch.tensor(feature2smplx(input_path), dtype=torch.float32)
        else:
            pose = torch.tensor(np.load(input_path), dtype=torch.float32)

    # hand_pose = pose[:, 66:66+90]
    # # import pdb; pdb.set_trace()
    # pose = torch.zeros_like(pose)
    # pose[:, 66:66+90] = hand_pose
    
    if face_replace_path is not None:
        face_data = torch.tensor(np.load(face_replace_path), dtype=torch.float32)
        if face_data.shape[0] != pose.shape[0]:
            # face_motion = torch.from_numpy(face_motion)
            # import pdb; pdb.set_trace()
            face_data = face_data[None].permute(0, 2, 1) # [1, num_feats, num_frames]
            face_data = F.interpolate(face_data, size=pose.shape[0], mode='linear')   # motion interpolate
            face_data = face_data.permute(0, 2, 1)[0]   # [num_frames, num_feats]
        assert face_data.shape[1] == 50
        pose[:, 159:209] = face_data


    
    
    # selected_index = torch.tensor([2, 8, 16, 19, 38, 47, 50, 52])
    # selected_index = torch.tensor([0, 19, 39, 58, 77, 96, 116, 135])
    # selected_index = torch.tensor([0, 39,  58, 77, 116, 135])
    # pose = torch.index_select(pose, 0, selected_index)

    # pose[..., 309:309+3] = 0
    # for i in range(len(pose)):
    #     pose[i, 309] = 0.25 * i
        # pose[i, 311] = -0.25 * i

    # import pdb; pdb.set_trace()

    # pose = pose[]
    use_flame = (pose.shape[1] == 322)
    cfg.set_additional_args(use_flame=use_flame, modify_root_joint=False)
    num_frames = pose.shape[0]
    zero_pose = torch.zeros((1, 3)).float().cuda().repeat(num_frames, 1)  # eye poses
    zeor_expr = torch.zeros((1, 10)).float().cuda().repeat(num_frames, 1)  # smplx face expression
    if face_carnical:
        neck_idx = 12
        pose[:, neck_idx*3:(neck_idx+1)*3] = zero_pose
        pose[:, :3] = 0

    if use_flame:
        # 322
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation
            'pose_body': pose[:, 3:3+63].to(comp_device),  # controls the body
            'pose_hand': pose[:, 66:66+90].to(comp_device),  # controls the finger articulation
            'pose_jaw': pose[:, 66+90:66+93].to(comp_device),  # controls the yaw pose
            'face_expr': pose[:, 159:159+50].to(comp_device),  # controls the face expression
            'face_shape': pose[:, 209:209+100].to(comp_device),  # controls the face shape
            'trans': pose[:, 309:309+3].to(comp_device),  # controls the global body position
            'betas': pose[:, 312:].to(comp_device),  # controls the body shape. Body shape is static
        }
    else:
        #172
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation 3
            'pose_body': pose[:, 3:3 + 63].to(comp_device),  # controls the body 63
            'pose_hand': pose[:, 66:66 + 90].to(comp_device),  # controls the finger articulation 90
            'pose_jaw': pose[:, 66 + 90:66 + 93].to(comp_device),  # controls the yaw pose 3
            'trans': pose[:, 159:159 + 3].to(comp_device),  # controls the global body position 3
            'betas': pose[:, 162:].to(comp_device),  # controls the body shape. Body shape is static 10
        }
        
    if norm_global_orient:
        for i in range(num_frames):
            body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1] = compute_canonical_transform(body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1])

    if transform:
        trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        trans = body_parms['trans'].cpu().numpy()
        trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
        trans[:, 1] = trans[:, 1] * (-1)
        body_parms['trans'] = torch.tensor(trans, dtype=torch.float32).to(comp_device)
        body_parms['root_orient'] = compute_canonical_transform(body_parms['root_orient'])

    if use_flame:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, transl=body_parms['trans'],
                             expression=zeor_expr, flame_betas=body_parms['face_shape'], flame_expression=body_parms['face_expr'])
    else:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, expression=zeor_expr,
                             transl=body_parms['trans'])
    vertices = output.vertices
    # import pdb; pdb.set_trace()
    joints = output.joints[:, smpl_x.joint_idx, :]
    faces = output.faces

    vertices_mesh=vertices.detach().cpu().numpy()
    # import trimesh
    # for i in range(len(vertices_mesh)):
        
    #     mesh = trimesh.Trimesh(vertices=vertices_mesh[i], faces=faces)
    #     mesh.export(f'/comp_robot/lushunlin/visualization/visualization/humanml_077/humanml_077_{i}.obj')
    # import pdb; pdb.set_trace()

    return vertices, joints, pose, faces



def process_teaser_hiss(input_path=None, face_replace_path=None, face_carnical=False, pose=None, norm_global_orient=False, transform=False):
    # generate mesh
    
    motion_representation = 'rot3d'
    if pose is None:
        if motion_representation=='rot6d':
            pose = torch.tensor(feature2smplx(input_path), dtype=torch.float32)
        else:
            pose = torch.tensor(np.load(input_path), dtype=torch.float32)

    # pose_jaw = pose[:, 66+90:66+93]
    # pose = torch.zeros_like(pose)
    # pose[:, 66+90:66+93] = pose_jaw
    
    if face_replace_path is not None:
        face_data = torch.tensor(np.load(face_replace_path), dtype=torch.float32)
        if face_data.shape[0] != pose.shape[0]:
            # face_motion = torch.from_numpy(face_motion)
            # import pdb; pdb.set_trace()
            face_data = face_data[None].permute(0, 2, 1) # [1, num_feats, num_frames]
            face_data = F.interpolate(face_data, size=pose.shape[0], mode='linear')   # motion interpolate
            face_data = face_data.permute(0, 2, 1)[0]   # [num_frames, num_feats]
        assert face_data.shape[1] == 53
        pose[:, 156:209] = face_data


    # import pdb; pdb.set_trace()
    selected_index = torch.tensor([16, 24, 43, 70])
    # selected_index = torch.tensor([0, 19, 39, 58, 77, 96, 116, 135])
    # selected_index = torch.tensor([0, 39,  58, 77, 116, 135])
    pose = torch.index_select(pose, 0, selected_index)

    pose[..., 309:309+3] = 0
    for i in range(len(pose)):
        pose[i, 309] = 0.65 * i
        # if i == 3:
        #     pose[i, 309] = 0.6 * i
        # pose[i, 311] = -0.25 * i

    # import pdb; pdb.set_trace()

    # pose = pose[]
    use_flame = (pose.shape[1] == 322)
    cfg.set_additional_args(use_flame=use_flame, modify_root_joint=False)
    num_frames = pose.shape[0]
    zero_pose = torch.zeros((1, 3)).float().cuda().repeat(num_frames, 1)  # eye poses
    zeor_expr = torch.zeros((1, 10)).float().cuda().repeat(num_frames, 1)  # smplx face expression
    if face_carnical:
        neck_idx = 12
        pose[:, neck_idx*3:(neck_idx+1)*3] = zero_pose
        pose[:, :3] = 0

    if use_flame:
        # 322
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation
            'pose_body': pose[:, 3:3+63].to(comp_device),  # controls the body
            'pose_hand': pose[:, 66:66+90].to(comp_device),  # controls the finger articulation
            'pose_jaw': pose[:, 66+90:66+93].to(comp_device),  # controls the yaw pose
            'face_expr': pose[:, 159:159+50].to(comp_device),  # controls the face expression
            'face_shape': pose[:, 209:209+100].to(comp_device),  # controls the face shape
            'trans': pose[:, 309:309+3].to(comp_device),  # controls the global body position
            'betas': pose[:, 312:].to(comp_device),  # controls the body shape. Body shape is static
        }
    else:
        #172
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation 3
            'pose_body': pose[:, 3:3 + 63].to(comp_device),  # controls the body 63
            'pose_hand': pose[:, 66:66 + 90].to(comp_device),  # controls the finger articulation 90
            'pose_jaw': pose[:, 66 + 90:66 + 93].to(comp_device),  # controls the yaw pose 3
            'trans': pose[:, 159:159 + 3].to(comp_device),  # controls the global body position 3
            'betas': pose[:, 162:].to(comp_device),  # controls the body shape. Body shape is static 10
        }
        
    if norm_global_orient:
        for i in range(num_frames):
            body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1] = compute_canonical_transform(body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1])

    if transform:
        trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        trans = body_parms['trans'].cpu().numpy()
        trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
        trans[:, 1] = trans[:, 1] * (-1)
        body_parms['trans'] = torch.tensor(trans, dtype=torch.float32).to(comp_device)
        body_parms['root_orient'] = compute_canonical_transform(body_parms['root_orient'])

    if use_flame:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, transl=body_parms['trans'],
                             expression=zeor_expr, flame_betas=body_parms['face_shape'], flame_expression=body_parms['face_expr'])
    else:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, expression=zeor_expr,
                             transl=body_parms['trans'])
    vertices = output.vertices
    # import pdb; pdb.set_trace()
    joints = output.joints[:, smpl_x.joint_idx, :]
    faces = output.faces

    vertices_mesh=vertices.detach().cpu().numpy()
    # import trimesh
    # for i in range(len(vertices_mesh)):
        
    #     mesh = trimesh.Trimesh(vertices=vertices_mesh[i], faces=faces)
    #     mesh.export(f'/comp_robot/lushunlin/visualization/visualization/humanml_077/humanml_077_{i}.obj')
    # import pdb; pdb.set_trace()

    return vertices, joints, pose, faces


def process_teaser_ballet(input_path=None, face_replace_path=None, face_carnical=False, pose=None, norm_global_orient=False, transform=False):
    # generate mesh
    
    motion_representation = 'rot3d'
    if pose is None:
        if motion_representation=='rot6d':
            pose = torch.tensor(feature2smplx(input_path), dtype=torch.float32)
        else:
            pose = torch.tensor(np.load(input_path), dtype=torch.float32)

    # pose_jaw = pose[:, 66+90:66+93]
    # pose = torch.zeros_like(pose)
    # pose[:, 66+90:66+93] = pose_jaw
    
    if face_replace_path is not None:
        face_data = torch.tensor(np.load(face_replace_path), dtype=torch.float32)
        if face_data.shape[0] != pose.shape[0]:
            # face_motion = torch.from_numpy(face_motion)
            # import pdb; pdb.set_trace()
            face_data = face_data[None].permute(0, 2, 1) # [1, num_feats, num_frames]
            face_data = F.interpolate(face_data, size=pose.shape[0], mode='linear')   # motion interpolate
            face_data = face_data.permute(0, 2, 1)[0]   # [num_frames, num_feats]
        assert face_data.shape[1] == 50
        pose[:, 159:209] = face_data


    # import pdb; pdb.set_trace()
    selected_index = torch.tensor([0, 60, 120, 180])
    # selected_index = torch.tensor([0, 19, 39, 58, 77, 96, 116, 135])
    # selected_index = torch.tensor([0, 39,  58, 77, 116, 135])
    pose = torch.index_select(pose, 0, selected_index)

    pose[..., 309:309+3] = 0
    for i in range(len(pose)):
        pose[i, 309] = 0.65 * i
        # if i == 3:
        #     pose[i, 309] = 0.6 * i
        # pose[i, 311] = -0.25 * i

    # import pdb; pdb.set_trace()

    # pose = pose[]
    use_flame = (pose.shape[1] == 322)
    cfg.set_additional_args(use_flame=use_flame, modify_root_joint=False)
    num_frames = pose.shape[0]
    zero_pose = torch.zeros((1, 3)).float().cuda().repeat(num_frames, 1)  # eye poses
    zeor_expr = torch.zeros((1, 10)).float().cuda().repeat(num_frames, 1)  # smplx face expression
    if face_carnical:
        neck_idx = 12
        pose[:, neck_idx*3:(neck_idx+1)*3] = zero_pose
        pose[:, :3] = 0

    if use_flame:
        # 322
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation
            'pose_body': pose[:, 3:3+63].to(comp_device),  # controls the body
            'pose_hand': pose[:, 66:66+90].to(comp_device),  # controls the finger articulation
            'pose_jaw': pose[:, 66+90:66+93].to(comp_device),  # controls the yaw pose
            'face_expr': pose[:, 159:159+50].to(comp_device),  # controls the face expression
            'face_shape': pose[:, 209:209+100].to(comp_device),  # controls the face shape
            'trans': pose[:, 309:309+3].to(comp_device),  # controls the global body position
            'betas': pose[:, 312:].to(comp_device),  # controls the body shape. Body shape is static
        }
    else:
        #172
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation 3
            'pose_body': pose[:, 3:3 + 63].to(comp_device),  # controls the body 63
            'pose_hand': pose[:, 66:66 + 90].to(comp_device),  # controls the finger articulation 90
            'pose_jaw': pose[:, 66 + 90:66 + 93].to(comp_device),  # controls the yaw pose 3
            'trans': pose[:, 159:159 + 3].to(comp_device),  # controls the global body position 3
            'betas': pose[:, 162:].to(comp_device),  # controls the body shape. Body shape is static 10
        }
        
    if norm_global_orient:
        for i in range(num_frames):
            body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1] = compute_canonical_transform(body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1])

    if transform:
        trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        trans = body_parms['trans'].cpu().numpy()
        trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
        trans[:, 1] = trans[:, 1] * (-1)
        body_parms['trans'] = torch.tensor(trans, dtype=torch.float32).to(comp_device)
        body_parms['root_orient'] = compute_canonical_transform(body_parms['root_orient'])

    if use_flame:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, transl=body_parms['trans'],
                             expression=zeor_expr, flame_betas=body_parms['face_shape'], flame_expression=body_parms['face_expr'])
    else:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, expression=zeor_expr,
                             transl=body_parms['trans'])
    vertices = output.vertices
    # import pdb; pdb.set_trace()
    joints = output.joints[:, smpl_x.joint_idx, :]
    faces = output.faces

    vertices_mesh=vertices.detach().cpu().numpy()
    # import trimesh
    # for i in range(len(vertices_mesh)):
        
    #     mesh = trimesh.Trimesh(vertices=vertices_mesh[i], faces=faces)
    #     mesh.export(f'/comp_robot/lushunlin/visualization/visualization/humanml_077/humanml_077_{i}.obj')
    # import pdb; pdb.set_trace()

    return vertices, joints, pose, faces

def process_hand(input_path=None, face_replace_path=None, face_carnical=False, pose=None, norm_global_orient=False, transform=False):
    # generate mesh
    motion_representation = 'rot3d'
    if pose is None:
        if motion_representation=='rot6d':
            pose = torch.tensor(feature2smplx(input_path), dtype=torch.float32)
        else:
            pose = torch.tensor(np.load(input_path), dtype=torch.float32)
    
    pose = torch.zeros_like(pose)
    
    if face_replace_path is not None:
        face_data = torch.tensor(np.load(face_replace_path), dtype=torch.float32)
        if face_data.shape[0] != pose.shape[0]:
            # face_motion = torch.from_numpy(face_motion)
            # import pdb; pdb.set_trace()
            face_data = face_data[None].permute(0, 2, 1) # [1, num_feats, num_frames]
            face_data = F.interpolate(face_data, size=pose.shape[0], mode='linear')   # motion interpolate
            face_data = face_data.permute(0, 2, 1)[0]   # [num_frames, num_feats]
        assert face_data.shape[1] == 50
        pose[:, 159:209] = face_data

    use_flame = (pose.shape[1] == 322)
    cfg.set_additional_args(use_flame=use_flame, modify_root_joint=False)
    num_frames = pose.shape[0]
    zero_pose = torch.zeros((1, 3)).float().cuda().repeat(num_frames, 1)  # eye poses
    zeor_expr = torch.zeros((1, 10)).float().cuda().repeat(num_frames, 1)  # smplx face expression
    if face_carnical:
        neck_idx = 12
        pose[:, neck_idx*3:(neck_idx+1)*3] = zero_pose
        pose[:, :3] = 0

    if use_flame:
        # 322
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation
            'pose_body': pose[:, 3:3+63].to(comp_device),  # controls the body
            'pose_hand': pose[:, 66:66+90].to(comp_device),  # controls the finger articulation
            'pose_jaw': pose[:, 66+90:66+93].to(comp_device),  # controls the yaw pose
            'face_expr': pose[:, 159:159+50].to(comp_device),  # controls the face expression
            'face_shape': pose[:, 209:209+100].to(comp_device),  # controls the face shape
            'trans': pose[:, 309:309+3].to(comp_device),  # controls the global body position
            'betas': pose[:, 312:].to(comp_device),  # controls the body shape. Body shape is static
        }
    else:
        #172
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation 3
            'pose_body': pose[:, 3:3 + 63].to(comp_device),  # controls the body 63
            'pose_hand': pose[:, 66:66 + 90].to(comp_device),  # controls the finger articulation 90
            'pose_jaw': pose[:, 66 + 90:66 + 93].to(comp_device),  # controls the yaw pose 3
            'trans': pose[:, 159:159 + 3].to(comp_device),  # controls the global body position 3
            'betas': pose[:, 162:].to(comp_device),  # controls the body shape. Body shape is static 10
        }
        
    if norm_global_orient:
        for i in range(num_frames):
            body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1] = compute_canonical_transform(body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1])

    if transform:
        trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        trans = body_parms['trans'].cpu().numpy()
        trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
        trans[:, 1] = trans[:, 1] * (-1)
        body_parms['trans'] = torch.tensor(trans, dtype=torch.float32).to(comp_device)
        body_parms['root_orient'] = compute_canonical_transform(body_parms['root_orient'])

    if use_flame:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, transl=body_parms['trans'],
                             expression=zeor_expr, flame_betas=body_parms['face_shape'], flame_expression=body_parms['face_expr'])
    else:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, expression=zeor_expr,
                             transl=body_parms['trans'])
    vertices = output.vertices
    # import pdb; pdb.set_trace()
    joints = output.joints[:, smpl_x.joint_idx, :]
    faces = output.faces

    vertices_mesh=vertices.detach().cpu().numpy()
    # import trimesh
    # for i in range(len(vertices_mesh)):
        
    #     mesh = trimesh.Trimesh(vertices=vertices_mesh[i], faces=faces)
    #     mesh.export(f'/comp_robot/lushunlin/visualization/visualization/humanml_077/humanml_077_{i}.obj')
    # import pdb; pdb.set_trace()

    return vertices, joints, pose, faces


def process_only_face(input_path=None, face_replace_path=None, face_carnical=False, pose=None, norm_global_orient=False, transform=False):
    # generate mesh
    motion_representation = 'rot3d'
    if pose is None:
        if motion_representation=='rot6d':
            pose = torch.tensor(feature2smplx(input_path), dtype=torch.float32)
        else:
            pose = torch.tensor(np.load(input_path), dtype=torch.float32)
    
    pose = torch.zeros_like(pose)

    if face_replace_path is not None:
        face_data = torch.tensor(np.load(face_replace_path), dtype=torch.float32)
        if face_data.shape[0] != pose.shape[0]:
            # face_motion = torch.from_numpy(face_motion)
            # import pdb; pdb.set_trace()
            face_data = face_data[None].permute(0, 2, 1) # [1, num_feats, num_frames]
            face_data = F.interpolate(face_data, size=pose.shape[0], mode='linear')   # motion interpolate
            face_data = face_data.permute(0, 2, 1)[0]   # [num_frames, num_feats]
        assert face_data.shape[1] == 50
        pose[:, 159:209] = face_data

    use_flame = (pose.shape[1] == 322)
    cfg.set_additional_args(use_flame=use_flame, modify_root_joint=False)
    num_frames = pose.shape[0]
    zero_pose = torch.zeros((1, 3)).float().cuda().repeat(num_frames, 1)  # eye poses
    zeor_expr = torch.zeros((1, 10)).float().cuda().repeat(num_frames, 1)  # smplx face expression
    if face_carnical:
        neck_idx = 12
        pose[:, neck_idx*3:(neck_idx+1)*3] = zero_pose
        pose[:, :3] = 0

    if use_flame:
        # 322
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation
            'pose_body': pose[:, 3:3+63].to(comp_device),  # controls the body
            'pose_hand': pose[:, 66:66+90].to(comp_device),  # controls the finger articulation
            'pose_jaw': pose[:, 66+90:66+93].to(comp_device),  # controls the yaw pose
            'face_expr': pose[:, 159:159+50].to(comp_device),  # controls the face expression
            'face_shape': pose[:, 209:209+100].to(comp_device),  # controls the face shape
            'trans': pose[:, 309:309+3].to(comp_device),  # controls the global body position
            'betas': pose[:, 312:].to(comp_device),  # controls the body shape. Body shape is static
        }
    else:
        #172
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation 3
            'pose_body': pose[:, 3:3 + 63].to(comp_device),  # controls the body 63
            'pose_hand': pose[:, 66:66 + 90].to(comp_device),  # controls the finger articulation 90
            'pose_jaw': pose[:, 66 + 90:66 + 93].to(comp_device),  # controls the yaw pose 3
            'trans': pose[:, 159:159 + 3].to(comp_device),  # controls the global body position 3
            'betas': pose[:, 162:].to(comp_device),  # controls the body shape. Body shape is static 10
        }
        
    if norm_global_orient:
        for i in range(num_frames):
            body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1] = compute_canonical_transform(body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1])

    if transform:
        trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        trans = body_parms['trans'].cpu().numpy()
        trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
        trans[:, 1] = trans[:, 1] * (-1)
        body_parms['trans'] = torch.tensor(trans, dtype=torch.float32).to(comp_device)
        body_parms['root_orient'] = compute_canonical_transform(body_parms['root_orient'])

    if use_flame:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, transl=body_parms['trans'],
                             expression=zeor_expr, flame_betas=body_parms['face_shape'], flame_expression=body_parms['face_expr'])
    else:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, expression=zeor_expr,
                             transl=body_parms['trans'])
    vertices = output.vertices
    # import pdb; pdb.set_trace()
    joints = output.joints[:, smpl_x.joint_idx, :]
    faces = output.faces

    vertices_mesh=vertices.detach().cpu().numpy()
    # import trimesh
    # # for i in range(len(vertices_mesh)):
        
    # mesh = trimesh.Trimesh(vertices=vertices_mesh[0], faces=faces)
    # mesh.export(f'/comp_robot/lushunlin/visualization/visualization/humanml_077/humanml_077.obj')
    # import pdb; pdb.set_trace()

    return vertices, joints, pose, faces


# def read_face_shape(path):
    
#     folder_path = os.path.join(os.path.dirname(path).split('/')[-1].split('_')[0], os.path.dirname(path).split('/')[-1].upper())
#     data_path = os.path.join('/comp_robot/linjing/data/Motion_Generation/face_data/BAUM/codes', folder_path)
#     result_data = []

#     face_shape = np.load(path)


#     face_shape = torch.tensor(face_shape, dtype=torch.float32)
#     return face_shape



def process_only_face_jaw(input_path=None, face_replace_path=None, face_carnical=False, pose=None, norm_global_orient=False, transform=False):
    # generate mesh
    motion_representation = 'rot3d'

    if face_replace_path is not None:
        face_data = torch.tensor(np.load(face_replace_path), dtype=torch.float32)
        face_frame = face_data.shape[0]
        # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    # if pose is None:
    #     if motion_representation=='rot6d':
    #         pose = torch.tensor(feature2smplx(input_path), dtype=torch.float32)
    #     else:
    #         pose = torch.tensor(np.load(input_path), dtype=torch.float32)
    
    pose = torch.zeros((face_frame, 322))

    if face_replace_path is not None:
        face_data = torch.tensor(np.load(face_replace_path), dtype=torch.float32)
        assert face_data.shape[0] == pose.shape[0]
        # if face_data.shape[0] != pose.shape[0]:
        #     # face_motion = torch.from_numpy(face_motion)
        #     # import pdb; pdb.set_trace()
        #     face_data = face_data[None].permute(0, 2, 1) # [1, num_feats, num_frames]
        #     face_data = F.interpolate(face_data, size=pose.shape[0], mode='linear')   # motion interpolate
        #     face_data = face_data.permute(0, 2, 1)[0]   # [num_frames, num_feats]
        # import pdb; pdb.set_trace()
        # if 'shape' not in face_replace_path:
        assert face_data.shape[1] == 53
        pose[:, 156:209] = face_data
        # else:
        #     pose[:, 156:209] = face_data[:, :53]
        #     pose[:, 159:209] = face_data[:, 53:]
        # pose[:, 156:159] = face_data[:, -3:]
        # pose[:, 159:209] = face_data[:, :-3]
        # import pdb; pdb.set_trace()

    use_flame = (pose.shape[1] == 322)
    cfg.set_additional_args(use_flame=use_flame, modify_root_joint=False)
    num_frames = pose.shape[0]
    zero_pose = torch.zeros((1, 3)).float().cuda().repeat(num_frames, 1)  # eye poses
    zeor_expr = torch.zeros((1, 10)).float().cuda().repeat(num_frames, 1)  # smplx face expression
    if face_carnical:
        neck_idx = 12
        pose[:, neck_idx*3:(neck_idx+1)*3] = zero_pose
        pose[:, :3] = 0

    if use_flame:
        # 322
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation
            'pose_body': pose[:, 3:3+63].to(comp_device),  # controls the body
            'pose_hand': pose[:, 66:66+90].to(comp_device),  # controls the finger articulation
            'pose_jaw': pose[:, 66+90:66+93].to(comp_device),  # controls the yaw pose
            'face_expr': pose[:, 159:159+50].to(comp_device),  # controls the face expression
            'face_shape': pose[:, 209:209+100].to(comp_device),  # controls the face shape
            'trans': pose[:, 309:309+3].to(comp_device),  # controls the global body position
            'betas': pose[:, 312:].to(comp_device),  # controls the body shape. Body shape is static
        }
    else:
        #172
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation 3
            'pose_body': pose[:, 3:3 + 63].to(comp_device),  # controls the body 63
            'pose_hand': pose[:, 66:66 + 90].to(comp_device),  # controls the finger articulation 90
            'pose_jaw': pose[:, 66 + 90:66 + 93].to(comp_device),  # controls the yaw pose 3
            'trans': pose[:, 159:159 + 3].to(comp_device),  # controls the global body position 3
            'betas': pose[:, 162:].to(comp_device),  # controls the body shape. Body shape is static 10
        }
        
    if norm_global_orient:
        for i in range(num_frames):
            body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1] = compute_canonical_transform(body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1])

    if transform:
        trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        trans = body_parms['trans'].cpu().numpy()
        trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
        trans[:, 1] = trans[:, 1] * (-1)
        body_parms['trans'] = torch.tensor(trans, dtype=torch.float32).to(comp_device)
        body_parms['root_orient'] = compute_canonical_transform(body_parms['root_orient'])

    if use_flame:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, transl=body_parms['trans'],
                             expression=zeor_expr, flame_betas=body_parms['face_shape'], flame_expression=body_parms['face_expr'])
    else:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, expression=zeor_expr,
                             transl=body_parms['trans'])
    vertices = output.vertices
    # import pdb; pdb.set_trace()
    joints = output.joints[:, smpl_x.joint_idx, :]
    faces = output.faces

    vertices_mesh=vertices.detach().cpu().numpy()
    # import trimesh
    # for i in range(len(vertices_mesh)):
        
    #     mesh = trimesh.Trimesh(vertices=vertices_mesh[i], faces=faces)
    #     mesh.export(f'/comp_robot/lushunlin/visualization/visualization/humanml_077/humanml_077_{i}.obj')
    # import pdb; pdb.set_trace()

    return vertices, joints, pose, faces

def process_smplx_data(smplx_data=None, face_carnical=False, pose=None, norm_global_orient=False, transform=False):
    # generate mesh
    # motion_representation = 'rot3d'
    # if pose is None:
    #     if motion_representation=='rot6d':
    #         pose = torch.tensor(feature2smplx(input_path), dtype=torch.float32)
    #     else:
    #         pose = torch.tensor(np.load(input_path), dtype=torch.float32)
    
    pose = torch.tensor(smplx_data, dtype=torch.float32)
    assert pose.shape[1] == 322
    # import pdb; pdb.set_trace()
    use_flame = (pose.shape[1] == 322)
    cfg.set_additional_args(use_flame=use_flame, modify_root_joint=False)
    num_frames = pose.shape[0]
    zero_pose = torch.zeros((1, 3)).float().cuda().repeat(num_frames, 1)  # eye poses
    zeor_expr = torch.zeros((1, 10)).float().cuda().repeat(num_frames, 1)  # smplx face expression
    if face_carnical:
        neck_idx = 12
        pose[:, neck_idx*3:(neck_idx+1)*3] = zero_pose
        pose[:, :3] = 0

    if use_flame:
        # 322
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation
            'pose_body': pose[:, 3:3+63].to(comp_device),  # controls the body
            'pose_hand': pose[:, 66:66+90].to(comp_device),  # controls the finger articulation
            'pose_jaw': pose[:, 66+90:66+93].to(comp_device),  # controls the yaw pose
            'face_expr': pose[:, 159:159+50].to(comp_device),  # controls the face expression
            'face_shape': pose[:, 209:209+100].to(comp_device),  # controls the face shape
            'trans': pose[:, 309:309+3].to(comp_device),  # controls the global body position
            'betas': pose[:, 312:].to(comp_device),  # controls the body shape. Body shape is static
        }
    else:
        #172
        body_parms = {
            'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation 3
            'pose_body': pose[:, 3:3 + 63].to(comp_device),  # controls the body 63
            'pose_hand': pose[:, 66:66 + 90].to(comp_device),  # controls the finger articulation 90
            'pose_jaw': pose[:, 66 + 90:66 + 93].to(comp_device),  # controls the yaw pose 3
            'trans': pose[:, 159:159 + 3].to(comp_device),  # controls the global body position 3
            'betas': pose[:, 162:].to(comp_device),  # controls the body shape. Body shape is static 10
        }
        
    if norm_global_orient:
        for i in range(num_frames):
            body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1] = compute_canonical_transform(body_parms['root_orient'][i:i+1], body_parms['trans'][i:i+1])

    if transform:
        trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        trans = body_parms['trans'].cpu().numpy()
        trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
        trans[:, 1] = trans[:, 1] * (-1)
        body_parms['trans'] = torch.tensor(trans, dtype=torch.float32).to(comp_device)
        body_parms['root_orient'] = compute_canonical_transform(body_parms['root_orient'])

    if use_flame:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, transl=body_parms['trans'],
                             expression=zeor_expr, flame_betas=body_parms['face_shape'], flame_expression=body_parms['face_expr'])
    else:
        output = smplx_layer(betas=body_parms['betas'], body_pose=body_parms['pose_body'], global_orient=body_parms['root_orient'],
                             left_hand_pose=body_parms['pose_hand'][:, :45], right_hand_pose=body_parms['pose_hand'][:, 45:],
                             jaw_pose=body_parms['pose_jaw'], leye_pose=zero_pose, reye_pose=zero_pose, expression=zeor_expr,
                             transl=body_parms['trans'])
    vertices = output.vertices
    # import pdb; pdb.set_trace()
    joints = output.joints
    faces = output.faces

    # import trimesh
    # mesh = trimesh.load('/home/linjing/code/OSX/demo/output.obj')
    # mesh.vertices = vertices[0].cpu().numpy()
    # mesh.export('/home/linjing/exp/align_global_orient/wo_transform.obj')

    return vertices, joints, pose, faces

def orient_align(input_path=None):
    pose = np.load(input_path)
    trans = pose[:, 309:309+3]
    trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    trans = np.dot(trans, trans_matrix)  # exchange the y and z axis
    trans[:, 2] = trans[:, 2] * (-1)
    root_orient = torch.from_numpy(pose[:, :3]).float().cuda()
    root_orient = compute_canonical_transform(root_orient)

    if 'AIST' in input_path:
        trans = trans/94.0
    elif 'NTU' in input_path:
        trans = trans/78.0

    pose[:, 309:309+3] = trans
    pose[:, :3] = root_orient.cpu().numpy()

    return pose


def process_mano(input_path, pose=None):
    # generate mesh
    motion_representation = 'rot3d'
    if pose is None:
        if motion_representation=='rot6d':
            pose = torch.tensor(feature2smplx(input_path), dtype=torch.float32)
        else:
            pose = torch.tensor(np.load(input_path), dtype=torch.float32)
    use_flame = (pose.shape[1] == 322)
    cfg.set_additional_args(use_flame=use_flame, modify_root_joint=False)
    body_parms = {
        'root_orient': pose[:, :3].to(comp_device),  # controls the global root orientation
        'pose_body': pose[:, 3:3+63].to(comp_device),  # controls the body
        'pose_left_hand': pose[:, 66:66+45].to(comp_device),  # controls the finger articulation
        'pose_right_hand': pose[:, 66+45:66+90].to(comp_device),  # controls the finger articulation
        'pose_jaw': pose[:, 66+90:66+93].to(comp_device),  # controls the yaw pose
        'face_expr': pose[:, 159:159+50].to(comp_device),  # controls the face expression
        'face_shape': pose[:, 209:209+100].to(comp_device),  # controls the face shape
        'trans': pose[:, 309:309+3].to(comp_device),  # controls the global body position
        'betas': pose[:, 312:].to(comp_device),  # controls the body shape. Body shape is static
    }
    num_frames = pose.shape[0]
    zero_betas = torch.zeros((1, 10)).float().cuda().repeat(num_frames, 1)  # eye poses

    lhand_output = lhand_mano_layer(global_orient=body_parms['root_orient'], hand_pose=body_parms['pose_left_hand'], betas=zero_betas)
    lhand_vertices = lhand_output.vertices
    lhand_joints = lhand_output.joints

    rhand_output = rhand_mano_layer(global_orient=body_parms['root_orient'], hand_pose=body_parms['pose_right_hand'], betas=zero_betas)
    rhand_vertices = rhand_output.vertices
    rhand_joints = rhand_output.joints

    return lhand_vertices, lhand_joints, rhand_vertices, rhand_joints

# def save_mesh(verts, save_path, transpose=False):
#     verts = verts.cpu().numpy()
#
#     if transpose:
#         trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
#         verts = np.dot(verts, trans_matrix)  # exchange the y and z axis
#         verts[:, :, 2] = verts[:, :, 2] * (-1)
#
#     degree = math.pi
#     degree = math.pi/2
#     orient_matrix = [[math.cos(degree), -math.sin(degree), 0],
#                      [math.sin(degree), math.cos(degree), 0],
#                      [0, 0, 1]]
#     rotate_matrix = np.array(orient_matrix)
#     verts = np.dot(verts, rotate_matrix)  # rotate 90 degree by z axis
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     np.save(save_path, verts)


def save_mesh(verts, save_path, transpose=False):
    if transpose:
        trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        trans_matrix = torch.from_numpy(trans_matrix).float().cuda()
        verts = torch.matmul(verts, trans_matrix)

        import trimesh
        mesh = trimesh.load('/home/linjing/code/OSX/demo/output.obj')
        verts[:, :, 2] = verts[:, :, 2] * (-1)
        mesh.vertices = verts[0].cpu().numpy()
        mesh.fix_normals()
        mesh.export('/home/linjing/exp/align_global_orient/transpose.obj')

    degree = math.pi/2
    orient_matrix = [[math.cos(degree), -math.sin(degree), 0],
                     [math.sin(degree), math.cos(degree), 0],
                     [0, 0, 1]]
    rotate_matrix = np.array(orient_matrix)
    rotate_matrix = torch.from_numpy(rotate_matrix).float().cuda()
    verts = torch.matmul(verts, rotate_matrix)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, verts.cpu().numpy())

if __name__ == '__main__':
    mesh = np.load('/home/linjing/exp/EMoGen/generated_motion_vis/mld_origin/samples_2023-03-08-21-33-55/Example_50_batch0_0_mesh.npy')
    mesh = torch.from_numpy(mesh).float().cuda()
    save_mesh(mesh, '/home/linjing/exp/EMoGen/presentation/page7/mld/1.npy', True)
