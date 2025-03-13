import torch
import numpy as np
import argparse
import pickle
import smplx

from utils import bvh, quat
from utils.face_z_align_util import rotation_6d_to_matrix, matrix_to_axis_angle
from tqdm import tqdm
import os

def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="body_models/human_model_files")
    parser.add_argument("--model_type", type=str, default="smpl", choices=["smpl", "smplx"])
    parser.add_argument("--gender", type=str, default="NEUTRAL", choices=["MALE", "FEMALE", "NEUTRAL"])
    parser.add_argument("--num_betas", type=int, default=10, choices=[10, 300])
    parser.add_argument("--poses", type=str, default="./output/Representation_272")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--output", type=str, default="./output/Representation_272")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--is_folder", action="store_true")
    return parser.parse_args()

def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.
    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def mirror_rot_trans(lrot, trans, names, parents):
    joints_mirror = np.array([(
        names.index("Left"+n[5:]) if n.startswith("Right") else (
        names.index("Right"+n[4:]) if n.startswith("Left") else 
        names.index(n))) for n in names])

    mirror_pos = np.array([-1, 1, 1])
    mirror_rot = np.array([1, 1, -1, -1])
    grot = quat.fk_rot(lrot, parents)
    trans_mirror = mirror_pos * trans
    grot_mirror = mirror_rot * grot[:,joints_mirror]
    
    return quat.ik_rot(grot_mirror, parents), trans_mirror


def accumulate_rotations(relative_rotations):
    """Accumulate relative rotations to get overall rotation"""
    # Initial rotation is rotation matrix
    R_total = [relative_rotations[0]]
    # Iterate through all relative rotations, accumulating them step by step
    for R_rel in relative_rotations[1:]:
        R_total.append(np.matmul(R_rel, R_total[-1]))
    
    return np.array(R_total)

def rotations_matrix_to_smplx85(rotations_matrix, translation):
    
    nfrm, njoint, _, _ = rotations_matrix.shape
    axis_angle = matrix_to_axis_angle(torch.from_numpy(rotations_matrix)).numpy().reshape(nfrm, -1)
    smplx_85 = np.concatenate([axis_angle, np.zeros((nfrm, 6)), translation, np.zeros((nfrm, 10))], axis=-1)
    return smplx_85


def recover_from_local_rotation(final_x, njoint):
    # take rotations_matrix: 

    nfrm, _ = final_x.shape
    rotations_matrix = rotation_6d_to_matrix(torch.from_numpy(final_x[:,8+6*njoint:8+12*njoint]).reshape(nfrm, -1, 6)).numpy()
    global_heading_diff_rot = final_x[:,2:8]
    velocities_root_xy_no_heading = final_x[:,:2]
    positions_no_heading = final_x[:, 8:8+3*njoint].reshape(nfrm, -1, 3)
    height = positions_no_heading[:, 0, 1]

    global_heading_rot = accumulate_rotations(rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy())
    inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
    # recover root rotation

    rotations_matrix[:,0,...] = np.matmul(inv_global_heading_rot, rotations_matrix[:,0,...])

    velocities_root_xyz_no_heading = np.zeros((velocities_root_xy_no_heading.shape[0], 3))
    velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
    velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
    velocities_root_xyz_no_heading[1:, :] = np.matmul(inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :,None]).squeeze(-1)
    root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)
    root_translation[:, 1] = height
    smplx_85 = rotations_matrix_to_smplx85(rotations_matrix, root_translation)
    return smplx_85


def smpl2bvh(model_path:str, poses:str, output:str, mirror:bool,
             model_type="smpl", gender="MALE", 
             num_betas=10, fps=60) -> None:
    """Save bvh file created by smpl parameters.

    Args:
        model_path (str): Path to smpl models.
        poses (str): Path to npz or pkl file.
        output (str): Where to save bvh.
        mirror (bool): Whether save mirror motion or not.
        model_type (str, optional): I prepared "smpl" only. Defaults to "smpl".
        gender (str, optional): Gender Information. Defaults to "MALE".
        num_betas (int, optional): How many pca parameters to use in SMPL. Defaults to 10.
        fps (int, optional): Frame per second. Defaults to 30.
    """
    
    names = [
        "Pelvis",
        "Left_hip",
        "Right_hip",
        "Spine1",
        "Left_knee",
        "Right_knee",
        "Spine2",
        "Left_ankle",
        "Right_ankle",
        "Spine3",
        "Left_foot",
        "Right_foot",
        "Neck",
        "Left_collar",
        "Right_collar",
        "Head",
        "Left_shoulder",
        "Right_shoulder",
        "Left_elbow",
        "Right_elbow",
        "Left_wrist",
        "Right_wrist",
        "Left_palm",
        "Right_palm",
    ]
    
    model = smplx.create(model_path=model_path, 
                        model_type=model_type,
                        gender=gender, 
                        batch_size=1)
    
    parents = model.parents.detach().cpu().numpy()
    # You can define betas like this.(default betas are 0 at all.)
    rest = model(
        # betas = torch.randn([1, num_betas], dtype=torch.float32)
    )
    rest_pose = rest.joints.detach().cpu().numpy().squeeze()[:24,:]
    
    root_offset = rest_pose[0]
    offsets = rest_pose - rest_pose[parents]
    offsets[0] = root_offset
    offsets *= 1

    
    scaling = None
    

    poses = np.load(poses)
    assert poses.shape[-1] == 272
    
    poses = recover_from_local_rotation(poses, 22)
    assert poses.shape[-1] == 85
    
    rots = poses[:, :72].reshape(-1, 24, 3)
    trans = poses[:, 72:75]
    
    
    if scaling is not None:
        trans /= scaling
    

    # # to quaternion
    rots = axis_angle_to_quaternion(torch.from_numpy(rots)).numpy()
    order = "zyx"
    pos = offsets[None].repeat(len(rots), axis=0)
    positions = pos.copy()
    positions[:,0] += trans
    # put positions on floor
    rotations = np.degrees(quat.to_euler(rots, order=order))
    
    bvh_data ={
        "rotations": rotations,
        "positions": positions,
        "offsets": offsets,
        "parents": parents,
        "names": names,
        "order": order,
        "frametime": 1 / fps,
    }
    
    if not output.endswith(".bvh"):
        output = output + ".bvh"
    
    os.makedirs(os.path.dirname(output), exist_ok=True)
    bvh.save(output, bvh_data)
    
    if mirror:
        rots_mirror, trans_mirror = mirror_rot_trans(
                rots, trans, names, parents)
        positions_mirror = pos.copy()
        positions_mirror[:,0] += trans_mirror
        rotations_mirror = np.degrees(
            quat.to_euler(rots_mirror, order=order))
        
        bvh_data ={
            "rotations": rotations_mirror,
            "positions": positions_mirror,
            "offsets": offsets,
            "parents": parents,
            "names": names,
            "order": order,
            "frametime": 1 / fps,
        }
        
        output_mirror = output.split(".")[0] + "_mirror.bvh"
        bvh.save(output_mirror, bvh_data)

if __name__ == "__main__":
    args = parse_args()
    if args.is_folder:
        for file in tqdm(findAllFile(args.poses)):
            if file.endswith(".npy"):
                smpl2bvh(model_path=args.model_path, model_type=args.model_type, 
                        mirror = args.mirror, gender=args.gender,
                        poses=file, num_betas=args.num_betas, 
                        fps=args.fps, output=file.replace(args.poses, args.output).replace(".npy", ".bvh"))
    else:
        smpl2bvh(model_path=args.model_path, model_type=args.model_type, 
                mirror = args.mirror, gender=args.gender,
                poses=args.poses, num_betas=args.num_betas, 
                fps=args.fps, output=args.output)

    print(f"Processed BVH file is saved in {args.output}")
