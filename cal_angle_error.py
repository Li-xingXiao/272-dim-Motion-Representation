'''This file is used to evaulate the angle error between recovered and ground truth SMPL rotations.'''

import numpy as np
from utils.face_z_align_util import expmap_to_quaternion, quaternion_to_matrix, quaternion_to_matrix_np, matrix_to_rotation_6d, qrot_np, rotation_6d_to_matrix, matrix_to_axis_angle
import copy
import torch
import scipy.ndimage as ndimage
from tqdm import tqdm
import os
import visualization.Animation as Animation

from scipy.spatial.transform import Rotation as R
import visualization.BVH_mod as BVH
from visualization import AnimationStructure
from visualization.Quaternions import Quaternions

from glob import glob
from collections import defaultdict  


import argparse

class BasicInverseKinematics:
    """
    Basic Inverse Kinematics Solver

    This is an extremely simple full body IK
    solver.

    It works given the following conditions:

        * All joint targets must be specified
        * All joint targets must be in reach
        * All joint targets must not differ
          extremely from the starting pose
        * No bone length constraints can be violated
        * The root translation and rotation are
          set to good initial values

    It works under the observation that if the
    _directions_ the joints are pointing toward
    match the _directions_ of the vectors between
    the target joints then the pose should match
    that of the target pose.

    Therefore it iterates over joints rotating
    each joint such that the vectors between it
    and it's children match that of the target
    positions.

    Parameters
    ----------

    animation : Animation
        animation input

    positions : (F, J, 3) ndarray
        target positions for each frame F
        and each joint J

    iterations : int
        Optional number of iterations.
        If the above conditions are met
        1 iteration should be enough,
        therefore the default is 1

    silent : bool
        Optional if to suppress output
        defaults to False
    """

    def __init__(self, animation, positions, iterations=1, silent=True):

        self.animation = animation
        self.positions = positions
        self.iterations = iterations
        self.silent = silent

    def __call__(self):

        children = AnimationStructure.children_list(self.animation.parents)

        for i in range(self.iterations):

            for j in AnimationStructure.joints(self.animation.parents):

                c = np.array(children[j])
                if len(c) == 0: continue

                anim_transforms = Animation.transforms_global(self.animation)
                anim_positions = anim_transforms[:, :, :3, 3]
                anim_rotations = Quaternions.from_transforms(anim_transforms)

                jdirs = anim_positions[:, c] - anim_positions[:, np.newaxis, j]
                ddirs = self.positions[:, c] - anim_positions[:, np.newaxis, j]

                jsums = np.sqrt(np.sum(jdirs ** 2.0, axis=-1)) + 1e-10
                dsums = np.sqrt(np.sum(ddirs ** 2.0, axis=-1)) + 1e-10

                jdirs = jdirs / jsums[:, :, np.newaxis]
                ddirs = ddirs / dsums[:, :, np.newaxis]

                angles = np.arccos(np.sum(jdirs * ddirs, axis=2).clip(-1, 1))
                axises = np.cross(jdirs, ddirs)
                axises = -anim_rotations[:, j, np.newaxis] * axises

                rotations = Quaternions.from_angle_axis(angles, axises)

                if rotations.shape[1] == 1:
                    averages = rotations[:, 0]
                else:
                    averages = Quaternions.exp(rotations.log().mean(axis=-2))

                self.animation.rotations[:, j] = self.animation.rotations[:, j] * averages

            if not self.silent:
                anim_positions = Animation.positions_global(self.animation)
                error = np.mean(np.sum((anim_positions - self.positions) ** 2.0, axis=-1) ** 0.5)
                print('[BasicInverseKinematics] Iteration %i Error: %f' % (i + 1, error))

        return self.animation

def lerp(a, l, r):
    return (1 - a) * l + a * r

def alpha(t):
    return 2.0 * t * t * t - 3.0 * t * t + 1

def remove_fs(glb, foot_contact, fid_l=(3, 4), fid_r=(7, 8), interp_length=5, force_on_floor=True):
    
    scale = 1. 
    
    height_thres = [0.06, 0.03] #[ankle, toe] meter
    if foot_contact is None:
        def foot_detect(positions, velfactor, heightfactor):
            feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
            feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
            feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
            feet_l_h = positions[:-1, fid_l, 1]
            feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(float)

            feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
            feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
            feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
            feet_r_h = positions[:-1, fid_r, 1]

            feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(float)

            return feet_l, feet_r

        
        feet_vel_thre = np.array([0.05, 0.2])
        feet_h_thre = np.array(height_thres) * scale
        feet_l, feet_r = foot_detect(glb, velfactor=feet_vel_thre, heightfactor=feet_h_thre)
        foot = np.concatenate([feet_l, feet_r], axis=-1).transpose(1, 0)  # [4, T-1]
        foot = np.concatenate([foot, foot[:, -1:]], axis=-1)
    else:
        foot = foot_contact.transpose(1, 0)

    T = len(glb)

    fid = list(fid_l) + list(fid_r)
    fid_l, fid_r = np.array(fid_l), np.array(fid_r)
    foot_heights = np.minimum(glb[:, fid_l, 1],
                              glb[:, fid_r, 1]).min(axis=1)  # [T, 2] -> [T]
    
    sort_height = np.sort(foot_heights)
    temp_len = len(sort_height)
    floor_height = np.mean(sort_height[int(0.25*temp_len):int(0.5*temp_len)])
    if floor_height > 0.5: # for motion like swim
        floor_height = 0

    glb[:, :, 1] -= floor_height
    for i, fidx in enumerate(fid):
        fixed = foot[i]  # [T]

        """
        for t in range(T):
            glb[t, fidx][1] = max(glb[t, fidx][1], 0.25)
        """

        s = 0
        while s < T:
            while s < T and fixed[s] == 0:
                s += 1
            if s >= T:
                break
            t = s
            avg = glb[t, fidx].copy()
            while t + 1 < T and fixed[t + 1] == 1:
                t += 1
                avg += glb[t, fidx].copy()
            avg /= (t - s + 1)

            if force_on_floor:
                avg[1] = 0.0

            for j in range(s, t + 1):
                glb[j, fidx] = avg.copy()

            s = t + 1

        for s in range(T):
            if fixed[s] == 1:
                continue
            l, r = None, None
            consl, consr = False, False
            for k in range(interp_length):
                if s - k - 1 < 0:
                    break
                if fixed[s - k - 1]:
                    l = s - k - 1
                    consl = True
                    break
            for k in range(interp_length):
                if s + k + 1 >= T:
                    break
                if fixed[s + k + 1]:
                    r = s + k + 1
                    consr = True
                    break

            if not consl and not consr:
                continue
            if consl and consr:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[l, fidx])
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[r, fidx])
                itp = lerp(alpha(1.0 * (s - l + 1) / (r - l + 1)),
                           ritp, litp)
                glb[s, fidx] = itp.copy()
                continue
            if consl:
                litp = lerp(alpha(1.0 * (s - l + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[l, fidx])
                glb[s, fidx] = litp.copy()
                continue
            if consr:
                ritp = lerp(alpha(1.0 * (r - s + 1) / (interp_length + 1)),
                            glb[s, fidx], glb[r, fidx])
                glb[s, fidx] = ritp.copy()

    targetmap = {}
    for j in range(glb.shape[1]):
        targetmap[j] = glb[:, j]

    
    return glb

class JointPositionToRotation:
    def __init__(self):
        """Initialize the converter"""
        self.template = BVH.load('./visualization/template.bvh', need_quater=True)
        self.re_order = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]
        self.re_order_inv = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12, 14, 18, 13, 15, 19, 16, 20, 17, 21]
        self.template_offset = self.template.offsets.copy()
        self.parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]
    
    def positions_to_rotations(self, positions, iterations=10, foot_ik=True, return_format='axis_angle'):
        """
        Convert joint positions to joint rotations
        
        Parameters:
        positions: np.ndarray, joint positions with shape (frames, 22, 3)
        iterations: int, number of iterations for IK optimization
        foot_ik: bool, whether to apply foot IK to eliminate sliding
        return_format: str, return format, options: 'quaternion', 'axis_angle', 'rotation_matrix'
        
        Returns:
        rotations: np.ndarray, joint rotations in the selected format
        """
        # 1. Apply reordering
        positions_reordered = positions[:, self.re_order].copy()
        
        # 2. Initialize animation
        new_anim = self.template.copy()
        new_anim.rotations = Quaternions.id(positions_reordered.shape[:-1])
        new_anim.positions = new_anim.positions[0:1].repeat(positions_reordered.shape[0], axis=0)
        new_anim.positions[:, 0] = positions_reordered[:, 0]  # Set root position
        
        # 3. Apply foot IK (if needed)
        if foot_ik:
            positions_reordered = remove_fs(positions_reordered, None, 
                                           fid_l=(3, 4), fid_r=(7, 8), 
                                           interp_length=5, force_on_floor=True)
        
        # 4. Apply IK solver
        # print("Performing IK calculation to get joint rotations...")
        ik_solver = BasicInverseKinematics(new_anim, positions_reordered, 
                                          iterations=iterations, silent=True)
        result_anim = ik_solver()
        
        # 5. Get rotations in quaternion format
        quaternions = result_anim.rotations.qs  # Shape: (frames, 22, 4)
        
        # 6. Reorder back to original joint order
        quaternions_reordered = quaternions[:, self.re_order_inv]
        
        # 7. Convert rotation representation according to desired format
        if return_format == 'quaternion':
            return quaternions_reordered
        
        # Use scipy for rotation format conversion
        rot = R.from_quat(quaternions_reordered.reshape(-1, 4))
        
        if return_format == 'axis_angle':
            # Convert to axis-angle representation
            rotvec = rot.as_rotvec()
            return rotvec.reshape(quaternions_reordered.shape[0], 22, 3)
        
        elif return_format == 'rotation_matrix':
            # Convert to rotation matrix
            rotmat = rot.as_matrix()
            return rotmat.reshape(quaternions_reordered.shape[0], 22, 3, 3)
        
        else:
            raise ValueError(f"Unsupported rotation format: {return_format}")



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
    # type: (Tensor) -> Tensor
    # calculate heading direction from quaternion
    # the heading is the direction on the xz plane
    # q must be normalized
    # this is the z axis heading
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 2] = 1
    rot_dir = my_quat_rotate(q, ref_dir)

    heading = torch.atan2(rot_dir[..., 0], rot_dir[..., 2])
    return heading


def calc_heading_quat_inv(q):
    # type: (Tensor) -> Tensor
    # calculate heading rotation from quaternion
    # the heading is the direction on the xz plane
    # q must be normalized
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 1] = 1

    return -heading, axis

def accumulate_rotations(relative_rotations):

    R_total = [relative_rotations[0]]
    for R_rel in relative_rotations[1:]:
        R_total.append(np.matmul(R_rel, R_total[-1]))
    
    return np.array(R_total)

def recover_from_local_position(final_x, njoint):
    # take positions_no_heading: local position on xz ori, no heading
    # velocities_root_xy_no_heading: to recover translation
    # global_heading_diff_rot: to recover root rotation
    nfrm, _ = final_x.shape
    positions_no_heading = final_x[:,8:8+3*njoint].reshape(nfrm, -1, 3) # frames, njoints * 3
    velocities_root_xy_no_heading = final_x[:,:2] # frames, 2
    global_heading_diff_rot = final_x[:,2:8] # frames, 6

    # recover global heading
    global_heading_rot = accumulate_rotations(rotation_6d_to_matrix(torch.from_numpy(global_heading_diff_rot)).numpy())
    inv_global_heading_rot = np.transpose(global_heading_rot, (0, 2, 1))
    # add global heading to position
    positions_with_heading = np.matmul(np.repeat(inv_global_heading_rot[:, None,:, :], njoint, axis=1), positions_no_heading[...,None]).squeeze(-1)

    # recover root translation
    # add heading to velocities_root_xy_no_heading

    velocities_root_xyz_no_heading = np.zeros((velocities_root_xy_no_heading.shape[0], 3))
    velocities_root_xyz_no_heading[:, 0] = velocities_root_xy_no_heading[:, 0]
    velocities_root_xyz_no_heading[:, 2] = velocities_root_xy_no_heading[:, 1]
    velocities_root_xyz_no_heading[1:, :] = np.matmul(inv_global_heading_rot[:-1], velocities_root_xyz_no_heading[1:, :,None]).squeeze(-1)

    root_translation = np.cumsum(velocities_root_xyz_no_heading, axis=0)


    # add root translation
    positions_with_heading[:, :, 0] += root_translation[:, 0:1]
    positions_with_heading[:, :, 2] += root_translation[:, 2:]

    return positions_with_heading


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

def rotations_matrix_to_smplx85(rotations_matrix, translation):
    nfrm, njoint, _, _ = rotations_matrix.shape
    axis_angle = matrix_to_axis_angle(torch.from_numpy(rotations_matrix)).numpy().reshape(nfrm, -1)
    smplx_85 = np.concatenate([axis_angle, np.zeros((nfrm, 6)), translation, np.zeros((nfrm, 10))], axis=-1)
    return smplx_85

def foot_detect(global_positions, thres):
    # derived from https://github.com/orangeduck/Motion-Matching/blob/37df18afc44e8acca3af5e85dff96effa6a34b03/resources/generate_database.py#L160
    
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


def smplx85_2_smplx322(smplx_no_shape_data):
    result = np.concatenate((smplx_no_shape_data[:,:66], np.zeros((smplx_no_shape_data.shape[0], 90)), np.zeros((smplx_no_shape_data.shape[0], 3)), np.zeros((smplx_no_shape_data.shape[0], 50)), np.zeros((smplx_no_shape_data.shape[0], 100)), smplx_no_shape_data[:,72:72+3], smplx_no_shape_data[:,75:]), axis=-1)
    # import pdb; pdb.set_trace()
    return result


def compute_joint_rotation_error(pred_rotations, gt_rotations):
    """
    Compute rotation error between predicted and ground truth rotations
    
    Parameters:
    pred_rotations: np.ndarray, shape [frames, joints, 3] in axis-angle format
    gt_rotations: np.ndarray, shape [frames, joints, 3] in axis-angle format
    
    Returns:
    results: dict, containing error metrics
    """
    # Convert to rotation matrices for comparison
    num_frames = pred_rotations.shape[0]
    num_joints = pred_rotations.shape[1]
    
    # Initialize storage for errors
    errors_per_joint = np.zeros((num_frames, num_joints))
    
    # Calculate rotation error for each joint
    for frame in range(num_frames):
        for joint in range(num_joints):
            # Convert axis-angle to rotation object
            pred_rot = R.from_rotvec(pred_rotations[frame, joint])
            gt_rot = R.from_rotvec(gt_rotations[frame, joint])
            
            # Compute relative rotation
            rel_rot = pred_rot * gt_rot.inv()
            
            # Extract angle error (geodesic distance) (minimum angle between rotations) 
            angle_error = rel_rot.magnitude() * 180 / np.pi  # Convert to degrees
            errors_per_joint[frame, joint] = angle_error
    
    # Calculate statistics
    mean_error_per_joint = np.mean(errors_per_joint, axis=0)  # Average error per joint
    joint_error_dict = {i: mean_error_per_joint[i] for i in range(num_joints)}
    overall_mean_error = np.mean(errors_per_joint)  # Overall average error
    overall_max_error = np.max(errors_per_joint)    # Maximum error
    
    results = {
        'overall_mean_error': overall_mean_error,
        'overall_max_error': overall_max_error,
        'mean_error_per_joint': joint_error_dict
    }
    
    return results


def evaluate_rotation_folders(pred_folder, gt_folder, ik=False):
    """
    Evaluate rotation errors between corresponding files in prediction and ground truth folders
    
    Parameters:
    pred_folder: str, path to folder containing prediction rotation files
    gt_folder: str, path to folder containing ground truth rotation files
    ik: bool, whether to use IK to recover joint rotations from joint positions
    
    Returns:
    avg_results: dict, containing average error metrics across all files
    """
    if ik:
        converter = JointPositionToRotation()
    
    pred_files = sorted(glob(os.path.join(pred_folder, "*.npy")))
    if not pred_files:
        print(f"No .npy files found in {pred_folder}")
        return None

    
    # Initialize accumulators for metrics
    all_results = []
    joint_errors_sum = None
    total_files = 0
    
    print(f"Found {len(pred_files)} files in prediction folder. Processing...")
    
    # Process each file pair
    for pred_file in tqdm(pred_files):
        file_name = os.path.basename(pred_file)
        gt_file = os.path.join(gt_folder, file_name)
        
        # Check if corresponding ground truth file exists
        if not os.path.exists(gt_file):
            print(f"Warning: No matching ground truth file for {file_name}")
            continue
        
       
        # Load prediction and ground truth data
        pred_data = np.load(pred_file)    # pred: dim=272

        if ik:
            pred_xyz = recover_from_local_position(pred_data, 22)

            # Inverse kinematics (IK) to recover joint rotations from joint positions
            pred_rotations = converter.positions_to_rotations(
                pred_xyz, 
                iterations=10,
                foot_ik=True,
                return_format='axis_angle'  # or 'quaternion' or 'rotation_matrix'
            )

        else:
            # direct recover from joint rotations
            pred_data = recover_from_local_rotation(pred_data, 22)    # pred: dim=85
            pred_rotations = pred_data[:, :66].reshape(-1, 22, 3)


        gt_data = np.load(gt_file)    # gt: dim=85 all face z+
        gt_rotations = gt_data[:, :66].reshape(-1, 22, 3)
            
        # Verify shapes
        if pred_rotations.shape != gt_rotations.shape:
            print(f"Warning: Shape mismatch for {file_name}. Pred: {pred_rotations.shape}, GT: {gt_rotations.shape}")
            continue
                
            
        # Compute errors
        results = compute_joint_rotation_error(pred_rotations, gt_rotations)
        all_results.append(results)
            
        # Accumulate joint errors for averaging
        if joint_errors_sum is None:
            joint_errors_sum = {joint_id: error for joint_id, error in results['mean_error_per_joint'].items()}
        else:
            for joint_id, error in results['mean_error_per_joint'].items():
                joint_errors_sum[joint_id] += error
            
        total_files += 1
            
        
    
    # Calculate average metrics across all files
    if total_files > 0:
        avg_overall_mean = sum(r['overall_mean_error'] for r in all_results) / total_files
        avg_overall_max = sum(r['overall_max_error'] for r in all_results) / total_files
        avg_joint_errors = {joint_id: error / total_files for joint_id, error in joint_errors_sum.items()}
        
        avg_results = {
            'overall_mean_error': avg_overall_mean,
            'overall_max_error': avg_overall_max,
            'mean_error_per_joint': avg_joint_errors,
            'num_files_processed': total_files
        }
        
        # Print average results
        print("\nAverage results across all files:")
        print(f"  Mean Error: {avg_overall_mean} degrees")
        print(f"  Max Error: {avg_overall_max} degrees")
        
        # Print per-joint errors
        print("\nAverage joint errors:")
        for joint_id in sorted(avg_joint_errors.keys()):
            print(f"  Joint {joint_id}: {avg_joint_errors[joint_id]} degrees")
    
        return avg_results
    else:
        print("No files were successfully processed.")
        return None

if __name__ == "__main__":
    # Folder paths
    pred_folder = "/cpfs01/user/xiaolixing/T2M-GPT/smpl2rotationrepresentation-main/humanml_272"
    gt_folder = "/cpfs01/user/xiaolixing/T2M-GPT/smpl2rotationrepresentation-main/humanml"

    parser = argparse.ArgumentParser(description='Evaluate rotation errors')
    # rot: direct recover from joint rotations
    # pos: Use IK to recover joint rotations from joint positions
    parser.add_argument('--mode', type=str, default='rot', choices=['rot', 'pos'], help='Evaluation mode')  
    args = parser.parse_args()

    
    # Evaluate all files in the folders
    if args.mode == 'rot':
        results = evaluate_rotation_folders(pred_folder, gt_folder, ik=False)  # direct recover from joint rotations, no need IK
    elif args.mode == 'pos':
        results = evaluate_rotation_folders(pred_folder, gt_folder, ik=True)  # Use IK operation to recover joint rotations from joint positions


