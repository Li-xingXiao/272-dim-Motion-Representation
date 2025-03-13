'''
This file is using smpl layer to get the global joint positions from smpl rotations.
'''

from human_body_prior.body_model.body_model import BodyModel
import numpy as np
from tqdm import tqdm
import os
import torch
import argparse

def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process SMPL rotation to joints.')
    parser.add_argument('--filedir', type=str, required=True, help='Input directory containing SMPL rotation files.')
    args = parser.parse_args()

    smplx_model = BodyModel(bm_fname='./body_models/human_model_files/smplx/SMPLX_NEUTRAL.npz', num_betas=10, model_type='smplx')
    smplx_model.eval()
    for p in smplx_model.parameters():
        p.requires_grad = False
    smplx_model.cuda()
    for i in tqdm(findAllFile(os.path.join(args.filedir, 'smpl_85_face_z_transform'))):
        data = torch.from_numpy(np.load(i)).cuda()
        joints = smplx_model(pose_body=data[:,3:66], root_orient=data[:,:3], trans=data[:, 72:72+3], betas=data[:, 75:]).Jtr
        output_path = i.replace('smpl_85_face_z_transform', 'smpl_85_face_z_transform_joints')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, joints.detach().cpu().numpy())

    print(f"Processed files are saved in {args.filedir}/smpl_85_face_z_transform_joints")


        