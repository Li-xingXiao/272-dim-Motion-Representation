'''
This file is used to calculate the mean and std of the dataset. (272-dim)
'''


import numpy as np
import sys
import os
from os.path import join as pjoin
from tqdm import tqdm
import argparse

def findAllFile(base):
    file_path = []
    for root, ds, fs in os.walk(base, followlinks=True):
        for f in fs:
            fullname = os.path.join(root, f)
            file_path.append(fullname)
    return file_path


def mean_variance(data_dir, save_dir):
    file_list = findAllFile(data_dir)
    data_list = []

    for file in tqdm(file_list):
        data = np.load(file)
        if np.isnan(data).any():
            print(file)
            continue
        data_list.append(data)

    data = np.concatenate(data_list, axis=0)
    print(data.shape)
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    joints_num = 22
    Std[:2] = Std[:2].mean() / 1.0
    Std[2:8] = Std[2:8].mean() / 1.0
    Std[8:8+3*joints_num] = Std[8:8+3*joints_num].mean() / 1.0
    Std[8+3*joints_num:8+6*joints_num] = Std[8+3*joints_num:8+6*joints_num].mean() / 1.0
    Std[8+6*joints_num:8+12*joints_num] = Std[8+6*joints_num:8+12*joints_num].mean() / 1.0

    os.makedirs(save_dir, exist_ok=True)
    np.save(pjoin(save_dir, 'Mean.npy'), Mean)
    np.save(pjoin(save_dir, 'Std.npy'), Std)

    return Mean, Std

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./')
    parser.add_argument('--output_dir', type=str, default='./')
    args = parser.parse_args()
    mean_variance(
        args.input_dir,
        args.output_dir
    )


