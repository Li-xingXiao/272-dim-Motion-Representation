import os
import os.path as osp
import sys
import numpy as np

class Config:

    flame_shape_params = 100
    flame_expression_params = 50
    face_corr_fname = '/cpfs01/user/xiaolixing/T2M-GPT/body_models/human_model_files/smplx/SMPL-X__FLAME_vertex_ids.npy'

    def set_additional_args(self, **kwargs):
        names = self.__dict__
        for k, v in kwargs.items():
            names[k] = v


cfg = Config()