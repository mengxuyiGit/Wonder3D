import numpy as np
import torch

def fov_to_ixt(fov, reso):
    ixt = np.eye(3, dtype=np.float32)
    ixt[0][2], ixt[1][2] = reso[0]/2, reso[1]/2
    focal = .5 * reso / np.tan(.5 * fov)
    ixt[[0,1],[0,1]] = focal
    return ixt

def get_proj_matrix(fovy, z_near=0.5, z_far=2.5):
    # self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
    # self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
    # self.proj_matrix[0, 0] = 1 / self.tan_half_fov
    # self.proj_matrix[1, 1] = 1 / self.tan_half_fov
    # self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
    # self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
    # self.proj_matrix[2, 3] = 1
    
    # replace the above self. with none
    tan_half_fov = np.tan(0.5 * fovy) # fovy = 0.69115037 already in radian
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (z_far + z_near) / (z_far - z_near)
    proj_matrix[3, 2] = - (z_far * z_near) / (z_far - z_near)
    proj_matrix[2, 3] = 1
    return proj_matrix
