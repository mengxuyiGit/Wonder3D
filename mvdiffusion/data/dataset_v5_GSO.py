import os
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import kiui
from kiui.cam import orbit_camera
# from core.options import Options
# from core.utils import get_rays, grid_distortion, orbit_camera_jitter


import glob
import einops
import math

from ipdb import set_trace as st
from utils.camera_utils import fov_to_ixt, get_proj_matrix

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


# exactly the same as self.load_ply() in the the gs.py 
def save_ply(path):
    from plyfile import PlyData, PlyElement

    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    # print("Number of points at loading : ", xyz.shape[0])

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    shs = np.zeros((xyz.shape[0], 3))
    shs[:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    shs[:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
    shs[:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot_")]
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
    gaussians = np.concatenate([xyz, opacities, scales, rots, shs], axis=1)
    gaussians = torch.from_numpy(gaussians).float() # cpu

    if compatible:
        gaussians[..., 3:4] = torch.sigmoid(gaussians[..., 3:4])
        gaussians[..., 4:7] = torch.exp(gaussians[..., 4:7])
        gaussians[..., 11:] = 0.28209479177387814 * gaussians[..., 11:] + 0.5

    return gaussians

import re
def extract_first_number(folder_name):
    match = re.search(r'\d+', folder_name)
    return int(match.group()) if match else None

def return_final_scene(scene_workspace, acceptable_epoch, verbose=False):
  
    for item in os.listdir(scene_workspace):
        if item.endswith("encoder_input"):
            continue
        
        item_epoch = extract_first_number(item)
        # if item_epoch is None or item.startswith('events'):
        if item_epoch is None or not os.path.isdir(os.path.join(scene_workspace, item)):
            continue

        # print(f"extract first number from item {item}: ",extract_first_number(item))
        # print(item)
        # print(item_epoch)
        if item.endswith('_success'):
            if verbose:
                print(f"Already early stopped.")
            return item
            
        elif item_epoch>=acceptable_epoch:# already achieved the max training epochs
            if verbose:
                print(f"Already achieved the acceptable training epochs.")
            # check content
            # return the correct folder
            return item
    return None

    # ---------
from torchvision import transforms
from PIL import Image
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_quaternion

# process the loaded splatters into 3-channel images
gt_attr_keys = ['pos', 'opacity', 'scale', 'rotation', 'rgbs']
start_indices = [0, 3, 4, 7, 11]
end_indices = [3, 4, 7, 11, 14]
attr_map = {key: (si, ei) for key, si, ei in zip (gt_attr_keys, start_indices, end_indices)}
ordered_attr_list = ["pos", # 0-3
                'opacity', # 3-4
                'scale', # 4-7
                "rotation", # 7-11
                "rgbs", # 11-14
            ] # must be an ordered list according to the channels

sp_min_max_dict = {
    "pos": (-0.7, 0.7), 
    "scale": (-10., -2.),
    "rotation": (-3., 3.)
    }

def load_splatter_mv_ply_as_dict(splatter_dir, device="cpu"):
    
    splatter_mv = torch.load(os.path.join(splatter_dir, "splatters_mv.pt")).detach().cpu() # [14, 384, 256]
    # splatter_mv = torch.load("splatters_mv_02.pt")[0]
    # print("Loading splatters_mv:", splatter_mv.shape) # [1, 14, 384, 256]

    splatter_3Channel_image = {}
            
    for attr_to_encode in ordered_attr_list:
        # print("latents_all_attr_list <-",attr_to_encode)
        si, ei = attr_map[attr_to_encode]
        
        sp_image = splatter_mv[si:ei]
        # print(f"{attr_to_encode}: {sp_image.min(), sp_image.max()}")

        #  map to 0,1
        if attr_to_encode in ["pos"]:
            sp_min, sp_max = sp_min_max_dict[attr_to_encode]
            sp_image = (sp_image - sp_min)/(sp_max - sp_min)
        elif attr_to_encode == "opacity":
            sp_image = sp_image.repeat(3,1,1)
        elif attr_to_encode == "scale":
            sp_image = torch.log(sp_image)
            sp_min, sp_max = sp_min_max_dict[attr_to_encode]
            sp_image = (sp_image - sp_min)/(sp_max - sp_min)
            sp_image = sp_image.clip(0,1)
        elif  attr_to_encode == "rotation":
            # print("processing rotation: ", si, ei)
            assert (ei - si) == 4
            quat = einops.rearrange(sp_image, 'c h w -> h w c')
            axis_angle = quaternion_to_axis_angle(quat)
            sp_image = einops.rearrange(axis_angle, 'h w c -> c h w')
            # print(f"{attr_to_encode}: {sp_image.min(), sp_image.max()}")
            # sp_min, sp_max = -3, 3
            sp_min, sp_max = sp_min_max_dict[attr_to_encode]
            sp_image = (sp_image - sp_min)/(sp_max - sp_min)
        elif attr_to_encode == "rgbs":
            pass
        
        # map to [-1,1]
        sp_image = sp_image * 2 - 1
        
        # print(f"{attr_to_encode}: {sp_image.min(), sp_image.max(), sp_image.shape}")
        assert sp_image.shape[0] == 3
        splatter_3Channel_image[attr_to_encode] = sp_image.detach().cpu()
    
    return splatter_3Channel_image

from typing import Literal, Tuple, Optional, Any, Dict
class GSODataset(Dataset):

    # def __init__(self, opt: Options, training=True, prepare_white_bg=False):
    def __init__(self,
        root_dir: str,
        dataset_type: str,
        num_views: int,
        bg_color: Any,
        img_wh: Tuple[int, int],
        object_list: str,
        groups_num: int=1,
        validation: bool = False,
        data_view_num: int = 6,
        num_validation_samples: int = 64,
        num_samples: Optional[int] = None,
        invalid_list: Optional[str] = None,
        trans_norm_system: bool = True,   # if True, transform all normals map into the cam system of front view
        augment_data: bool = False,
        read_normal: bool = True,
        read_color: bool = False,
        read_depth: bool = False,
        read_mask: bool = True,
        mix_color_normal: bool = False,
        suffix: str = 'png',
        subscene_tag: int = 3,
        backup_scene: str = "9438abf986c7453a9f4df7c34aa2e65b",
        overfit: bool = False,
        debug: bool = False,
        read_first_view_only: bool = False,
        lmdb_6view_base: str = None,
        rendering_loss_2dgs: bool = False,
        render_views: int = 10,
        render_size: int = 256,
        ):
        
        # self.opt = opt
        # if self.opt.model_type == 'LGM':
        #     self.opt.bg = 1.0
        # self.training = training
        # self.prepare_white_bg = prepare_white_bg
        self.training = not validation
        self.num_views = num_views

        self.data_path_rendering = {}
        for scene_path in sorted(glob.glob(root_dir + "/*")):
            scene_name = scene_path.split("/")[-1]
            self.data_path_rendering[scene_name] = scene_path  
        
        all_items = [k for k in self.data_path_rendering.keys()]
        
        self.overfit = overfit
        self.dataset_type = dataset_type
      
        num_val = min(50, len(all_items)//2) # when using small dataset to debug
        if self.training:
            self.items = all_items # NOTE: all scenes are used for training and val
            if self.overfit:
                # print(f"[WARN]: always fetch the 0th item. For debug use only")
                # self.items = all_items[:1]
                print(f"[WARN]: always fetch the 1th item. For debug use only")
                self.items = all_items[2:3]
        else:
            self.items = all_items
            # if self.overfit:
            if overfit:
                # print(f"[WARN]: always fetch the 0th item. For debug use only")
                # self.items = all_items[:1]
                print(f"[WARN]: always fetch the 1th item. For debug use only")
                self.items = all_items[2:3]
        
        # naive split
        # if self.training:
        #     self.items = self.items[:-self.opt.batch_size]
        # else:
        #     self.items = self.items[-self.opt.batch_size:]
        # self.items = self.items[:16]
        print(f"There are total {len(self.items)} in dataloader")
        
        # # default camera intrinsics
        # self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        # self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        # self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        # self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        # self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        # self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        # self.proj_matrix[2, 3] = 1

        self.img_wh = img_wh
        self.render_size = np.array([render_size]*2)
        self.cam_radius = 1.5
        
    def worker_init_open_db(self):
        # do nothing
        return 

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        
        scene_name = self.items[idx]
        if self.overfit:
            print(f"[WARN]: always fetch the {idx} item. For debug use only")
        
        uid = self.data_path_rendering[scene_name]
        # splatter_uid = self.data_path_vae_splatter[scene_name] 
        # if self.opt.verbose:
        #     print(f"uid:{uid}\nsplatter_uid:{splatter_uid}")
        
        results = {}

        # load num_views images
        images = []
        images_white = []
        masks = []
        cam_poses = []
        
        vid_cnt = 0

        # TODO: choose views, based on your rendering settings
        # if self.training:
        #     # input views are in (36, 72), other views are randomly selected
        #     vids = np.random.permutation(np.arange(36, 73))[:self.opt.num_input_views].tolist() + np.random.permutation(100).tolist()
        # else:
        #     # fixed views
        #     vids = np.arange(36, 73, 4).tolist() + np.arange(100).tolist()
      
        vids = np.arange(0, 16).tolist()

        # cond_path = os.path.join(uid, f'000.png')
        # # cond = to_rgb_image(Image.open("/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/data_test/anya_rgba.png"))
        # # cond_path = "/mnt/kostas-graid/sw/envs/xuyimeng/Repo/LGM/data_test/anya_rgba.png"
        # from PIL import Image
        # cond = np.array(Image.open(cond_path).resize((self.opt.input_size, self.opt.input_size)))
        # # print(f"cond size:{Image.open(cond_path)}")
        # mask = cond[..., 3:4] / 255
        # cond = cond[..., :3] * mask + (1 - mask) * int(self.opt.bg * 255)
        # results['cond'] = cond.astype(np.uint8)

        # splatter_original_Channel_mvimage_dict = load_splatter_mv_ply_as_dict(splatter_uid)
        # if self.opt.train_unet_single_attr is not None:
        #     for attr in self.opt.train_unet_single_attr:
        #         results[attr] = splatter_original_Channel_mvimage_dict[attr]
        #         # print(f"[dtaloader] result update {attr}")
        # else:
        #     results.update(splatter_original_Channel_mvimage_dict)
        # # for attr_to_encode in ordered_attr_list:
        # #     sp_image = results[attr_to_encode]
        # #     print(f"[just updated dataloader]{attr_to_encode}: {sp_image.min(), sp_image.max()}")
        
        print("vids:", vids)
        elevations, azimuths = [], []
        for vid in vids:

            image_path = os.path.join(uid, f'{vid:03d}.png')
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255 # [512, 512, 4] in [0, 1]
            # print("images shape: ", image.shape) # 320x320x4 for LVIS 46K too
            image = torch.from_numpy(image)

            camera_path = os.path.join(uid, f'{vid:03d}_pose.npy')
            if os.path.exists(camera_path):
                cam = np.load(camera_path, allow_pickle=True).item()
                # print(f"{vid} - elevation: {cam['elevation']}, azimuth: {cam['azimuth']}")
                c2w = orbit_camera(-cam['elevation'], cam['azimuth'], radius=cam['radius'])
                elevation = cam['elevation']
                azimuth = cam['azimuth']
            else:
                # print("camera path not found:", camera_path)
                # exit()
                # st()
                elevation = 0.
                azimuth = (vid / 16) * 360.
                c2w = orbit_camera(-elevation, azimuth, radius=1.5) 
            
            if vid == 0:
                elevations.append(elevation)
                azimuths.append(azimuth)
        
            c2w = torch.from_numpy(c2w)
          
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]
            
            # if self.prepare_white_bg:
            #     image_white = image[:3] * mask + (1 - mask) * 1.0
            #     image_white = image_white[[2,1,0]].contiguous() # bgr to rgb
            #     images_white.append(image_white)
            
            image = image[:3] * mask + (1 - mask) * 1.0 # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb
            images.append(image)
            
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            vid_cnt += 1
            if self.training and (vid_cnt == self.opt.num_views):
                break

        # if vid_cnt < self.opt.num_views:
        #     print(f"vid_cnt{vid_cnt} < self.opt.num_views{self.opt.num_views}")
        #     print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
        #     n = self.opt.num_views - vid_cnt
        #     images = images + [images[-1]] * n
        #     if self.prepare_white_bg:
        #         images_white = images_white + [images_white[-1]] * n
        #     masks = masks + [masks[-1]] * n
        #     cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, C, H, W]
        # if self.prepare_white_bg:
        #     images_white = torch.stack(images_white, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]
        
        # normalized camera feats as in paper (transform the first pose to a fixed position)
        radius = torch.norm(cam_poses[0, :3, 3])
        cam_poses[:, :3, 3] *= self.cam_radius / radius

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0]) # w2c_1
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4], c2c_1
        
        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        results['cam_poses'] = cam_poses # [V, 4, 4]

        #TOOD: check the camera poses, potential problems
        # 3. different fovy with training dataset                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
        
        
        normalize_to_elevation_30 = False
        if normalize_to_elevation_30:
            # normalized camera feats as in paper (transform the first pose to a fixed position)
            cam_poses_0 = torch.tensor(orbit_camera(-30, 30, radius=self.opt.cam_radius))
            transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses_0)
            cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]
            

        # images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        # masks_input = F.interpolate(masks[:self.opt.num_input_views].clone().unsqueeze(1), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        # if self.prepare_white_bg:
        #     images_input_white = F.interpolate(images_white[:self.opt.num_input_views].clone(), size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        # cam_poses_input = cam_poses[:self.opt.num_input_views].clone()

        # # data augmentation
        # if self.training:
        #     # apply random grid distortion to simulate 3D inconsistency
        #     if random.random() < self.opt.prob_grid_distortion:
        #         images_input[1:] = grid_distortion(images_input[1:])
        #     # apply camera jittering (only to input!)
        #     if random.random() < self.opt.prob_cam_jitter:
        #         cam_poses_input[1:] = orbit_camera_jitter(cam_poses_input[1:])

        # # FIXME: we don't need this for zero123plus?
        # if self.opt.model_type == 'LGM':
        #     images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        # render_input_views = True #self.opt.render_input_views
        
        results['imgs_in'] =  F.interpolate(images[0:1], size=(self.img_wh[0], self.img_wh[1]), mode='bilinear', align_corners=False).repeat(self.num_views, 1, 1, 1) # [1, C, output_size, output_size]
        # results['imgs_out'] = F.interpolate(images, size=(self.img_wh[0], self.img_wh[1]), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        # results['masks'] = F.interpolate(masks.unsqueeze(1), size=(self.render_size[0], self.render_size[1]), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]
        # results['normals_out'] = F.interpolate(normal_final, size=(self.render_size[0], self.render_size[1]), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
            
        
        # if self.prepare_white_bg:
        #     results['images_output_white'] = F.interpolate(images_white, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        # if self.opt.verbose:
        #     print(f"images_input:{images_input.shape}") # [20, 3, input_size, input_size] input_size=128
        #     print("images_output", results['images_output'].shape) # [20, 3, 512, 512]
        
        # results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]
        # if not render_input_views:
        #     results['images_output'] = results['images_output'][self.opt.num_input_views:]
        #     results['masks_output'] = results['masks_output'][self.opt.num_input_views:]

        # # build rays for input views
        # if self.opt.model_type == 'LGM':
        #     rays_embeddings = []
        #     for i in range(self.opt.num_input_views):
        #         rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
        #         rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
        #         rays_embeddings.append(rays_plucker)

        #     rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        #     final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        #     results['input'] = final_input
        # else:
        # results['input'] = images_input
        # results['masks_input'] = masks_input
        
        # lgm_images_input = F.interpolate(images[:self.opt.num_input_views].clone(), size=(256, 256), mode='bilinear', align_corners=False) # [V, C, H, W]
        # lgm_images_input = TF.normalize(lgm_images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # ## for adding additonal input for lgm
        # rays_embeddings = []
        # for i in range(self.opt.num_input_views):
        #     rays_o, rays_d = get_rays(cam_poses_input[i], 256, 256, self.opt.fovy) # [h, w, 3]
        #     rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
        #     rays_embeddings.append(rays_plucker)

        # rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        # final_input = torch.cat([lgm_images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        # results['input_lgm'] = final_input


        # # should use the same world coord as gs renderer to convert depth into world_xyz
        # results['c2w_colmap'] = cam_poses[:self.opt.num_input_views].clone() 

        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        # cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
    
        results['fovy'] = 0.69115037 # TODO
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ get_proj_matrix(results['fovy']) # [V, 4, 4]
        
        # cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        # if render_input_views:
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        # results['cam_pos'] = cam_pos

        # else:
        #     st()
        #     results['cam_view'] = cam_view[self.opt.num_input_views:]
        #     results['cam_view_proj'] = cam_view_proj[self.opt.num_input_views:]
        #     # results['cam_pos'] = cam_pos[self.opt.num_input_views:]

        results['scene_name'] = scene_name
    
        # for wonder3d
        #1. camera_embbedding
        elevations_cond = torch.tensor([elevations[0]]*6).float()
        
        # camera_embeddings = torch.stack([elevations_cond, elevations-elevations_cond, azimuths-azimuths_cond], dim=-1) # (Nv, 3)
        camera_embeddings = torch.stack([elevations_cond, torch.zeros_like(elevations_cond), torch.tensor([0.,  90., 180., 270.,  30., 330.])], dim=-1) # (Nv, 3)
        results['camera_embeddings'] = camera_embeddings
        print("camera_embeddings:", camera_embeddings.shape)
        
        # splatter task embeddings
        splatter_class_all = torch.eye(5).float()
        for i, key in enumerate(gt_attr_keys):
            results[f"{key}_task_embeddings"] = torch.stack([splatter_class_all[i]]*self.num_views, dim=0)

            
        return results
