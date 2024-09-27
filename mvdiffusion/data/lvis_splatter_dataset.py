from typing import Dict
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from einops import rearrange
from typing import Literal, Tuple, Optional, Any
import cv2
import random

import json
import os, sys
import math

import PIL.Image
from .normal_utils import trans_normal, normal2img, img2normal
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import kiui
from kiui.cam import orbit_camera
import einops
import glob   
from ipdb import set_trace as st

from utils.splatter_utils import load_splatter_mv_ply_as_dict, gt_attr_keys
from utils.camera_utils import fov_to_ixt, get_proj_matrix

class ObjaverseDataset(Dataset):
    def __init__(self,
        root_dir: str,
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
        dataset_type: str = "lvis",
        data_path_vae_splatter: str = None,
        splatter_mode: str = "2dgs",
        ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.training = not validation
        self.num_views = num_views
        self.img_wh = img_wh
        self.num_cond = 1
        self.items = []
        # self.root_dir = '/mnt/lingjie_cache/lvis_dataset/testing'
        self.root_dir = '/home/chenwang/data/lvis_dataset/testing'
        # self.data_path_vae_splatter = '/mnt/lingjie_cache/lvis_splatters/testing'
        self.data_path_vae_splatter = "/mnt/kostas-graid/datasets/xuyimeng/lvis/splatter_data_2dgs/testing"
        self.use_2dgs = splatter_mode == "2dgs"
        print(f"Use 2dgs: {self.use_2dgs}")

        self.overfit = overfit
        self.mix_color_normal = mix_color_normal 
    
        self.cam_radius = 1.5
        self.dataset_type = dataset_type
        self.rendering_loss_2dgs = rendering_loss_2dgs   
        self.render_views = render_views
        self.read_first_view_only = read_first_view_only
        self.render_size = (render_size, render_size)
        
        # paths = json.load(open(os.path.join(self.root_dir, 'valid_paths.json'), 'r'))
        # self.items = [os.path.join(self.root_dir, path) for path in paths]

        excluded_splits = [] # used for test
        included_splits = [split for split in os.listdir(self.root_dir) if split not in excluded_splits]
        # scene_path_patterns = [os.path.join(self.root_dir, split, "*") for split in included_splits]
        scene_path_patterns = [os.path.join(self.data_path_vae_splatter, split, "*", "splatters_mv_inference", "*") for split in included_splits]
       
        all_scene_paths = []
        for pattern in scene_path_patterns:
            all_scene_paths.extend(sorted(glob.glob(pattern)))
        
        # remove invalid uids
        invalid_list = '/mnt/kostas_home/lilym/LGM/LGM/data_lists/lvis_invalid_uids_nineviews.json'
        if invalid_list is not None:
            print(f"Filter invalid objects by {invalid_list}")
            with open(invalid_list) as f:
                invalid_objects = json.load(f)
            invalid_objects = [os.path.basename(o).replace(".glb", "") for o in invalid_objects]
        else:
            invalid_objects = []
        
        # valid_list = '/mnt/lingjie_cache/lvis_dataset/testing/valid_paths.json'
        valid_list = f'{self.root_dir}/valid_paths.json'
        if valid_list is not None:
            print(f"ALSO Filter valid objects by {valid_list}")
            with open(valid_list) as f:
                valid_objects = json.load(f)
      
      
        self.data_path_rendering = {}
        self.data_path_vae_splatter = {}
        
        print("before filtering, number of scenes in  dataset:", len(all_scene_paths))
        
        for scene_path in all_scene_paths:
    
            if overfit and len(self.data_path_vae_splatter) > 3:
                break
   
            if not os.path.isdir(scene_path):
                continue
    
            scene_name = scene_path.split('/')[-1]
            scene_range = scene_path.split('/')[-4]
            # print("scene name:", scene_name)
            if scene_name.split("_")[-1] in invalid_objects:
                # rendering_folder = os.path.join(self.root_dir, scene_range, scene_name.split("_")[-1])
                # print(f"{rendering_folder} is invalid")
                # print(f"{scene_name} is invalid")
                continue 
            
            # if scene_name in self.data_path_vae_splatter.keys():
            #     continue
            
            if not os.path.exists(os.path.join(scene_path, "splatters_mv.pt")):
                continue
            
            
            rendering_path = os.path.join(scene_range, scene_name.split("_")[-1])
            if valid_list is not None and rendering_path not in valid_objects:
                # print(f"{rendering_path} is not in valid list")
                continue
            
            # pass all checks
            self.data_path_rendering[scene_name] = os.path.join(self.root_dir, scene_range, scene_name.split("_")[-1])
            self.data_path_vae_splatter[scene_name] = scene_path
               
        assert len(self.data_path_vae_splatter) == len(self.data_path_rendering)
        print("number of scenes in  dataset:", len(self.data_path_vae_splatter))
      
        self.items = [k for k in self.data_path_vae_splatter.keys()]
      
        if overfit:
            self.items = self.items[0]*30

        
        num_validation_samples = 10
        # naive split
        if self.training:
            self.items = self.items[:-num_validation_samples]
        else:
            self.items = self.items[-num_validation_samples:]
        
        print(f'[final] [{"train" if self.training else "validation"}] dataloader:', len(self.items)) 
        
    
    def worker_init_open_db(self):
        return 


    # def get_bg_color(self):
    #     if self.bg_color == 'white':
    #         bg_color = np.array([1., 1., 1.], dtype=np.float32)
    #     elif self.bg_color == 'black':
    #         bg_color = np.array([0., 0., 0.], dtype=np.float32)
    #     elif self.bg_color == 'gray':
    #         bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    #     elif self.bg_color == 'random':
    #         bg_color = np.random.rand(3)
    #     elif self.bg_color == 'three_choices':
    #         white = np.array([1., 1., 1.], dtype=np.float32)
    #         black = np.array([0., 0., 0.], dtype=np.float32)
    #         gray = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    #         bg_color = random.choice([white, black, gray])
    #     elif isinstance(self.bg_color, float):
    #         bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
    #     else:
    #         raise NotImplementedError
    #     return bg_color

    def __len__(self):
        return len(self.items)
    
    
    def __getitem_mix__(self, idx):
        
        results = {}

        # print("__mix___")
        uid = self.items[idx]
        renderings_path = self.data_path_rendering[uid]
        splatter_uid = self.data_path_vae_splatter[uid]
        
        # splatters
        selected_attr = random.choice(gt_attr_keys)
        selected_attr_idx = gt_attr_keys.index(selected_attr)
        splatter_class = torch.tensor([0, 0, 0, 0, 0]).float()
        splatter_class[selected_attr_idx] = 1
        task_embeddings = torch.stack([splatter_class]*self.num_views, dim=0)  # (Nv, 5)
        
        results['task_embeddings'] = task_embeddings
        # print("task_embeddings", task_embeddings)
        
        splatter_original_Channel_mvimage_dict = load_splatter_mv_ply_as_dict(splatter_uid, use_2dgs=self.use_2dgs, return_gassians=self.rendering_loss_2dgs, selected_attr_list=[selected_attr]) # [-1,1]

        if self.rendering_loss_2dgs:
            results['gaussians_gt'] = splatter_original_Channel_mvimage_dict['gaussians_gt']
            del splatter_original_Channel_mvimage_dict['gaussians_gt']
            if 'gaussians_recon' in splatter_original_Channel_mvimage_dict.keys():
                results['gaussians_recon'] = splatter_original_Channel_mvimage_dict['gaussians_recon']
                del splatter_original_Channel_mvimage_dict['gaussians_recon']
        
        normal_final = splatter_original_Channel_mvimage_dict[selected_attr]
        normal_final = einops.rearrange(normal_final, 'c (m h) (n w) -> (m n) c h w', m=3, n=2)
        results['imgs_out'] = normal_final
        
        
        # images
        images = []
        masks = []
        cam_poses = []
      
        if self.read_first_view_only:
            vids = [0]
        else:
            vids = [0] + np.arange(1, 7).tolist() + np.random.permutation(np.arange(1, 50 + 6)).tolist()[:self.render_views-7]
        
        elevations, azimuths = [], []
        for vid in vids:
            image_path = os.path.join(renderings_path, f'{vid:03d}.png')
            camera_path = os.path.join(renderings_path, f'{vid:03d}.npy')

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255 # [512, 512, 4] in [0, 1]
            image = torch.from_numpy(image)

            cam = np.load(camera_path, allow_pickle=True).item()
            # c2w = orbit_camera(-cam['elevation'], cam['azimuth'], radius=cam['radius'])
            # c2w = torch.from_numpy(c2w)
            
            # # normalized camera feats as in paper (transform the first pose to a fixed position)
            # radius = torch.norm(c2w[:3, 3])
            # c2w[:3, 3] *= self.cam_radius / radius # 1.5 is the default scale
          
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]
            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(mask.squeeze(0))
            # cam_poses.append(c2w)

            elevations.append(cam['elevation'])
            azimuths.append(cam['azimuth'])

          
        images = torch.stack(images, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        # cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]
        # transform = torch.inverse(cam_poses[1])
        # cam_poses[1:] = transform.unsqueeze(0) @ cam_poses[1:]  # [V, 4, 4] # relative pose

        # resize render ground-truth images, range still in [0, 1]
        # results['imgs_in'] = F.interpolate(images[0:1], size=(self.img_wh[0], self.img_wh[1]), mode='bilinear', align_corners=False).repeat(self.num_views, 1, 1, 1) # [V, C, output_size, output_size]
        # no downsample
        results['imgs_in'] = images[0:1].repeat(self.num_views, 1, 1, 1) # [V, C, output_size, output_size]

        
        results['masks'] = F.interpolate(masks.unsqueeze(1), size=(self.img_wh[0], self.img_wh[1]), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]
        # results['cam_poses'] = cam_poses # [V, 4, 4]
        results['scene_name'] = uid


        # camera embeddings
        # view 0 is cond, view 1-6 are splatter views
        elevations_cond = torch.as_tensor([elevations[0]] * self.num_views).float()  # fixed only use 4 views to train
        azimuths_cond = torch.as_tensor([azimuths[0]] * self.num_views).float()  # fixed only use 4 views to train

        if self.read_first_view_only:
            # print("read_first_view_only")
            camera_embeddings = torch.stack([elevations_cond, torch.tensor([30, -20]*3).to(elevations_cond.dtype), torch.tensor([30,90,150,210,270,330]).to(elevations_cond.dtype)], dim=-1) # (Nv, 3)
        else:
            elevations = torch.as_tensor(elevations[1:(self.num_views+1)]).float()
            azimuths = torch.as_tensor(azimuths[1:(self.num_views+1)]).float()
        
            # print("elevations", elevations, "ele cond", elevations_cond)
            # print("azi", azimuths.shape)
            # assert len(elevations) == self.num_views
            camera_embeddings = torch.stack([elevations_cond, elevations-elevations_cond, (azimuths-azimuths_cond+360)%360], dim=-1) # (Nv, 3)
        
        results['camera_embeddings'] = camera_embeddings
        # print("camera_embeddings", camera_embeddings)

        return results



    def __getitem_joint__(self, idx):
        # We only need imgs_in, imgs_out, camera_embeddings
           
        results = {}

        uid = self.items[idx]
        renderings_path = self.data_path_rendering[uid]
        splatter_uid = self.data_path_vae_splatter[uid]
        
        # splatters
        splatter_original_Channel_mvimage_dict = load_splatter_mv_ply_as_dict(splatter_uid, use_2dgs=self.use_2dgs, return_gassians=self.rendering_loss_2dgs) # [-1,1]

        if self.rendering_loss_2dgs:
            results['gaussians_gt'] = splatter_original_Channel_mvimage_dict['gaussians_gt']
            del splatter_original_Channel_mvimage_dict['gaussians_gt']
            # print("adding results -- gaussians_gt", results['gaussians_gt'].shape)
            if 'gaussians_recon' in splatter_original_Channel_mvimage_dict.keys():
                results['gaussians_recon'] = splatter_original_Channel_mvimage_dict['gaussians_recon']
                del splatter_original_Channel_mvimage_dict['gaussians_recon']
        
        assert len(splatter_original_Channel_mvimage_dict.keys()) == 5
        for key, value in splatter_original_Channel_mvimage_dict.items():
            results[f"{key}_out"] = einops.rearrange(value, 'c (m h) (n w) -> (m n) c h w', m=3, n=2)
            # print(key, results[f"{key}_out"].shape)
            # assert results[f"{key}_out"].shape[-2:] == self.img_wh
        
        # images
        images = []
        masks = []
        cam_poses = []
      
        if self.read_first_view_only:
            # assert NotImplementedError, "check the camera embedding elevations"
            assert self.training, "only training of __joint__ is supported for read_first_view_only"
            vids = [0]
        else:
            vids = [0] + np.arange(1, 7).tolist() + np.random.permutation(np.arange(1, 50 + 6)).tolist()[:self.render_views-7]
        
        elevations, azimuths = [], []
        for vid in vids:
            image_path = os.path.join(renderings_path, f'{vid:03d}.png')
            camera_path = os.path.join(renderings_path, f'{vid:03d}.npy')

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255 # [512, 512, 4] in [0, 1]
            # print("image", image.shape)
            image = torch.from_numpy(image)

            cam = np.load(camera_path, allow_pickle=True).item()
            c2w = orbit_camera(-cam['elevation'], cam['azimuth'], radius=cam['radius'])
            c2w = torch.from_numpy(c2w)
            
            # # normalized camera feats as in paper (transform the first pose to a fixed position)
            # radius = torch.norm(c2w[:3, 3])
            # c2w[:3, 3] *= self.cam_radius / radius # 1.5 is the default scale
          
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]
            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            elevations.append(cam['elevation'])
            azimuths.append(cam['azimuth'])

          
        images = torch.stack(images, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        if not self.read_first_view_only:
            # normalized camera feats as in paper (transform the first pose to a fixed position)
            radius = torch.norm(cam_poses[0, :3, 3])
            cam_poses[:, :3, 3] *= self.cam_radius / radius
            # print("normalize to camera 1")
            transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[1])
            cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]
            # opengl to colmap camera for gaussian renderer
            cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
            results['cam_poses'] = cam_poses # [V, 4, 4]

        # resize render ground-truth images, range still in [0, 1]
        # results['imgs_out'] = F.interpolate(images, size=(self.render_size[0], self.img_wh[1]), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        # results['imgs_in'] = results['imgs_out'][0].unsqueeze(0).repeat(self.num_views, 1, 1, 1) # [1, C, output_size, output_size]
        results['imgs_in'] = images[0:1].repeat(self.num_views, 1, 1, 1) # [V, C, output_size, output_size]
        # results['imgs_in'] = F.interpolate(images[0:1], size=(self.img_wh[0], self.img_wh[1]), mode='bilinear', align_corners=False, antialias=True) # [V, C, output_size, output_size]
        results['imgs_out'] = F.interpolate(images, size=(self.render_size[0], self.render_size[1]), mode='bilinear', align_corners=False, antialias=True) # [V, C, output_size, output_size]

        
        # results['masks'] = F.interpolate(masks.unsqueeze(1), size=(self.img_wh[0], self.img_wh[1]), mode='bilinear', align_corners=False, antialias=True) # [V, 1, output_size, output_size]
        results['scene_name'] = uid
        
        results['fovy'] = cam['fov']
        # print("fovy", results['fovy'])
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ get_proj_matrix(results['fovy']) # [V, 4, 4]
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        

        # rays_embeddings = []
        # for i in range(self.num_views):
        #     rays_o, rays_d = get_rays(cam_poses[i], self.img_wh[0] // 8, self.img_wh[1] // 8, 50) # [h, w, 3]
        #     rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
        #     rays_embeddings.append(rays_plucker)
        # rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        # results['raymaps'] = rays_embeddings
        
        # splatter task embeddings
        splatter_class_all = torch.eye(5).float()
        for i, key in enumerate(gt_attr_keys):
            results[f"{key}_task_embeddings"] = torch.stack([splatter_class_all[i]]*self.num_views, dim=0)
        
        
        # camera embeddings
        # view 0 is cond, view 1-6 are splatter views
        elevations_cond = torch.as_tensor([elevations[0]] * self.num_views).float()  # fixed only use 4 views to train
        azimuths_cond = torch.as_tensor([azimuths[0]] * self.num_views).float()  # fixed only use 4 views to train

        if self.read_first_view_only:
            # print("read_first_view_only")
            camera_embeddings = torch.stack([elevations_cond, torch.tensor([30, -20]*3).to(elevations_cond.dtype), torch.tensor([30,90,150,210,270,330]).to(elevations_cond.dtype)], dim=-1) # (Nv, 3)
        else:
            elevations = torch.as_tensor(elevations[1:(self.num_views+1)]).float()
            azimuths = torch.as_tensor(azimuths[1:(self.num_views+1)]).float()
        
            # print("elevations", elevations, "ele cond", elevations_cond)
            # print("azi", azimuths.shape)
            # assert len(elevations) == self.num_views
            camera_embeddings = torch.stack([elevations_cond, elevations-elevations_cond, (azimuths-azimuths_cond+360)%360], dim=-1) # (Nv, 3)
        
        results['camera_embeddings'] = camera_embeddings
        # print("camera_embeddings (__joint__)", camera_embeddings)

        return results


     
    def __getitem__(self, index):
        try:    
            if self.mix_color_normal:
                data = self.__getitem_mix__(index)
                # data = self.__getitem_joint__(index)
            else:
                data = self.__getitem_joint__(index)
            return data
        except:
            print("load error ", self.all_objects[index%len(self.all_objects)] )
            return self.backup_data

        
if __name__ == "__main__":
    train_dataset = ObjaverseDataset(
        root_dir="/ghome/l5/xxlong/.objaverse/hf-objaverse-v1/renderings",
        size=(128, 128),
        ext="hdf5",
        default_trans=torch.zeros(3),
        return_paths=False,
        total_view=8,
        validation=False,
        object_list=None,
        views_mode='fourviews'
    )
    data0 = train_dataset[0]
    data1  = train_dataset[50]
    # print(data)