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
import pdb
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import kiui
# from glob 
import glob   

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
        backup_scene: str = "9438abf986c7453a9f4df7c34aa2e65b"
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
        self.root_dir = root_dir
        
        
        # paths = json.load(open(os.path.join(self.root_dir, 'valid_paths.json'), 'r'))
        # self.items = [os.path.join(self.root_dir, path) for path in paths]

        excluded_splits = ["40000-49999"] # used for test
        included_splits = [split for split in os.listdir(self.root_dir) if split not in excluded_splits]
        scene_path_patterns = [os.path.join(self.root_dir, split, "*") for split in included_splits]
        all_scene_paths = []
        for pattern in scene_path_patterns:
            all_scene_paths.extend(sorted(glob.glob(pattern)))
        
        self.data_path_rendering = {}
        for i, scene_path in enumerate(all_scene_paths):
            scene_name = scene_path.split('/')[-1]
            if not os.path.isdir(scene_path):
                continue
            self.data_path_rendering[scene_name] = scene_path

        # self.items = [k for k in self.data_path_rendering.keys()]
        self.items = [ self.data_path_rendering[k] for k in self.data_path_rendering.keys()]
        print('num of data:', len(self.items))

        # naive split
        if self.training:
            self.items = self.items[:-4]
        else:
            self.items = self.items[-4:]


    def get_bg_color(self):
        if self.bg_color == 'white':
            bg_color = np.array([1., 1., 1.], dtype=np.float32)
        elif self.bg_color == 'black':
            bg_color = np.array([0., 0., 0.], dtype=np.float32)
        elif self.bg_color == 'gray':
            bg_color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        elif self.bg_color == 'random':
            bg_color = np.random.rand(3)
        elif self.bg_color == 'three_choices':
            white = np.array([1., 1., 1.], dtype=np.float32)
            black = np.array([0., 0., 0.], dtype=np.float32)
            gray = np.array([0.5, 0.5, 0.5], dtype=np.float32)
            bg_color = random.choice([white, black, gray])
        elif isinstance(self.bg_color, float):
            bg_color = np.array([self.bg_color] * 3, dtype=np.float32)
        else:
            raise NotImplementedError
        return bg_color

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # We only need imgs_in, imgs_out, camera_embeddings
        uid = self.items[idx]
        results = {}

        # load num_views images
        images = []
        masks = []
        cam_poses = []
        
        vid_cnt = 0
        if self.training:
            # vids = np.random.permutation(np.arange(6, 50 + 6)).tolist()[:self.num_views]
            vids = np.arange(0, 10).tolist()[:self.num_views]
        else:
            # fixed views
            # vids = np.arange(36, 73, 4).tolist() + np.arange(100).tolist()
            vids = np.arange(0, 50).tolist()
        
        elevations, azimuths = [], []
        for vid in vids:
            image_path = os.path.join(uid, f'{vid:03d}.png')
            camera_path = os.path.join(uid, f'{vid:03d}.npy')

            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255 # [512, 512, 4] in [0, 1]
            image = torch.from_numpy(image)

            cam = np.load(camera_path, allow_pickle=True).item()
            from kiui.cam import orbit_camera
            c2w = orbit_camera(-cam['elevation'], cam['azimuth'], radius=cam['radius'])
            c2w = torch.from_numpy(c2w)
            # scale up radius to fully use the [-1, 1]^3 space!
            c2w[:3, 3] *= 1.5 / 1.5 # 1.5 is the default scale
          
            image = image.permute(2, 0, 1) # [4, 512, 512]
            mask = image[3:4] # [1, 512, 512]
            image = image[:3] * mask + (1 - mask) # [3, 512, 512], to white bg
            image = image[[2,1,0]].contiguous() # bgr to rgb

            images.append(image)
            masks.append(mask.squeeze(0))
            cam_poses.append(c2w)

            elevations.append(cam['elevation'])
            azimuths.append(cam['azimuth'])

            vid_cnt += 1
            if vid_cnt == self.num_views:
                break

        if vid_cnt < self.num_views:
            print(f'[WARN] dataset {uid}: not enough valid views, only {vid_cnt} views found!')
            n = self.num_views - vid_cnt
            images = images + [images[-1]] * n
            masks = masks + [masks[-1]] * n
            cam_poses = cam_poses + [cam_poses[-1]] * n
          
        images = torch.stack(images, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        transform = torch.inverse(cam_poses[0])
        cam_poses[1:] = transform.unsqueeze(0) @ cam_poses[1:]  # [V, 4, 4] # relative pose

        # resize render ground-truth images, range still in [0, 1]
        results['imgs_out'] = F.interpolate(images, size=(self.img_wh[0], self.img_wh[1]), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['imgs_in'] = results['imgs_out'][0].unsqueeze(0).repeat(self.num_views, 1, 1, 1) # [1, C, output_size, output_size]

        elevations = torch.as_tensor(elevations).float()
        azimuths = torch.as_tensor(azimuths).float()
        elevations_cond = torch.as_tensor([elevations[0]] * self.num_views).float()  # fixed only use 4 views to train
        azimuths_cond = torch.as_tensor([azimuths[0]] * self.num_views).float()  # fixed only use 4 views to train
        camera_embeddings = torch.stack([elevations_cond, elevations-elevations_cond, azimuths-azimuths_cond], dim=-1) # (Nv, 3)
        results['camera_embeddings'] = camera_embeddings
        
        results['masks'] = F.interpolate(masks.unsqueeze(1), size=(self.img_wh[0], self.img_wh[1]), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]
        results['cam_poses'] = cam_poses # [V, 4, 4]
        results['path'] = uid

        observation_masks = torch.ones(self.num_views, 1, 64, 64)
        observation_masks[self.num_cond:] *= 0.0
        results['ob_masks'] = observation_masks

        # rays_embeddings = []
        # for i in range(self.num_views):
        #     rays_o, rays_d = get_rays(cam_poses[i], self.img_wh[0] // 8, self.img_wh[1] // 8, 50) # [h, w, 3]
        #     rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
        #     rays_embeddings.append(rays_plucker)
        # rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        # results['raymaps'] = rays_embeddings
        
        # TODO 1. add task embedding, and train with lara+normal 
        
        
        
        # TODO 2. check elevation and azimuths

        return results
        
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