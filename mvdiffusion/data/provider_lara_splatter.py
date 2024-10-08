import numpy as np
import glob
import random
import torch
# from dataLoader.utils import build_rays
from scipy.spatial.transform import Rotation as R

import h5py
import os
import einops

from ipdb import set_trace as st
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import Literal, Tuple, Optional, Any

import lmdb
import pickle
from concurrent.futures import ThreadPoolExecutor

from utils.splatter_utils import load_splatter_mv_ply_as_dict, gt_attr_keys

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def fov_to_ixt(fov, reso):
    ixt = np.eye(3, dtype=np.float32)
    ixt[0][2], ixt[1][2] = reso[0]/2, reso[1]/2
    focal = .5 * reso / np.tan(.5 * fov)
    ixt[[0,1],[0,1]] = focal
    return ixt


class LmdbWrapper():
    def __init__(self, path) -> None:
        self.db = lmdb.open(path, subdir=True, readonly=True,
                            lock=False, readahead=False, meminit=False)
        self.txn = self.db.begin(buffers=True, write=False)

    def get_data(self, key):
        # tmp = self.txn.get(("%s_%03d"%(name, vid%256)).encode())
        # return BytesIO(tmp)
        value = self.txn.get(key.encode('utf-8'))
        if value is not None:
            return pickle.loads(value)
        else:
            print(f"Key {key} not found.")
            return None

    

class gobjverse(torch.utils.data.Dataset):
    # def __init__(self, cfg):
    # def __init__(self, opt: Options, name=None, training=True):
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
        ):
        super(gobjverse, self).__init__()

        self.training = not validation
        self.num_views = num_views
        self.img_wh = img_wh
        self.overfit = overfit
        # self.num_cond = 1
        self.mix_color_normal = mix_color_normal 
    
        self.data_root = root_dir
        self.cam_radius = 1.5
        
        self.training =  self.training = not validation
        self.split = 'train' if self.training else 'test'
        self.img_size = np.array([512]*2)

        self.metas = h5py.File(self.data_root, 'r')
        print("Loading data from", self.data_root)
        print("Number of scenes", len(self.metas.keys()))
        scenes_name = np.array(sorted(self.metas.keys())) # [:1000]
        
        
        if 'splits' in scenes_name:
            self.scenes_name = self.metas['splits']['test'][:].astype(str) #self.metas['splits'][self.split]
        else:
            n_scenes = 300000
            i_test = np.arange(len(scenes_name))[::10][:10] # only test 10 scenes
            i_train = np.array([i for i in np.arange(len(scenes_name)) if
                            (i not in i_test)])[:n_scenes]
            
            # debug = True
            # if debug:
            #     i_test = i_test[:2]
            #     i_train = i_train[:100]
            
            if overfit:
                i_test = [90]
                i_train = i_test*1000
                i_test = i_test*10
            self.scenes_name = scenes_name[i_train] if self.split=='train' else scenes_name[i_test]
            
            print("Number of scenes [before reading splatter mv]", len(self.scenes_name))
            
        # splatter mv data
        self.splatter_root = "/mnt/kostas-graid/datasets/xuyimeng/lara/splatter_data/*/*/splatters_mv_inference"
     
        
        ##################### LMDB CREATION ##################################################
        coverage = "overfit" if overfit else "whole"
        self.lmdb_path = f'/mnt/lingjie_cache/xuyimeng/lara/data_path_splatter_{self.split}_{coverage}.lmdb'
        create_lmdb = False
        self.lmdbFiles = None
        
        if True: # create lmdb
            if not os.path.exists(self.lmdb_path):
                create_lmdb = True
            else:
                # Open the LMDB database in read-only mode
                env = lmdb.open(self.lmdb_path, readonly=True)
                with env.begin() as txn:
                    cursor = txn.cursor()
                    num_keys = sum(1 for _ in cursor)
                    print(f"Number of keys in the database: {num_keys}")
                    # Check if the number of keys is less than required
                    if overfit:
                        create_lmdb = (num_keys < 1)
                        print("Overfitting, 0 keys. Creation is needed.") if create_lmdb else print("Overfitting, 1 key is enough.")
                    else:
                        desired_num_keys = 200000 if self.split == 'train' else len(self.scenes_name)
                        create_lmdb = (num_keys < desired_num_keys)
                        print(f"Number of keys in {self.split} split is less than {len(self.scenes_name)}. Creation is needed.") if create_lmdb else print(f"Number of keys in {self.split} split is enough: {num_keys}")
                        
                env.close()
        
            if create_lmdb:
                self.create_lmdb_database()
        ########################################################################

        # read all keys from lmdb
        self.read_all_keys()
       
        self.b2c = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.n_group = 4 # cfg.n_group
        
        self.load_normal = True
        self.fixed_input_views = np.arange(0, 24)[::6].tolist() + [2,22] # same elevation
    
    def worker_init_open_db(self):
        np.random.seed(torch.initial_seed() % 2**32)
        # self.open_lmdb_database()
        self.lmdbFiles = LmdbWrapper(self.lmdb_path)
    
    def create_lmdb_database(self):
        print(f"Creating LMDB database: {self.lmdb_path}...")
        env = lmdb.open(self.lmdb_path, map_size=1099511627776)  # 1 TB map size
        
        def check_scene(scene_path):
            scene_name = scene_path.split('/')[-1].split('_')[-1]
            if scene_name not in self.scenes_name:
                return None
            if not os.path.isdir(scene_path) or not os.path.exists(os.path.join(scene_path, "splatters_mv.pt")):
                return None
            return scene_name, scene_path

        pattern = f"{self.splatter_root}/*"
        all_scene_paths = sorted(glob.glob(pattern))

        final_scenes_name = []
        with ThreadPoolExecutor() as executor:
            results = executor.map(check_scene, all_scene_paths)
            with env.begin(write=True) as txn:
                for result in results:
                    if result:
                        scene_name, scene_path = result
                        txn.put(scene_name.encode('utf-8'), pickle.dumps(scene_path))  # LMDB
                        final_scenes_name.append(scene_name)

        print("Number of scenes [final] [create lmdb]", len(final_scenes_name))
        env.close()

    def open_lmdb_database(self):
        print(f"Opening existing LMDB database: {self.lmdb_path}  ...")
        self.lmdb_env = lmdb.open(self.lmdb_path, readonly=True, max_readers=128, lock=False, readahead=False, meminit=False)

    def close_lmdb_database(self):
        self.lmdb_env.close()
        
    def read_all_keys(self):
        self.open_lmdb_database()
        final_scenes_name = []
        with self.lmdb_env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                # if self.scenes_name.get(key.decode('utf-8')) is not None:
                if np.any(self.scenes_name == key.decode('utf-8')):
                    final_scenes_name.append(key.decode('utf-8'))  # Decode the key if necessary
        if self.overfit:
            final_scenes_name = final_scenes_name*len(self.scenes_name)
        self.scenes_name = final_scenes_name
        print("Number of scenes [final] [read existing lmdb]", len(self.scenes_name))
        self.close_lmdb_database()

    def __del__(self):
        # Close the LMDB environment when the dataloader is destroyed
        self.lmdb_env.close()
       
    
    def __getitem_mix__(self, index):
    
        scene_name = self.scenes_name[index]
        # print("scene_name", scene_name)
        scene_info = self.metas[scene_name]

        # if self.split=='train' and self.n_group > 1:
        #     # print("111")
        #     src_view_id = [random.choices(scene_info['groups'][f'groups_{self.n_group}_{i}'])[0] for i in torch.randperm(self.n_group).tolist()]
        #     view_id = src_view_id + [random.choices(scene_info['groups'][f'groups_{self.n_group}_{i}'])[0] for i in torch.randperm(self.n_group).tolist()]
        # elif self.n_group == 1:
        #     # print("222")
        #     src_view_id = [scene_info['groups'][f'groups_4_{i}'][0] for i in range(1)]
        #     view_id = src_view_id + [scene_info['groups'][f'groups_4_{i}'][-1] for i in range(4)]
        # else:
        #     # print("333")
        #     src_view_id = [scene_info['groups'][f'groups_{self.n_group}_{i}'][0] for i in range(self.n_group)]
        #     view_id = src_view_id + [scene_info['groups'][f'groups_4_{i}'][-1] for i in range(4)]
        
        view_id = self.fixed_input_views # + np.random.permutation(np.arange(0,38))[:(self.num_views-self.opt.num_input_views)].tolist()
        assert len(view_id) == self.num_views

        tar_img, bg_colors, tar_nrms, tar_msks, tar_c2ws, tar_w2cs, tar_ixts, tar_eles, tar_azis = self.read_views(scene_info, view_id, scene_name)
        
        results = {}
    
        images = torch.from_numpy(tar_img).permute(0,3,1,2) # [V, C, H, W]
        normals = torch.from_numpy(tar_nrms).permute(0,3,1,2) # [V, C, H, W]
        # depths = tar_img #[TODO: lara processed data has no depth]
        masks = torch.from_numpy(tar_msks).to(images.dtype) #.unsqueeze(1) # [V, C, H, W]
        cam_poses = torch.from_numpy(tar_c2ws)
        

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        radius = torch.norm(cam_poses[0, :3, 3])
        cam_poses[:, :3, 3] *= self.cam_radius / radius
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]
        

        # rotate normal!
        normal_final = normals
        V, _, H, W = normal_final.shape # [1, h, w, 3]
        normal_final = (transform[:3, :3].unsqueeze(0) @ normal_final.permute(0, 2, 3, 1).reshape(-1, 3, 1)).reshape(V, H, W, 3).permute(0, 3, 1, 2).contiguous()
        # normalize normal
        normal_final = normal_final / (torch.norm(normal_final, dim=1, keepdim=True) + 1e-6)
        # AFTER rotating normal, map normal to range [0,1]
        normal_final = normal_final / 2.0 + 0.5
        # make the bg of normal map to img bg
        # print("bg_color", bg_colors.min(), bg_colors.max(), "normal_final", normal_final.min(), normal_final.max())
        normal_final = normal_final * masks.unsqueeze(1) + (torch.from_numpy(bg_colors)[...,None,None] - masks.unsqueeze(1)) # ! if you would like predict depth; modify here

        
        # read splatter
        splatter_uid = self.lmdbFiles.get_data(scene_name)
    
        if self.overfit and self.split == 'test':
            selected_attr = gt_attr_keys[index%5]
        else:
            selected_attr = random.choice(gt_attr_keys)
        # selected_attr = 'rgbs'
        splatter_original_Channel_mvimage_dict = load_splatter_mv_ply_as_dict(splatter_uid, selected_attr_list=[selected_attr]) # [-1,1]
        normal_final = splatter_original_Channel_mvimage_dict[selected_attr]
        normal_final = einops.rearrange(normal_final, 'c (m h) (n w) -> (m n) c h w', m=3, n=2)
        # print("selected_attr", selected_attr)
        # print("splatter_final:", normal_final.min(), normal_final.max(), normal_final.shape)
        
        
        read_color, read_normal, read_depth = False, True, False # here noraml == splatter
        
        # resize render ground-truth images, range still in [0, 1]
        results['imgs_in'] =  F.interpolate(images[0:1], size=(self.img_wh[0], self.img_wh[1]), mode='bilinear', align_corners=False).repeat(self.num_views, 1, 1, 1) # [1, C, output_size, output_size]

        if read_color:
            results['imgs_out'] = F.interpolate(images, size=(self.img_wh[0], self.img_wh[1]), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        if read_normal:
            if normal_final.shape[-2:] == self.img_wh[0]:
                print("same size")
                pass
            else:
                results['imgs_out'] = F.interpolate(normal_final, size=(self.img_wh[0], self.img_wh[1]), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
            
        results['masks'] = F.interpolate(masks.unsqueeze(1), size=(self.img_wh[0], self.img_wh[1]), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        results['cam_poses'] = cam_poses # [V, 4, 4]

        elevations = torch.as_tensor(tar_eles).float()
        azimuths = torch.as_tensor(tar_azis).float()
        elevations_cond = torch.as_tensor([elevations[0]] * self.num_views).float()  # fixed only use 4 views to train
        azimuths_cond = torch.as_tensor([azimuths[0]] * self.num_views).float()  # fixed only use 4 views to train
        
        
        # print("elevations_cond", elevations_cond)
        # print("elevations", elevations)
        # print("azimuths", azimuths)
        # # print("view_id", view_id)
        # # tar_img, bg_colors, tar_nrms, tar_msks, tar_c2ws, tar_w2cs, tar_ixts, tar_eles, tar_azis = self.read_views(scene_info, [0], scene_name)
        # # print("elevations", elevations  - tar_eles)
        # # print("azimuths", azimuths - tar_azis)
        
        results.update({
            'elevations_cond': torch.deg2rad(elevations_cond),
            'elevations_cond_deg': elevations_cond,
            'elevations': torch.deg2rad(elevations),
            'azimuths':  torch.deg2rad(azimuths),
            'elevations_deg': elevations,
            'azimuths_deg': azimuths,
        })
        
        camera_embeddings = torch.stack([elevations_cond, elevations-elevations_cond, azimuths-azimuths_cond], dim=-1) # (Nv, 3)
        results['camera_embeddings'] = camera_embeddings

        # task embedding
        # normal_class = torch.tensor([1, 0]).float()
        # normal_task_embeddings = torch.stack([normal_class]*self.num_views, dim=0)  # (Nv, 2)
        # color_class = torch.tensor([0, 1]).float()
        # color_task_embeddings = torch.stack([color_class]*self.num_views, dim=0)  # (Nv, 2)
        # if read_normal or read_depth:
        #     task_embeddings = normal_task_embeddings
        # if read_color:
        #     task_embeddings = color_task_embeddings
        
        selected_attr_idx = gt_attr_keys.index(selected_attr)
        splatter_class = torch.tensor([0, 0, 0, 0, 0]).float()
        splatter_class[selected_attr_idx] = 1
        task_embeddings = torch.stack([splatter_class]*self.num_views, dim=0)  # (Nv, 5)
            
        results['task_embeddings'] = task_embeddings

        return results
    

    def __getitem_joint__(self, index):
    
        scene_name = self.scenes_name[index]
        # print("scene_name", scene_name)
        scene_info = self.metas[scene_name]

        # if self.split=='train' and self.n_group > 1:
        #     # print("111")
        #     src_view_id = [random.choices(scene_info['groups'][f'groups_{self.n_group}_{i}'])[0] for i in torch.randperm(self.n_group).tolist()]
        #     view_id = src_view_id + [random.choices(scene_info['groups'][f'groups_{self.n_group}_{i}'])[0] for i in torch.randperm(self.n_group).tolist()]
        # elif self.n_group == 1:
        #     # print("222")
        #     src_view_id = [scene_info['groups'][f'groups_4_{i}'][0] for i in range(1)]
        #     view_id = src_view_id + [scene_info['groups'][f'groups_4_{i}'][-1] for i in range(4)]
        # else:
        #     # print("333")
        #     src_view_id = [scene_info['groups'][f'groups_{self.n_group}_{i}'][0] for i in range(self.n_group)]
        #     view_id = src_view_id + [scene_info['groups'][f'groups_4_{i}'][-1] for i in range(4)]
        
        view_id = self.fixed_input_views # + np.random.permutation(np.arange(0,38))[:(self.num_views-self.opt.num_input_views)].tolist()
        # print("view_id", len(view_id))
        assert len(view_id) == self.num_views

        tar_img, bg_colors, tar_nrms, tar_msks, tar_c2ws, tar_w2cs, tar_ixts, tar_eles, tar_azis = self.read_views(scene_info, view_id, scene_name)
        
        results = {}
    
        images = torch.from_numpy(tar_img).permute(0,3,1,2) # [V, C, H, W]
        normals = torch.from_numpy(tar_nrms).permute(0,3,1,2) # [V, C, H, W]
        # depths = tar_img #[TODO: lara processed data has no depth]
        masks = torch.from_numpy(tar_msks).to(images.dtype) #.unsqueeze(1) # [V, C, H, W]
        cam_poses = torch.from_numpy(tar_c2ws)
        

        # normalized camera feats as in paper (transform the first pose to a fixed position)
        radius = torch.norm(cam_poses[0, :3, 3])
        cam_poses[:, :3, 3] *= self.cam_radius / radius
        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]
        

        # rotate normal!
        normal_final = normals
        V, _, H, W = normal_final.shape # [1, h, w, 3]
        normal_final = (transform[:3, :3].unsqueeze(0) @ normal_final.permute(0, 2, 3, 1).reshape(-1, 3, 1)).reshape(V, H, W, 3).permute(0, 3, 1, 2).contiguous()
        # normalize normal
        normal_final = normal_final / (torch.norm(normal_final, dim=1, keepdim=True) + 1e-6)
        # AFTER rotating normal, map normal to range [0,1]
        normal_final = normal_final / 2.0 + 0.5
        # make the bg of normal map to img bg
        # print("bg_color", bg_colors.min(), bg_colors.max(), "normal_final", normal_final.min(), normal_final.max())
        normal_final = normal_final * masks.unsqueeze(1) + (torch.from_numpy(bg_colors)[...,None,None] - masks.unsqueeze(1)) # ! if you would like predict depth; modify here
        
    
        # resize render ground-truth images, range still in [0, 1]
        results['imgs_out'] = F.interpolate(images, size=(self.img_wh[0], self.img_wh[1]), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['imgs_in'] = results['imgs_out'][0].unsqueeze(0).repeat(self.num_views, 1, 1, 1) # [1, C, output_size, output_size]

        rendering_loss_2dgs = False
        if rendering_loss_2dgs:
            results['masks'] = F.interpolate(masks.unsqueeze(1), size=(self.img_wh[0], self.img_wh[1]), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]
            results['normals_out'] = F.interpolate(normal_final, size=(self.img_wh[0], self.img_wh[1]), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        else:
            del results['imgs_out']
        
        # read splatter attriubtes
        splatter_uid = self.lmdbFiles.get_data(scene_name)
        splatter_original_Channel_mvimage_dict = load_splatter_mv_ply_as_dict(splatter_uid) # [-1,1]
        for key, value in splatter_original_Channel_mvimage_dict.items():
            results[f"{key}_out"] = einops.rearrange(value, 'c (m h) (n w) -> (m n) c h w', m=3, n=2)
            # print(key, results[f"{key}_out"].shape)
            # assert results[f"{key}_out"].shape[-2:] == self.img_wh
        
        
        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        results['cam_poses'] = cam_poses # [V, 4, 4]

        elevations = torch.as_tensor(tar_eles).float()
        azimuths = torch.as_tensor(tar_azis).float()
        elevations_cond = torch.as_tensor([elevations[0]] * self.num_views).float()  # fixed only use 4 views to train
        azimuths_cond = torch.as_tensor([azimuths[0]] * self.num_views).float()  # fixed only use 4 views to train
        
        
        results.update({
            'elevations_cond': torch.deg2rad(elevations_cond),
            'elevations_cond_deg': elevations_cond,
            'elevations': torch.deg2rad(elevations),
            'azimuths':  torch.deg2rad(azimuths),
            'elevations_deg': elevations,
            'azimuths_deg': azimuths,
        })
        
        camera_embeddings = torch.stack([elevations_cond, elevations-elevations_cond, azimuths-azimuths_cond], dim=-1) # (Nv, 3)
        results['camera_embeddings'] = camera_embeddings

        # # task embedding
        # normal_class = torch.tensor([1, 0]).float()
        # normal_task_embeddings = torch.stack([normal_class]*self.num_views, dim=0)  # (Nv, 2)
        # color_class = torch.tensor([0, 1]).float()
        # color_task_embeddings = torch.stack([color_class]*self.num_views, dim=0)  # (Nv, 2)
        # results['normal_task_embeddings'] = normal_task_embeddings
        # results['color_task_embeddings'] = color_task_embeddings
        
        # splatter task embeddings
        splatter_class_all = torch.eye(5).float()
        for i, key in enumerate(gt_attr_keys):
            results[f"{key}_task_embeddings"] = torch.stack([splatter_class_all[i]]*self.num_views, dim=0)

        

        results['scene_name'] = scene_name #uid.split('/')[-1]
      
        return results
    
    
    def __getitem__(self, index):
        try:    
            if self.mix_color_normal:
                data = self.__getitem_mix__(index)
            else:
                data = self.__getitem_joint__(index)
            return data
        except:
            print("load error ", self.all_objects[index%len(self.all_objects)] )
            return self.backup_data

    
    def read_views(self, scene, src_views, scene_name):
        src_ids = src_views
        bg_colors = []
        ixts, exts, w2cs, imgs, msks, normals = [], [], [], [], [], []
        eles, azis = [], []
        for i, idx in enumerate(src_ids):
            
            # if self.split!='train' or i < self.n_group:
            #     bg_color = np.ones(3).astype(np.float32)
            # else:
            #     bg_color = np.ones(3).astype(np.float32)*random.choice([0.0, 0.5, 1.0])
            bg_color = np.ones(3).astype(np.float32)

            bg_colors.append(bg_color)
            
            img, normal, mask = self.read_image(scene, idx, bg_color, scene_name)
            imgs.append(img)
            ixt, ext, w2c, ele, azi = self.read_cam(scene, idx)
            ixts.append(ixt)
            exts.append(ext)
            w2cs.append(w2c)
            msks.append(mask)
            normals.append(normal)
            eles.append(ele)
            azis.append(azi)
        return np.stack(imgs), np.stack(bg_colors), np.stack(normals), np.stack(msks), np.stack(exts), np.stack(w2cs), np.stack(ixts), np.stack(eles), np.stack(azis)

    
    def get_elevation_azimuth(self, z):

        # Normalize the z vector to ensure it's a unit vector
        z = z / np.linalg.norm(z)

        # Elevation: angle with the z-axis
        elevation = np.arccos(z[2])

        # Azimuth: angle in the XY plane
        azimuth = np.arctan2(z[1], z[0])

        # Convert to degrees for easier interpretation
        elevation = np.degrees(elevation) - 90 
        azimuth = np.degrees(azimuth)
        azimuth = (azimuth.astype(np.float16) + 180) % 360 

        # print(f"Elevation: {elevation:.2f} degrees")
        # print(f"Azimuth: {azimuth:.2f} degrees")
        return elevation, azimuth
        
    def read_cam(self, scene, view_idx):
        c2w = np.array(scene[f'c2w_{view_idx}'], dtype=np.float32)
        # print("c2w", c2w.shape, "view_idx", view_idx)   
           
        ele, azi = self.get_elevation_azimuth(c2w[:3,2])

        lara = False
        if not lara:
            # blender world + opencv cam --> opengl world & cam
            c2w[1] *= -1
            c2w[[1, 2]] = c2w[[2, 1]]
            c2w[:3, 1:3] *= -1 # invert up and forward direction
            
        w2c = np.linalg.inv(c2w)
        fov = np.array(scene[f'fov_{view_idx}'], dtype=np.float32)
        ixt = fov_to_ixt(fov, self.img_size)
     
        
        return ixt, c2w, w2c, ele, azi

    def read_image(self, scene, view_idx, bg_color, scene_name):
        
        img = np.array(scene[f'image_{view_idx}'])

        mask = (img[...,-1] > 0).astype('uint8')
        img = img.astype(np.float32) / 255.
        img = (img[..., :3] * img[..., -1:] + bg_color*(1 - img[..., -1:])).astype(np.float32) # [0,1]
        

        if self.load_normal:

            normal = np.array(scene[f'normal_{view_idx}'])
            normal = normal.astype(np.float32) / 255. * 2 - 1.0
            normal[...,[0,1]] = -normal[...,[0,1]]
            normal = normal[...,[1,2,0]]
        

            # rectify normal directions
            normal = normal[..., ::-1]
            normal[..., 0] *= -1
            # print("normal", normal.min(), normal.max())
            normal = normal * np.expand_dims(mask, axis=-1) # to [0, 0, 0] bg
            # + bg_color * (1 - np.expand_dims(mask, axis=-1)) # to [1, 1, 1] bg

            return img, normal, mask

        return img, None, mask


    def __len__(self):
        return len(self.scenes_name)

def get_K_from_params(params):
    K = np.zeros((3, 3)).astype(np.float32)
    K[0][0], K[0][2], K[1][2] = params[:3]
    K[1][1] = K[0][0]
    K[2][2] = 1.
    return K
