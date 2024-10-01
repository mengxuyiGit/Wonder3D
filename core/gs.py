import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from diff_gaussian_rasterization import (
#     GaussianRasterizationSettings,
#     GaussianRasterizer,
# )
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

# from core.options import Options
from typing import Dict as Options

import kiui
from kiui.op import inverse_sigmoid
from ipdb import set_trace as st
from utils.point_utils import depth_to_normal
from addict import Dict
from ipdb import set_trace as st


class GaussianRenderer:
    # def __init__(self, opt: Options):
    def __init__(self, output_size=256, fov_degrees=60):
        
        # self.opt = opt
        self.opt = Dict()
        # self.opt.fovy = 45 #todo: passs in fov 
        self.opt.output_size = output_size
        
        
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        
        # # intrinsics
        # self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        # self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        # self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        # self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        # self.proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        # self.proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        # self.proj_matrix[2, 3] = 1
        
        # fovy = 0.69115037 
        print(f"[GS render] fovy: {fov_degrees}")
        fovy = np.deg2rad(fov_degrees)
        self.tan_half_fov = np.tan(0.5 * fovy)
    
    def set_fov(self, fovy):
        # self.opt.fovy = fovy
        # self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.tan_half_fov = np.tan(0.5 * fovy)
        
    def render(self, gaussians, cam_view, cam_view_proj, cam_pos, fovy=None, bg_color=None, scale_modifier=1):
        # gaussians: [B, N, 14]
        # cam_view, cam_view_proj: [B, V, 4, 4]
        # cam_pos: [B, V, 3]
        
        # self.set_fov(fovy)
        
        # if self.opt.verbose_main:
        # print(f"gs.render input: {gaussians.shape}")

        device = gaussians.device
        B, V = cam_view.shape[:2]

        # loop of loop...
        images = []
        alphas = []

        render_normals = []
        render_dists = []
        surf_depths = []
        surf_normals = []
        
        
        for b in range(B):
            if self.opt.verbose_main:
                print(f"render the {b}th gaussian")

            # pos, opacity, scale, rotation, shs
            means3D = gaussians[b, :, 0:3].contiguous().float()
            opacity = gaussians[b, :, 3:4].contiguous().float()
            scales = gaussians[b, :, 4:7].contiguous().float()
            rotations = gaussians[b, :, 7:11].contiguous().float()
            rgbs = gaussians[b, :, 11:].contiguous().float() # [N, 3]

            for v in range(V):
                
                # render novel views
                view_matrix = cam_view[b, v].float()
                view_proj_matrix = cam_view_proj[b, v].float()
                campos = cam_pos[b, v].float()

                raster_settings = GaussianRasterizationSettings(
                    image_height=self.opt.output_size,
                    image_width=self.opt.output_size,
                    tanfovx=self.tan_half_fov,
                    tanfovy=self.tan_half_fov,
                    bg=self.bg_color if bg_color is None else bg_color,
                    scale_modifier=scale_modifier,
                    viewmatrix=view_matrix,
                    projmatrix=view_proj_matrix,
                    sh_degree=0,
                    campos=campos,
                    prefiltered=False,
                    debug=False,
                )
                
                # print(self.tan_half_fov)
                # st()

                rasterizer = GaussianRasterizer(raster_settings=raster_settings)
                
                raster_2dgs = True

                if not raster_2dgs:
                    # Rasterize visible Gaussians to image, obtain their radii (on screen).
                    st()
                    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                        means3D=means3D,
                        means2D=torch.zeros_like(means3D, dtype=torch.float32, device=device),
                        shs=None,
                        colors_precomp=rgbs,
                        opacities=opacity,
                        scales=scales,
                        rotations=rotations,
                        cov3D_precomp=None,
                    )
                else:
                    # print("beofre 2DGS rasterizer...")
                    # print("means3D: ", means3D.shape, means3D.device, "rgbs: ", rgbs.shape, rgbs.device, "opacity: ", opacity.shape, opacity.device, 
                    #       "scales: ", scales.shape, scales.device, "rotations: ", rotations.shape, rotations.device)
                    
                    # print("scales: ", scales.mean(dim=0))
             
                    rendered_image, radii, allmap = rasterizer(
                        means3D=means3D,
                        means2D=torch.zeros_like(means3D, dtype=torch.float32, device=device),
                        shs=None,
                        colors_precomp=rgbs,
                        opacities=opacity,
                        scales=scales[:,1:],
                        rotations=rotations,
                        cov3D_precomp=None,
                    )   
                    
                     # additional regularizations
                    render_alpha = allmap[1:2]

                    # get normal map
                    # transform normal from view space to world space
                    render_normal = allmap[2:5]
                    # render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
                    render_normal = (render_normal.permute(1,2,0) @ (view_matrix[:3,:3].T)).permute(2,0,1)
                    
                    # get median depth map
                    render_depth_median = allmap[5:6]
                    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

                    # get expected depth map
                    render_depth_expected = allmap[0:1]
                    render_depth_expected = (render_depth_expected / render_alpha)
                    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
                    
                    # get depth distortion map
                    render_dist = allmap[6:7]

                    # psedo surface attributes
                    # surf depth is either median or expected by setting depth_ratio to 1 or 0
                    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
                    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
                    depth_ratio = 1
                    surf_depth = render_depth_expected * (1-depth_ratio) + (depth_ratio) * render_depth_median
                    
                    
                    # get viewpoint camera

                    # Initialize the dictionary
                    viewpoint_camera = Dict()
                    
                    viewpoint_camera.world_view_transform = view_matrix
                    viewpoint_camera.image_width = self.opt.output_size
                    viewpoint_camera.image_height = self.opt.output_size
                    viewpoint_camera.full_proj_transform = view_proj_matrix
                    
                    
                    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
                    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
                    surf_normal = surf_normal.permute(2,0,1)
                    # remember to multiply with accum_alpha since render_normal is unnormalized.
                    surf_normal = surf_normal * (render_alpha).detach()

                    
                    # print("finish 2dgs rasterization")

                    # special fro 2DGS postprocessing
                    rendered_alpha = render_alpha
                    
                    render_normals.append(render_normal)
                    render_dists.append(render_dist)
                    surf_depths.append(surf_depth)
                    surf_normals.append(surf_normal)
                    

                rendered_image = rendered_image.clamp(0, 1)

                images.append(rendered_image)
                alphas.append(rendered_alpha)

        images = torch.stack(images, dim=0).view(B, V, 3, self.opt.output_size, self.opt.output_size)
        alphas = torch.stack(alphas, dim=0).view(B, V, 1, self.opt.output_size, self.opt.output_size)
        
        res = {
            "image": images, # [B, V, 3, H, W]
            "alpha": alphas, # [B, V, 1, H, W]
        }
        
        if raster_2dgs:
            res["rend_normal"] = torch.stack(render_normals, dim=0).view(B, V, 3, self.opt.output_size, self.opt.output_size)
            res["rend_dist"] = torch.stack(render_dists, dim=0).view(B, V, 1, self.opt.output_size, self.opt.output_size)
            res["surf_depth"] = torch.stack(surf_depths, dim=0).view(B, V, 1, self.opt.output_size, self.opt.output_size)
            res["surf_normal"] = torch.stack(surf_normals, dim=0).view(B, V, 3, self.opt.output_size, self.opt.output_size) # [B, V, 3, H, W]
            
            # for k, v in res.items():
            #     print(k, v.shape)
        return res

    def save_ply(self, gaussians, path, compatible=True):
        # gaussians: [B, N, 14]
        # compatible: save pre-activated gaussians as in the original paper

        assert gaussians.shape[0] == 1, 'only support batch size 1'

        from plyfile import PlyData, PlyElement
     
        means3D = gaussians[0, :, 0:3].contiguous().float()
        opacity = gaussians[0, :, 3:4].contiguous().float()
        scales = gaussians[0, :, 4:7].contiguous().float()
        rotations = gaussians[0, :, 7:11].contiguous().float()
        shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

        # prune by opacity
        # mask = opacity.squeeze(-1) >= 0.005
        mask = opacity.squeeze(-1) >= -0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        # invert activation to make it compatible with the original ply format
        if compatible:
            opacity = inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)
    
    def load_ply(self, path, compatible=True):

        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        print("Number of points at loading : ", xyz.shape[0])

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