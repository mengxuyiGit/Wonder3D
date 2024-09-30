import torch
import os
import einops
from pytorch3d.transforms import quaternion_to_axis_angle, axis_angle_to_quaternion
import kiui

# process the loaded splatters into 3-channel images
gt_attr_keys = ['pos', 'opacity', 'scale', 'rotation', 'rgbs']
start_indices = [0, 3, 4, 7, 11]
end_indices = [3, 4, 7, 11, 14]
attr_map = {key: (si, ei) for key, si, ei in zip (gt_attr_keys, start_indices, end_indices)}

### 2DGS
ordered_attr_list = ["pos", # 0-3
                'opacity', # 3-4
                'scale', # 4-7
                "rotation", # 7-11
                "rgbs", # 11-14
            ] # must be an ordered list according to the channels

sp_min_max_dict = {
    "pos": (-0.7, 0.7), 
    "opacity": (-14., 9.),
    "scale": (-10., -2.),
    "rotation": (-5., 5.) #  (-6., 6.)
    }


# def fuse_splatters(splatters):
#     # fuse splatters
#     B, V, C, H, W = splatters.shape
    
#     x = splatters.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
    
#     # # SINGLE VIEW splatter 
#     # x = splatters.permute(0, 1, 3, 4, 2)[:,0].reshape(B, -1, 14)
#     return x

use_inverse_op = False
def load_splatter_mv_ply_as_dict(splatter_dir, device="cpu", range_01=True, use_2dgs=True, selected_attr_list=None, return_gassians=False, denormalization_cam_pose=None):
    
    splatter_mv = torch.load(os.path.join(splatter_dir, "splatters_mv.pt"), map_location='cpu', weights_only=True).detach().cpu()

    # # denormalize the splatter 
    # if denormalization_cam_pose is not None:
    #     # print("denormalization_cam_pose", denormalization_cam_pose)
    #     xyz = splatter_mv[:3]
    #     _, h, w = xyz.shape
    #     xyz = einops.rearrange(xyz, 'c h w -> (h w) c')
    #     xyz_1 = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1 )
    #     xyz_1 = xyz_1 @ denormalization_cam_pose.T
    #     xyz = xyz_1[:, :3] / xyz_1[:, 3:]
    #     # reshape back
    #     xyz = einops.rearrange(xyz, '(h w) c -> c h w', h=h, w=w)
    #     splatter_mv[:3] = xyz
    #     # print("xyz range", xyz.min(), xyz.max())
    # comment: for jgpu version, the splatter is already denormalized
             
        
    # print("\nLoading splatters_mv:", splatter_mv.shape) # [1, 14, 384, 256]

    splatter_3Channel_image = {}
    if return_gassians:
        
        splatter_3Channel_image["gaussians_gt"] = splatter_mv.reshape(14, -1).permute(1,0)
    
    if selected_attr_list is None:
        selected_attr_list = ordered_attr_list
    # print("selected_attr_list:", selected_attr_list)
            
    for attr_to_encode in selected_attr_list:
        si, ei = attr_map[attr_to_encode]
        
        sp_image = splatter_mv[si:ei]
        # print(f"{attr_to_encode}: {sp_image.min(), sp_image.max()}")

        #  map to 0,1
        if attr_to_encode == "pos":
       
            sp_min, sp_max = sp_min_max_dict[attr_to_encode]
            sp_image = (sp_image - sp_min)/(sp_max - sp_min)
        elif attr_to_encode == "opacity":
            if use_inverse_op:
                # print("opacity", sp_image.min(), sp_image.max(), sp_image.mean())
                sp_image = kiui.op.inverse_sigmoid(sp_image)
                # print("opacity [inverse]", sp_image.min(), sp_image.max(), sp_image.mean())
                sp_min, sp_max = sp_min_max_dict[attr_to_encode]
                sp_image =( sp_image - sp_min) / (sp_max - sp_min)
                # print("opacity [inverse] [normalized]", sp_image.min(), sp_image.max(), sp_image.mean())
       
            sp_image = sp_image.repeat(3,1,1)
        elif attr_to_encode == "scale":
            sp_image = torch.log(sp_image)
            sp_min, sp_max = sp_min_max_dict[attr_to_encode]
            sp_image = (sp_image - sp_min)/(sp_max - sp_min)
            if use_2dgs:
                sp_image[0]*=0
        elif  attr_to_encode == "rotation":
            assert (ei - si) == 4
            
            quat = einops.rearrange(sp_image, 'c h w -> h w c')
            axis_angle = quaternion_to_axis_angle(quat)
            sp_image = einops.rearrange(axis_angle, 'h w c -> c h w')
            sp_min, sp_max = sp_min_max_dict[attr_to_encode]
            # print("rotation:", sp_image.view(3,-1).min(dim=1), sp_image.view(3,-1).max(dim=1))
            sp_image = (sp_image - sp_min)/(sp_max - sp_min)
            
        elif attr_to_encode == "rgbs":
            # print("rgbs(utils)", sp_image.min(), sp_image.max())
            pass
        
        if range_01:
            sp_image = sp_image.clip(0,1)
        else:
            # map to [-1,1]
            sp_image = sp_image * 2 - 1
            sp_image = sp_image.clip(-1,1)
        
        # print(f"{attr_to_encode}: {sp_image.min(), sp_image.max(), sp_image.shape}")
        assert sp_image.shape[0] == 3
        splatter_3Channel_image[attr_to_encode] = sp_image.detach().cpu()
    
    return splatter_3Channel_image


def load_splatter_mv_ply_as_dict_debug(splatter_dir, device="cpu", range_01=True, use_2dgs=True, selected_attr_list=None, return_gassians=False):
    
    splatter_mv = torch.load(os.path.join(splatter_dir, "splatters_mv.pt"), map_location='cpu').detach().cpu()
        
    # print("\nLoading splatters_mv:", splatter_mv.shape) # [1, 14, 384, 256]

    splatter_3Channel_image = {}
    if return_gassians:
        splatter_3Channel_image["gaussians_gt"] = splatter_mv.reshape(14, -1).permute(1,0)
    
    if selected_attr_list is None:
        selected_attr_list = ordered_attr_list
    # print("selected_attr_list:", selected_attr_list)
            
    for attr_to_encode in selected_attr_list:
        si, ei = attr_map[attr_to_encode]
        
        sp_image = splatter_mv[si:ei]
        # print(f"{attr_to_encode}: {sp_image.min(), sp_image.max()}")
        print(attr_to_encode, sp_image.min(), sp_image.max(), sp_image.mean())
        
        #  map to 0,1
        if attr_to_encode == "pos":
            sp_min, sp_max = sp_min_max_dict[attr_to_encode]
            sp_image = (sp_image - sp_min)/(sp_max - sp_min)
        elif attr_to_encode == "opacity":
            # sp_image = sp_image.repeat(3,1,1)
            # from ipdb import set_trace as st; st() # debug: use normalization
            if use_inverse_op:
                sp_image = kiui.op.inverse_sigmoid(sp_image)
                sp_min, sp_max = sp_min_max_dict[attr_to_encode]
                sp_image =( sp_image - sp_min) / (sp_max - sp_min)
            sp_image = sp_image.repeat(3,1,1)
            
        elif attr_to_encode == "scale":
            sp_image = torch.log(sp_image)
            sp_min, sp_max = sp_min_max_dict[attr_to_encode]
            sp_image = (sp_image - sp_min)/(sp_max - sp_min)
            if use_2dgs:
                sp_image[0]*=0
        elif  attr_to_encode == "rotation":
            assert (ei - si) == 4
            
            quat = einops.rearrange(sp_image, 'c h w -> h w c')
            axis_angle = quaternion_to_axis_angle(quat)
            sp_image = einops.rearrange(axis_angle, 'h w c -> c h w')
            sp_min, sp_max = sp_min_max_dict[attr_to_encode]
            # print("rotation:", sp_image.view(3,-1).min(dim=1), sp_image.view(3,-1).max(dim=1))
            sp_image = (sp_image - sp_min)/(sp_max - sp_min)
            
        elif attr_to_encode == "rgbs":
            # print("rgbs(utils)", sp_image.min(), sp_image.max())
            pass
        
        if range_01:
            sp_image = sp_image.clip(0,1)
        else:
            # map to [-1,1]
            sp_image = sp_image * 2 - 1
            sp_image = sp_image.clip(-1,1)
        
        # print(f"{attr_to_encode}: {sp_image.min(), sp_image.max(), sp_image.shape}")
        assert sp_image.shape[0] == 3
        splatter_3Channel_image[attr_to_encode] = sp_image.detach().cpu()
    
    debug = True
    if debug and return_gassians:
        gaussians_recon = reconstruct_gaussians(splatter_3Channel_image)
        assert  gaussians_recon.shape == splatter_3Channel_image["gaussians_gt"].shape
        splatter_3Channel_image['gaussians_recon'] = gaussians_recon
        
    return splatter_3Channel_image

def reconstruct_gaussians(splatter_3Channel_image, key_suffix=None):
    recon_list = []
    for k in ordered_attr_list:
        k = k + (key_suffix if key_suffix is not None else "")
        # print(k, splatter_3Channel_image[k].shape)
        recon = denormalize_and_activate(k, splatter_3Channel_image[k][None])
        # print("[recon]", k, recon.shape, recon.min(), recon.max(), recon.mean(), '\n')
        recon_list.append(recon)
    # fuse the reconstructions
    gaussians_recon = torch.cat(recon_list, dim=1) # [B, 14, h, w]
    gaussians_recon = gaussians_recon.reshape(1, 14, -1)[0].permute(1,0)
    
    return gaussians_recon


def reconstruct_gaussians_batch(splatter_3Channel_image, key_suffix=None):
    recon_list = []
    for k in ordered_attr_list:
        k = k + (key_suffix if key_suffix is not None else "")
        # print(k, splatter_3Channel_image[k].shape)
        recon = denormalize_and_activate(k, splatter_3Channel_image[k])
        # print("[recon]", k, recon.shape, recon.min(), recon.max(), recon.mean(), '\n')
        recon_list.append(recon)

    # fuse the reconstructions
    B = splatter_3Channel_image[k].shape[0]
    gaussians_recon = torch.cat(recon_list, dim=1) # [B, 14, h, w]
    gaussians_recon = gaussians_recon.reshape(B, 14, -1).permute(0, 2, 1)
    
    return gaussians_recon

def get_fused_gaussians(splatters_bdv):
    B, D, V, C, H, W  = splatters_bdv.shape
    assert len(ordered_attr_list) == D
    
    decoded_attr_list = []
    for i, _attr in enumerate(ordered_attr_list):
        batch_attr_image = splatters_bdv[:, i]
        batch_attr_image = einops.rearrange(batch_attr_image, 'b v c h w -> (b v) c h w')
        print(f"[before denorm]{_attr}: {batch_attr_image.min(), batch_attr_image.max(), batch_attr_image.shape}")
        decoded_attr = denormalize_and_activate(_attr, batch_attr_image) # B C H W
        print(f"[after denorm]{decoded_attr.min(), decoded_attr.max(), decoded_attr.shape}")
        decoded_attr = einops.rearrange(decoded_attr, '(b v) c h w -> b v c h w', v=V)
        decoded_attr_list.append(decoded_attr)
            
    splatter_mv = torch.cat(decoded_attr_list, dim=2) # [B, v, 14, h, w]
    # B, V, C, H, W = splatter_mv.shape
    gaussians = splatter_mv.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
    return gaussians

def denormalize_and_activate(attr, mv_image): # batchified
    # mv_image: B C H W, in range [0,1]
    
    sp_image_o = mv_image.clip(0,1)
    # print("no clip in denormalize_and_activate")
    
    if attr == "pos":
        sp_min, sp_max = sp_min_max_dict[attr]
        sp_image_o = sp_image_o * (sp_max - sp_min) + sp_min
        # sp_image_o = torch.clamp(sp_image_o, min=sp_min, max=sp_max)
        # print(f"pos:{sp_image_o.min(), sp_image_o.max()}")
    elif attr == "scale":
        sp_min, sp_max = sp_min_max_dict["scale"]
        # sp_image_o = sp_image_o.clip(0,1) 
        sp_image_o = sp_image_o * (sp_max - sp_min) + sp_min
        sp_image_o = torch.exp(sp_image_o)
    elif attr == "opacity":
        sp_image_o = torch.mean(sp_image_o, dim=1, keepdim=True) # avg.
        if use_inverse_op:
            # denormalize
            sp_min, sp_max = sp_min_max_dict["opacity"]
            sp_image_o = sp_image_o *((sp_max - sp_min)) + sp_min
            sp_image_o = torch.sigmoid(sp_image_o)
            # print("opacity [denormalized]", sp_image_o.min(), sp_image_o.max(), sp_image_o.mean())
            
    elif attr == "rotation": 
        # sp_image_o = sp_image_o.clip(0,1) 
        sp_min, sp_max = sp_min_max_dict["rotation"]
        sp_image_o = sp_image_o * (sp_max - sp_min) + sp_min
        ag = einops.rearrange(sp_image_o, 'b c h w -> b h w c')
        quaternion = axis_angle_to_quaternion(ag)
        sp_image_o = einops.rearrange(quaternion, 'b h w c -> b c h w')   
        
    return sp_image_o


## render
from collections import defaultdict
def gs_render_batch(gs, gaussians, data, device):
    gs_results_batch = defaultdict(list)
    for _gaussian, cam_view, cam_view_proj, cam_pos, fovy in zip(gaussians, data['cam_view'].to(device), data['cam_view_proj'].to(device), data['cam_poses'].to(device), data['fovy'].to(device)):
        gs_results = gs.render(gaussians=_gaussian[None], cam_view=cam_view[None], cam_view_proj=cam_view_proj[None], cam_pos=cam_pos[None], fovy=fovy[None])
        for k, v in gs_results.items():
            gs_results_batch[k].append(v)
    for k, v in gs_results_batch.items():
        gs_results_batch[k] = torch.cat(v, dim=0)
    return gs_results_batch
                            