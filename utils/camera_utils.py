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


def az_el_to_points(azimuths, elevations):
    x = np.cos(azimuths)*np.cos(elevations)
    y = np.sin(azimuths)*np.cos(elevations)
    z = np.sin(elevations)
    return np.stack([x,y,z],-1) #


# def set_camera_location(cam_pt):
#     # from https://blender.stackexchange.com/questions/18530/
#     x, y, z = cam_pt # sample_spherical(radius_min=1.5, radius_max=2.2, maxz=2.2, minz=-2.2)
#     # camera = bpy.data.objects["Camera"]
#     camera.location = x, y, z

#     # direction = - camera.location
#     # rot_quat = direction.to_track_quat('-Z', 'Y')
#     # camera.rotation_euler = rot_quat.to_euler()
#     return camera

# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    location, rotation = cam.matrix_world.decompose()[0:2]
    R = np.asarray(rotation.to_matrix())
    t = np.asarray(location)
   
    cam_rec = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)
    R = R.T
    t = -R @ t
    R_world2cv = cam_rec @ R
    t_world2cv = cam_rec @ t

    RT = np.concatenate([R_world2cv,t_world2cv[:,None]],1)
    return RT

import math
import mathutils
def RT_from_ele_azi(elevation, azimuth, radius):
    # Compute translation vector (camera position)
    x = radius * math.cos(elevation) * math.sin(azimuth)
    y = radius * math.sin(elevation)
    z = radius * math.cos(elevation) * math.cos(azimuth)
    translation_vector = mathutils.Vector((x, y, z))

    # Define the view direction (target is at the origin)
    view_direction = -translation_vector.normalized()

    # Define the up vector (initially the Y axis)
    up_vector = mathutils.Vector((0, 1, 0))

    # Compute the right vector (cross product of up and view vectors)
    right_vector = up_vector.cross(view_direction).normalized()

    # Recompute the up vector to ensure orthogonality
    up_vector = view_direction.cross(right_vector).normalized()

    # Construct the rotation matrix
    rotation_matrix = mathutils.Matrix((
        right_vector,
        up_vector,
        view_direction
    )).transposed()  # Transpose because Blender uses column-major order
    
    
    # Construct the 4x4 transformation matrix (RT matrix)
    RT_matrix = mathutils.Matrix((
        (rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2], translation_vector[0]),
        (rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2], translation_vector[1]),
        (rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2], translation_vector[2]),
        (0, 0, 0, 1)
    ))

    # Compute the inverse (camera pose matrix)
    camera_pose = RT_matrix.inverted()
    return camera_pose

import numpy as np
def get_c2w_from_elevation_azimuth(elevation, azimuth, distance=1.0):
    # Convert degrees to radians
    
    elevation = elevation + 90
    elevation = np.radians(elevation)
    
    azimuth = (azimuth - 180) % 360 # align with read_cam
    print('azimuth', azimuth)
    azimuth = np.radians(azimuth)

    # Rotation matrix for azimuth (rotation around z-axis)
    R_z = np.array([
        [np.cos(azimuth), -np.sin(azimuth), 0],
        [np.sin(azimuth), np.cos(azimuth), 0],
        [0, 0, 1]
    ])

    # Rotation matrix for elevation (rotation around y-axis)
    R_y = np.array([
        [np.cos(elevation), 0, np.sin(elevation)],
        [0, 1, 0],
        [-np.sin(elevation), 0, np.cos(elevation)]
    ])

    # Combined rotation matrix (apply azimuth first, then elevation)
    R = R_y @ R_z

    # Translation: camera at a certain distance from the origin
    translation = np.array([0, 0, distance])

    # Create the full 4x4 camera-to-world (c2w) transformation matrix
    c2w = np.eye(4)
    c2w[:3, :3] = R       # Set the rotation part
    c2w[:3, 3] = R @ translation  # Set the translation part (translated and rotated)

    ## align with read_cam
    c2w[1] *= -1
    c2w[[1, 2]] = c2w[[2, 1]]
    c2w[:3, 1:3] *= -1 # invert up and forward direction
    
    print("before w2c", c2w)
    w2c = np.linalg.inv(c2w)

    return w2c
# # Example usage
# elevation = 5.97821044921875  # In degrees
# azimuth = 0.0  # In degrees
# radius = 2.0  # Example radius (distance from the target)

# camera_pose = compute_camera_pose(elevation, azimuth, radius)
# print(camera_pose)
