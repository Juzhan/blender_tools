import os
import sys
import copy
import math
import numpy as np

import bpy

from scipy.spatial.transform import Rotation

import tools.transform

def new_camera(name='Camera'):
    '''
    A new camera in scene
    
    Returns:
        camera_object
    '''
    # https://b3d.interplanety.org/en/how-to-create-camera-through-the-blender-python-api/
    camera_data = bpy.data.cameras.new(name=name)
    camera_object = bpy.data.objects.new('Camera', camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    bpy.context.scene.camera = camera_object
    return camera_object

def set_camera( name, pos, angles=None, look_at_pos=None, fov=60.0, len_type='PERSP' ):
    '''
    Args:
        name: str
        pos: list[float] [3]
        angles: list[float] [3]
        look_at_pos: list[float] [3]
    
    Returns:
        cam_rot: np.array [3]
        cam_dir: np.array [3]
    '''
    
    # start my rotation computation    
    pos = np.array(pos)
    
    if angles:
        cam_rot = [ np.deg2rad(ang) for ang in angles ]
    elif look_at_pos:
        look_at_pos = np.array(look_at_pos)
        cam_dir = look_at_pos - pos
        cam_rot = transform.vec_to_euler( cam_dir )

    scene = bpy.data.scenes['Scene']
    camera = scene.camera
    if camera is None:
        camera = new_camera(name)

    camera.data.type = len_type
    if len_type == 'ORTHO':
        camera.data.ortho_scale = 5
        
    camera.data.angle = np.deg2rad(fov)
    camera.rotation_mode = 'XYZ'
    camera.rotation_euler = cam_rot
    camera.location = pos

    return camera


def moving_camera(moving_obj, static_obj, rot_range, dis_range, height_range, lookat_offset_range, frame_id, frames_num):

    start_rot, end_rot = rot_range
    start_dis, end_dis = dis_range
    start_height, end_height = height_range
    
    start_look, end_look = lookat_offset_range
    start_look = np.array(start_look)
    end_look = np.array(end_look)

    rate = frame_id / (1.0 * frames_num)
    if rate > 1: rate = 1

    rot_deg = rate * (end_rot - start_rot) + start_rot
    look = rate * (end_look - start_look) + start_look
    height = rate * (end_height - start_height) + start_height
    
    # x^2
    rate = (frame_id**2) / frames_num**2
    print(rate)
    if rate > 1: rate = 1

    dis = rate * (end_dis - start_dis) + start_dis

    set_camera_pos(moving_obj, static_obj, rot_around_z=-rot_deg, cam_dis=dis, cam_height=height, look_at_offset= look)

    
def set_camera_pos(moving_obj, static_obj, rot_around_z=-60, cam_dis=0.57, obj_rate=1, cam_height=0.205, look_at_offset=[0,0,0.04], update_light=True, follow_static=False):
    static_obj = np.array(static_obj)
    moving_obj = np.array(moving_obj)

    ho_center = static_obj * obj_rate + moving_obj * (1-obj_rate) + look_at_offset
    h2o = static_obj - moving_obj
    h2o[2] = 0
    h2o /= np.linalg.norm(h2o)

    rot = Rotation.from_euler('XYZ', [0, 0, np.deg2rad(rot_around_z)]).as_matrix()
    trans_mat = np.identity(4)
    trans_mat[:3, :3] = rot
    cam_direction = np.zeros(4)
    if follow_static:
        cam_direction[:3] = h2o
    else:
        cam_direction[:3] = [1, 0, 0]

    cam_direction = trans_mat @ cam_direction
    cam_direction = cam_direction[:3]
    
    cam_direction = cam_direction / np.linalg.norm(cam_direction)
    
    cam_lower = ho_center + cam_direction * cam_dis
    
    cam_pos = np.array( [ cam_lower[0], cam_lower[1], cam_height ] )
    
    # update light
    # cam = set_camera("cam", cam_pos, ho_center)
    
    # if update_light:
    #     euler = cam.rotation_euler
    #     set_world_light([0,0, -euler[2] + np.deg2rad(34)])
        

