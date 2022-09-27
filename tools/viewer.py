import numpy as np

import bpy

import tools.transform as transform

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
