import bpy
import os
import sys
import copy
import math
import numpy as np

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)

from transform import *

def do_render(render_path, color_management='Standard', color_mode='RGBA', bg_transparent=True, format="PNG", adaptive_threshold=0.1):
    '''
    Args:
        render_path: str
    '''
    # objects = bpy.context.scene.objects
    # for obj in objects:
    #     obj.select_set(True)
    # bpy.ops.transform.rotate(value=3.14159, orient_axis='Z', orient_type='GLOBAL')

    # bpy.context.scene.view_settings.view_transform = 'Filmic' # gray
    bpy.context.scene.view_settings.view_transform = color_management
    render = bpy.context.scene.render
    render.use_file_extension = True
    render.filepath = render_path
    # In case a previous renderer changed these settings
    #Store as RGB by default unless the user specifies store_alpha as true in yaml
    render.image_settings.color_mode = color_mode
    #set the background as transparent if transparent_background is true in yaml
    render.film_transparent = bg_transparent
    render.image_settings.file_format = format

    bpy.context.scene.cycles.adaptive_threshold = adaptive_threshold

    bpy.ops.render.render(write_still=True)

def new_sun(name="Sun"):
    '''
    Add new sun in scene
    
    Returns:
        light_object: bpy.data.objects['Sun']
    '''
    
    # create light datablock, set attributes
    light_data = bpy.data.lights.new(name=name, type='SUN')

    # create new object with our light datablock
    light_object = bpy.data.objects.new(name=name, object_data=light_data)

    # link light object
    bpy.context.collection.objects.link(light_object)
    # make it active 
    # bpy.context.view_layer.objects.active = light_object
    return light_object

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

def set_render( resolution_x=640, resolution_y=480, engine="CYCLES", samples=128 ):
    # 'CYCLES' or 'BLENDER_EEVEE'
    
    bpy.context.scene.render.engine = engine
    bpy.context.scene.cycles.samples = samples
    bpy.context.scene.cycles.device = 'GPU'
    render = bpy.context.scene.render
    render.resolution_x = resolution_x
    render.resolution_y = resolution_y

def set_camera( cam_name, cam_pos, look_at_pos, fov=60.0, len_type='PERSP' ):
    '''
    Args:
        cam_name: str
        cam_pos: list[float] [3]
        look_at_pos: list[float] [3]
    
    Returns:
        cam_rot: np.array [3]
        cam_dir: np.array [3]
    '''
    
    # start my rotation computation    
    look_at_pos = np.array(look_at_pos)
    cam_pos = np.array(cam_pos)
    
    cam_dir = look_at_pos - cam_pos
    
    scene = bpy.data.scenes['Scene']
    camera = scene.camera
    if camera is None:
        camera = new_camera(cam_name)

    camera.data.type = len_type
    if len_type == 'ORTHO':
        camera.data.ortho_scale = 5
        # camera.data.ortho_scale = 3.278

    camera.data.angle = np.deg2rad(fov)
    camera.rotation_mode = 'XYZ'
    camera.rotation_euler = vec_to_euler( cam_dir )
    camera.location = cam_pos

    cam_rot = np.array(camera.rotation_euler)
    return cam_rot, cam_dir / np.linalg.norm(cam_dir)

def set_sun( energy = 3, sun_rot = [0, 0, 0], sun_name='Sun' ):
    '''
    Args:
        energy: float
        sun_rot: list[float] [3]
        sun_name: str
    
    Returns:
        sun: light object
    '''
    scene = bpy.data.scenes['Scene']
    try:
        sun = scene.objects[sun_name]
    except:
        sun = new_sun(sun_name)

    sun.data.type = 'SUN'
    sun.rotation_euler = sun_rot
    sun.data.energy = energy
    return sun

def set_hdr_background( hdr_path, rotation=[0,0,0], strength=1 ):
    '''
    Add hdr for scene
    
    Args:
        hdr: str
        rotation: list[float] [3]
    
    Returns:
        None
    '''
    world = bpy.data.worlds.get('World')
    
    world.use_nodes = True

    nodes = world.node_tree.nodes
    nodes.clear()

    links = world.node_tree.links
    
    output_node = nodes.new(type="ShaderNodeOutputWorld")
    
    hdr_tex = nodes.new(type="ShaderNodeTexEnvironment")
    mapping = nodes.new(type="ShaderNodeMapping")
    tex_coord = nodes.new(type="ShaderNodeTexCoord")
    bg = nodes.new(type="ShaderNodeBackground")
    
    img = bpy.data.images.load(hdr_path)
    hdr_tex.image = img

    mapping.inputs['Rotation'].default_value = rotation
    bg.inputs['Strength'].default_value = strength

    links.new( tex_coord.outputs['Object'], mapping.inputs['Vector'] )
    links.new( mapping.outputs['Vector'], hdr_tex.inputs['Vector'] )
    links.new( hdr_tex.outputs['Color'], bg.inputs['Color'] )
    links.new( bg.outputs['Background'], output_node.inputs['Surface'] )
