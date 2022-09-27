import numpy as np

import bpy

def new_light(name="Sun", light_type='SUN'):
    '''
    Add new sun in scene
    
    Returns:
        light_object: bpy.data.objects['xxx']
    '''
    
    # create light datablock, set attributes
    light_data = bpy.data.lights.new(name=name, type=light_type)

    # create new object with our light datablock
    light_object = bpy.data.objects.new(name=name, object_data=light_data)

    # link light object
    bpy.context.collection.objects.link(light_object)
    # make it active 
    # bpy.context.view_layer.objects.active = light_object
    return light_object

def set_sun( energy = 3, angles = [0, 0, 0], sun_name='Sun' ):
    '''
    Args:
        energy: float
        angles: list[float] [3]
        sun_name: str
    
    Returns:
        sun: light object
    '''
    scene = bpy.data.scenes['Scene']
    try:
        sun = scene.objects[sun_name]
    except:
        sun = new_light(sun_name, 'SUN')

    sun.data.type = 'SUN'
    sun.rotation_euler = [ np.deg2rad(ang) for ang in angles ]
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

def set_world_light_rot(rotation):
    world = bpy.data.worlds.get('World')
    nodes = world.node_tree.nodes
    tex_coord = nodes["Mapping"]
    tex_coord.inputs['Rotation'].default_value = rotation