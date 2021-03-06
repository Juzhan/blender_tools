import bpy
import math
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# some color looks nice
colors = [ "#ffffff",  "#9075de", "#e55953", "#e28678", "#dbcc7b", "#9dc756", "#77cb45", "#c0bed3", "#a0c8c0" ]
colors = [ "#FBB5AF", "#FBE06F", "#B0E586", "#8AD4D5", "#718DD5", "#A38DDE", "#9ED68C", "#61abff", "#ffb056", '#A9CBFF' ]

pure_colors = [ 
    [1,1,1, 1],
    [0,1,0, 1],
    [0,0,1, 1],
    [1,1,0, 1],
    [1,0,1, 1],
    [0,1,1, 1],
    [0,0.3,0.8,1],
    [0,0,0,1],
]

def srgb_to_linearrgb(c):
    if   c < 0:       return 0
    elif c < 0.04045: return c/12.92
    else:             return ((c+0.055)/1.055)**2.4

def hex_to_rgb(hex_str):

    h = int('0x' + hex_str[1:], 0)
    
    r = (h & 0xff0000) >> 16
    g = (h & 0x00ff00) >> 8
    b = (h & 0x0000ff)
    return [srgb_to_linearrgb(c/0xff) for c in (r,g,b)]

def hex_to_rgba(hex_str, alpha=1):

    h = int('0x' + hex_str[1:], 0)
    
    r = (h & 0xff0000) >> 16
    g = (h & 0x00ff00) >> 8
    b = (h & 0x0000ff)
    return [srgb_to_linearrgb(c/0xff) for c in (r,g,b)] + [alpha]

#============----------------   material   ----------------============#

def image_material( object, mat_name, color, color_texture, normal_texture, shadow_mode='OPAQUE'):
    '''
    Args:
        object: bpy.data.objects[xxx]
        mat_name: str
        color: list[float] 4
        color_texture: str
            path to color texture image
        normal_texture: str
            path to normal texture image
        shadow_mode: str
            "OPAQUE" for shadow
            "NONE" for none
    
    Returns:
        None
    '''

    obj_name = object.name
    mat_name = mat_name + "_" + obj_name

    mat = bpy.data.materials.get(mat_name)

    if mat == None:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    nodes.clear()
    links = mat.node_tree.links
    
    principled_node = nodes.new('ShaderNodeBsdfPrincipled')
    output_node = nodes.new('ShaderNodeOutputMaterial')

    # principled_node.inputs.get("Base Color").default_value = color
    principled_node.inputs.get("Roughness").default_value = 1.0
    principled_node.inputs.get("Emission Strength").default_value = 0
    principled_node.inputs.get("Alpha").default_value = color[3]

    links.new( principled_node.outputs['BSDF'], output_node.inputs['Surface'] )

    # add color image
    image_node = nodes.new(type="ShaderNodeTexImage")
    image_data = bpy.data.images.load(filepath=color_texture)
    image_data.colorspace_settings.name = 'sRGB'
    image_node.image = image_data
    links.new( image_node.outputs['Color'], principled_node.inputs['Base Color'] )
    
    if normal_texture is not None:
        # add normal image
        normal_map_node = nodes.new(type="ShaderNodeNormalMap")
        normal_map_node.inputs["Strength"].default_value = 1.0
        links.new( normal_map_node.outputs['Normal'], principled_node.inputs['Normal'] )
        
        normal_image_node = nodes.new(type="ShaderNodeTexImage")
        normal_data = bpy.data.images.load(filepath=normal_texture)
        normal_data.colorspace_settings.name = 'Non-Color'
        normal_image_node.image = normal_data
        links.new( normal_image_node.outputs['Color'], normal_map_node.inputs['Color'] )
    
    if color[-1] < 1:
        mat.blend_method = 'BLEND'

    mat.shadow_method = shadow_mode

    obj_mats = object.data.materials
    obj_mats.clear()
    obj_mats.append(mat)
    object.active_material = mat

def pure_color_material( object, mat_name, color, shadow_mode='OPAQUE'):
    '''
    Args:
        object: bpy.data.objects[xxx]
        mat_name: str
        color: list[float] 3
    
    Returns:
        None
    '''
    mat = bpy.data.materials.get(mat_name)

    if mat == None:
        mat = bpy.data.materials.new(mat_name)
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    links = mat.node_tree.links

    output_node = nodes.new('ShaderNodeOutputMaterial')
    rgb_color_node = nodes.new(type="ShaderNodeRGB")
    rgb_color_node.location = -200, 300
    rgb_color_node.color = color

    rgb_color_node.outputs[0].default_value[0] = color[0]
    rgb_color_node.outputs[0].default_value[1] = color[1]
    rgb_color_node.outputs[0].default_value[2] = color[2]
    links.new( rgb_color_node.outputs['Color'], output_node.inputs['Surface'] )
    
    mat.shadow_method = shadow_mode

    obj_mats = object.data.materials
    obj_mats.clear()
    obj_mats.append(mat)
    object.active_material = mat

def color_material( object, mat_name, color=(1,0,0,1), shadow_mode='OPAQUE'):
    '''
    Args:
        object: bpy.data.objects[xxx]
        mat_name: str
        color: list[float] 4
        shadow_mode: str (only work for eevee?)
            "OPAQUE" for shadow \\
            "NONE" for none
    
    Returns:
        None
    '''
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
    
    mat.name = mat_name
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    links = mat.node_tree.links

    principled_node = nodes.new('ShaderNodeBsdfPrincipled')
    output_node = nodes.new('ShaderNodeOutputMaterial')
    

    principled_node.inputs.get("Base Color").default_value = color
    principled_node.inputs.get("Alpha").default_value = color[3]

    # principled_node.inputs.get("Roughness").default_value = 1.0
    # principled_node.inputs.get("Emission Strength").default_value = 0
    
    links.new( principled_node.outputs['BSDF'], output_node.inputs['Surface'] )

    if color[-1] < 1:
        mat.blend_method = 'BLEND'

    mat.shadow_method = shadow_mode

    obj_mats = object.data.materials
    obj_mats.clear()
    obj_mats.append(mat)
    object.active_material = mat

def new_color_material(mat_name, color, shadow_mode='OPAQUE'):
    '''
    Args:
        mat_name: str
        color: list[float] 4
        shadow_mode: str (only work for eevee?)
            "OPAQUE" for shadow \\
            "NONE" for none
    
    Returns:
        material: bpy.data.materials[xxx]
    '''
    
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
    
    mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    links = mat.node_tree.links
    principled_node = nodes.new('ShaderNodeBsdfPrincipled')
    output_node = nodes.new('ShaderNodeOutputMaterial')

    principled_node.inputs.get("Base Color").default_value = color
    principled_node.inputs.get("Alpha").default_value = color[3]

    links.new( principled_node.outputs['BSDF'], output_node.inputs['Surface'] )

    if color[-1] < 1:
        mat.blend_method = 'BLEND'

    mat.shadow_method = shadow_mode
    return mat

def new_pure_color_material( mat_name, color, shadow_mode="NONE"):
    '''
    Args:
        mat_name: str
        color: list[float] 3
        shadow_mode: str (only work for eevee?)
            "OPAQUE" for shadow \\
            "NONE" for none
    
    Returns:
        material: bpy.data.materials[xxx]
    '''
    

    mat = bpy.data.materials.get(mat_name)

    if mat == None:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    nodes.clear()

    links = mat.node_tree.links

    output_node = nodes.new('ShaderNodeOutputMaterial')

    rgb_color_node = nodes.new(type="ShaderNodeRGB")
    rgb_color_node.location = -200, 300
    rgb_color_node.color = color

    rgb_color_node.outputs[0].default_value[0] = color[0]
    rgb_color_node.outputs[0].default_value[1] = color[1]
    rgb_color_node.outputs[0].default_value[2] = color[2]
    links.new( rgb_color_node.outputs['Color'], output_node.inputs['Surface'] )
    
    mat.shadow_method = shadow_mode

    return mat


def set_vertex_color(obj, vertex_colors, vertex_color_name):
    vertex_colors = np.array(vertex_colors)

    mesh = obj.data

    # add vertex color
    if not mesh.vertex_colors:
        mesh.vertex_colors.new(name=vertex_color_name)
    
    mloops = np.zeros((len(mesh.loops)), dtype=np.int)
    mesh.loops.foreach_get("vertex_index", mloops)
    
    # st color for each loop 
    colors = vertex_colors[np.array(mloops)]
    colors = colors.flatten()
    mesh.vertex_colors[vertex_color_name].data.foreach_set("color", colors)

def vertex_color_material( object, mat_name, vertex_color_name):
    '''
    Args:
        object: bpy.data.objects[xxx]
        mat_name: str
        vertex_color_name: str
    
    Returns:
        material: bpy.data.materials[xxx]
    '''
    
    mat = bpy.data.materials.get(mat_name)

    if mat == None:
        mat = bpy.data.materials.new(mat_name)
        mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    nodes.clear()

    links = mat.node_tree.links

    output_node = nodes.new('ShaderNodeOutputMaterial')

    diffuse_node = nodes.new("ShaderNodeBsdfDiffuse")

    attribute_node = nodes.new("ShaderNodeAttribute")

    links.new( diffuse_node.outputs['BSDF'], output_node.inputs['Surface'] )
    links.new( attribute_node.outputs['Color'], diffuse_node.inputs['Color'] )

    attribute_node.attribute_type = 'GEOMETRY'
    attribute_node.attribute_name = vertex_color_name

    obj_mats = object.data.materials
    obj_mats.clear()
    obj_mats.append(mat)
    object.active_material = mat

    return mat


def rgb_color_material( object, mat_name, vertex_color_name, shadow_mode='OPAQUE'):
    '''
    Args:
        object: bpy.data.objects[xxx]
        mat_name: str
        color: list[float] 4
        shadow_mode: str (only work for eevee?)
            "OPAQUE" for shadow \\
            "NONE" for none
    
    Returns:
        None
    '''
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(mat_name)
    
    mat.name = mat_name
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    links = mat.node_tree.links

    vertex_color_node = nodes.new('ShaderNodeVertexColor')
    principled_node = nodes.new('ShaderNodeBsdfPrincipled')
    output_node = nodes.new('ShaderNodeOutputMaterial')
    
    vertex_color_node.layer_name = vertex_color_name

    links.new( vertex_color_node.outputs['Color'], principled_node.inputs['Base Color'] )
    
    links.new( principled_node.outputs['BSDF'], output_node.inputs['Surface'] )

    mat.shadow_method = shadow_mode

    obj_mats = object.data.materials
    obj_mats.clear()
    obj_mats.append(mat)
    object.active_material = mat
    return mat


def set_mat_color(mat_name, color):
    if mat_name not in bpy.data.materials.keys():
        print(" NOT SUCH MATERIAL: %s" % mat_name)
        return
    
    mat = bpy.data.materials.get(mat_name)
    mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = color
    mat.node_tree.nodes["Principled BSDF"].inputs['Alpha'].default_value = color[3]

def change_pc_mat(alpha):
    mat_name = 'PCParticlesMaterial'
    mat = bpy.data.materials[mat_name]
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    if 'Principled BSDF' not in nodes.keys():
        node = nodes.new('ShaderNodeBsdfPrincipled')

    else:
        nodes['Principled BSDF']

    image_node = nodes['Image Texture']
    output_node = nodes['Material Output']
    links.new( node.outputs['BSDF'], output_node.inputs['Surface'] )
    links.new( image_node.outputs['Color'], node.inputs['Base Color'] )

    node.inputs['Alpha'].default_value = alpha
    


#============----------------   color   ----------------============#

def get_cmap_color(data, min_color=None, max_color=None, cmap_type='jet'):
    data = np.array(data)
    
    if min_color:
        colors = [min_color, max_color]
        n_bin = 100
        cmap = LinearSegmentedColormap.from_list('my', colors, N=n_bin)
    else:
        cmap = plt.get_cmap(cmap_type)

    data_max = data.max()
    data_min = data.min()
    
    data = (data - data_min) / (data_max - data_min)
    # data = 1-data
    cmap_colors = cmap(data)

    return cmap_colors

def hsv2rgb(h, s, v):
    """
    Return [R, G, B]
    """
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b

def rgb2hsv(r, g, b):
    """
    Return [H, S, V]
    """
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx
    return h, s, v

