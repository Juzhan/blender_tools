import bpy
import os
import sys
import copy
import math
import numpy as np

# dir = os.path.dirname(bpy.data.filepath)
# if not dir in sys.path:
#     sys.path.append(dir)

def do_render(render_path, color_mode='RGBA', \
    bg_transparent=True, format="PNG", adaptive_threshold=0.1, exr_codec='DWAA'):
    '''
    Args:
        render_path: str
    '''

    render = bpy.context.scene.render
    render.use_file_extension = True
    
    render.filepath = render_path
    
    render.image_settings.color_mode = color_mode
    
    render.film_transparent = bg_transparent
    render.image_settings.file_format = format

    bpy.context.scene.cycles.adaptive_threshold = adaptive_threshold

    if format == 'OPEN_EXR':
        bpy.context.scene.render.image_settings.exr_codec = exr_codec

    bpy.ops.render.render(write_still=True)

def set_render( resolution_x=640, resolution_y=480, engine="CYCLES", samples=128, color_management='Standard' ):
    # 'CYCLES' or 'BLENDER_EEVEE'
    bpy.context.scene.render.engine = engine
    bpy.context.scene.cycles.samples = samples
    bpy.context.scene.cycles.device = 'GPU'
    render = bpy.context.scene.render
    render.resolution_x = resolution_x
    render.resolution_y = resolution_y

    
    # bpy.context.scene.view_settings.view_transform = 'Filmic' # gray
    bpy.context.scene.view_settings.view_transform = color_management