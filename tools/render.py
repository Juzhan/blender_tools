import bpy

def do_render(render_path, color_mode='RGBA', \
    bg_transparent=True, format="PNG", adaptive_threshold=0.1, exr_codec='DWAA'):
    '''
    Args:
        render_path: str
    '''

    render = bpy.context.scene.render
    render.use_file_extension = True
    
    render.filepath = render_path
    
    render.film_transparent = bg_transparent
    render.image_settings.file_format = format

    bpy.context.scene.cycles.adaptive_threshold = adaptive_threshold

    if format == 'OPEN_EXR':
        bpy.context.scene.render.image_settings.exr_codec = exr_codec

    animation = False

    if 'mp4' in render_path or format == 'FFMPEG':
        bpy.context.scene.render.ffmpeg.format = 'MPEG4'
        animation = True
        render.image_settings.color_mode = 'RGB'
    else:
        render.image_settings.color_mode = color_mode
        
    bpy.ops.render.render(write_still=True, animation=animation)


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


def compose_background(use_node=True, bg_color=[1,1,1,1]):

    bpy.data.scenes["Scene"].use_nodes = use_node

    tree = bpy.data.scenes["Scene"].node_tree
    mixnode = tree.nodes.new(type="CompositorNodeMixRGB")
    layernode = tree.nodes['Render Layers']
    compositenode = tree.nodes['Composite']
    tree.links.new(layernode.outputs['Image'], mixnode.inputs[2])
    tree.links.new(mixnode.outputs['Image'], compositenode.inputs['Image'])
    mixnode.blend_type = 'MIX'
    mixnode.use_alpha = True
    
    mixnode.inputs[1].default_value = bg_color