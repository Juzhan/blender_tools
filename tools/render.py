import bpy

def do_render(render_path, animation=False):
    '''
    Args:
        render_path: str
    '''

    render = bpy.context.scene.render
    
    render.filepath = render_path
    
    bpy.ops.render.render(write_still=True, animation=animation)


def set_render( resolution_x=1000, resolution_y=1000, engine="CYCLES", samples=128, 
               color_management='Standard', color_mode='RGBA',
                bg_transparent=True, format="PNG", adaptive_threshold=0.1, exr_codec='DWAA', fps=30 ):

    # 'CYCLES' or 'BLENDER_EEVEE'
    bpy.context.scene.render.engine = engine
    bpy.context.scene.cycles.samples = samples
    bpy.context.scene.cycles.device = 'GPU'
    render = bpy.context.scene.render
    render.resolution_x = resolution_x
    render.resolution_y = resolution_y

    # bpy.context.scene.view_settings.view_transform = 'Filmic' # gray
    bpy.context.scene.view_settings.view_transform = color_management

    render.use_file_extension = True
    
    render.fps = fps
    
    render.film_transparent = bg_transparent
    render.image_settings.file_format = format

    bpy.context.scene.cycles.adaptive_threshold = adaptive_threshold

    if format == 'OPEN_EXR':
        bpy.context.scene.render.image_settings.exr_codec = exr_codec

    if format == 'FFMPEG':
        bpy.context.scene.render.ffmpeg.format = 'MPEG4'
        render.image_settings.color_mode = 'RGB'
    else:
        render.image_settings.color_mode = color_mode


def set_render_frames(start_frame, end_frame):
    '''
    Args:
        start_frame: int
        end_frame: int
    '''
    
    bpy.data.scenes["Scene"].frame_start = start_frame
    bpy.data.scenes["Scene"].frame_end = end_frame
    bpy.data.scenes["Scene"].frame_current = start_frame


def compose_background(use_node=True, bg_color=[1,1,1,1]):

    bpy.data.scenes["Scene"].use_nodes = use_node

    if use_node:
        tree = bpy.data.scenes["Scene"].node_tree
        mixnode = tree.nodes.new(type="CompositorNodeMixRGB")
        layernode = tree.nodes['Render Layers']
        compositenode = tree.nodes['Composite']
        tree.links.new(layernode.outputs['Image'], mixnode.inputs[2])
        tree.links.new(mixnode.outputs['Image'], compositenode.inputs['Image'])
        mixnode.blend_type = 'MIX'
        mixnode.use_alpha = True
        
        mixnode.inputs[1].default_value = bg_color