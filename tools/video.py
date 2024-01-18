
import bpy

# NOTE you need to run with blender UI

def to_seq_editor(func):
    '''
    switch blender into video editor
    '''
    def wrapper(*args, **kwargs):
        area = bpy.context.area
        old_type = area.type
        area.type = 'SEQUENCE_EDITOR'
        
        ret = func(*args, **kwargs)
        
        area.type = old_type 
        return ret
    return wrapper


def search_seq(name):
    editor = bpy.data.scenes["Scene"].sequence_editor
    for seq in editor.sequences:
        if seq.name == name:
            return seq
    return None

def clear_seqs():    
    sequences = bpy.data.scenes["Scene"].sequence_editor.sequences

    for seq in sequences:
        print(seq.name)
        if 'Speed' not in seq.name:
            sequences.remove(seq)

def remove_seq(seq_name):
    print(seq_name)
    seq = set_active_seq(seq_name)
    sequences = bpy.data.scenes["Scene"].sequence_editor.sequences
    sequences.remove(seq)



@to_seq_editor
def add_img_seq( seq_name, image_folder, image_name_list, frame_start=1, channel_index=3):
    editor = bpy.data.scenes["Scene"].sequence_editor

    files = []

    for image_name in image_name_list:
        files.append( {"name": image_name} )

    bpy.ops.sequencer.image_strip_add(directory=image_folder, files=files, \
                                      relative_path=True, show_multiview=False, \
                                        frame_start=frame_start, frame_end=len(image_name_list), \
                                            channel=channel_index, fit_method='FIT')

    seq = editor.sequences_all[files[0]['name']]
    seq.name = seq_name
    seq.blend_type = 'ALPHA_OVER'

    return seq

@to_seq_editor
def add_video_seq( seq_name, video_path, frame_start=1, channel_index=3):
    editor = bpy.data.scenes["Scene"].sequence_editor

    files = []

    video_name = video_path.split("/")[-1].split("\\")[-1]
    
    bpy.ops.sequencer.movie_strip_add(filepath=video_path, 
                                    relative_path=True, show_multiview=False, 
                                    frame_start=frame_start, channel=channel_index, fit_method='FIT', 
                                    set_view_transform=False, adjust_playback_rate=True, use_framerate=False)

    seq = editor.sequences_all[video_name]
    seq.name = seq_name
    seq.blend_type = 'ALPHA_OVER'

    remove_seq(video_name.split('.')[0] + '.001')

    return seq

@to_seq_editor
def speed_up( seq_name, speed_factor):

    seq = set_active_seq(seq_name)

    fend = seq.frame_final_end
    fstart = seq.frame_final_start
    fduration = seq.frame_final_duration

    bpy.ops.sequencer.effect_strip_add(type='SPEED', frame_start=fstart, frame_end=fend)
    
    speed = search_seq('Speed')
    speed.name = seq_name + '_Speed'

    speed.speed_control = 'MULTIPLY'
    speed.speed_factor = speed_factor

    seq.frame_final_duration = int(fduration / speed_factor)


@to_seq_editor
def set_active_seq(seq_name):
    seq = search_seq(seq_name)
    scene = bpy.data.scenes["Scene"]
    editor = scene.sequence_editor
    editor.active_strip = seq
    return seq


@to_seq_editor
def add_fade(seq_name, fade_type='IN'):
    seq = search_seq(seq_name)
    scene = bpy.data.scenes["Scene"]
    editor = scene.sequence_editor
    editor.active_strip = seq
    bpy.ops.sequencer.fades_add(type=fade_type)
    return seq


@to_seq_editor
def add_color_background(seq_name, color, frame_start, frame_end, channel_index=1):

    bpy.ops.sequencer.effect_strip_add(type='COLOR', frame_start=frame_start, frame_end=frame_end, channel=channel_index)
    
    color_seq = search_seq('Color')
    color_seq.name = seq_name + '_color'

    color_seq.color = color

    return color_seq


    
    
    
    
