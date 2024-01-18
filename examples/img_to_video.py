import bpy
import os
import sys

# change to your path here
ROOT_DIR = "F:/Project/blender_tools"
DATA_DIR = os.path.join(ROOT_DIR, "env_data")

sys.path.append(ROOT_DIR)

import importlib

from tools import render
from tools import video

importlib.reload(render)
importlib.reload(video)

# NOTE you need to run with blender UI

def main():
    output_path = os.path.join(ROOT_DIR, f"./doc/images/animate/img2video.mp4")

    image_folder = os.path.join(ROOT_DIR, "./doc/images/animate")
    image_name_list = [ f's_{i}.png' for i in range(5) ]

    target_seconds = 2

    fps = 20

    reso_x = 400
    reso_y = 400

    sequence_name = 'img_seq'

    video.clear_seqs()
    sequence = video.add_img_seq( sequence_name, image_folder, image_name_list)
    
    speed_up_scale = len(image_name_list) / (fps * target_seconds)
    video.speed_up( sequence.name, speed_up_scale)
    
    start_frame = int(sequence.frame_start)
    end_frame = int(sequence.frame_final_duration)

    video.add_color_background( 'background', [1,1,1], start_frame, end_frame*2)

    render.set_render_frames( start_frame, end_frame )

    render.set_render(reso_x, reso_y, format='FFMPEG', fps=fps)
    render.do_render( output_path, animation=True )



if __name__ == '__main__':
    main()