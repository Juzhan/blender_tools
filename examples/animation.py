import bpy
import os
import sys
import numpy as np

# change to your path here
ROOT_DIR = "F:/Project/blender_tools"
DATA_DIR = os.path.join(ROOT_DIR, "env_data")

sys.path.append(ROOT_DIR)

import importlib

from tools import render
from tools import models
from tools import lighter
from tools import viewer
from tools import scene
from tools import animator

importlib.reload(render)
importlib.reload(models)
importlib.reload(lighter)
importlib.reload(viewer)
importlib.reload(scene)
importlib.reload(animator)

@scene.add_model_in_collection
def add_scene(collection_name='Scene'):
    
    plane = models.add_shape("plane", 'plane', [0,0,0], [0,0,0], [0.2,0.2,1], [1,1,1,1], None )

    banana = models.add_model(
        os.path.join(DATA_DIR, "model/banana/from_dae.obj"), 'banana', 
        [0,0,0.02], [0,0,0], [0,0,0,1], 
        texture_path=os.path.join(DATA_DIR, "model/banana/texture_map.png")
    )

    cube = models.add_shape("cube", 'cube', [0, 0.1, 0.03], [0,0,np.deg2rad(-45)], [0.03,0.03,0.03], [0,0,0,1], 
        texture_path=os.path.join(DATA_DIR, "texture/castle_brick_02_red_2k_jpg/castle_brick_02_red_ao_2k.jpg"),
        normal_path=os.path.join(DATA_DIR, "texture/castle_brick_02_red_2k_jpg/castle_brick_02_red_nor_2k.jpg") )

    collection_axis_name = f'{collection_name}_Empty'
    return collection_axis_name

def main():
    # clean all object
    scene.clear_all()

    # set render resolution
    width = 400
    height = 400
    render.set_render(width, height)

    # add cam
    viewer.set_camera("cam", pos=[0.41, 0, 0.31], angles=[56, 0, 90])
    # add light
    lighter.set_hdr_background( os.path.join(DATA_DIR, "background_and_light/artist_workshop_2k.hdr") )

    # add scene
    collection_axis_name = add_scene('Scene')
    axis = bpy.data.objects[collection_axis_name]

    render.compose_background(False)

    #============----------------   exmaple 1   ----------------============#
    
    rot_steps = 5
    for i in range(rot_steps+1):
        # rotate the z axis of scene
        axis.rotation_euler[2] = np.deg2rad( i * 360.0 / rot_steps )
        # render
        render.do_render( os.path.join(ROOT_DIR, f"./doc/images/animate/s_{i}.png") )

    #============----------------   exmaple 2   ----------------============#

    rot_steps = 10
    frame_list = [i*5 for i in range(rot_steps+1)]
    value_list = [ np.deg2rad( i * 360.0 / rot_steps ) for i in range(rot_steps+1) ]

    animator.insert_keys(axis, frame_list, value_list, 'rotation_euler', index=2)

    animator.set_render_frames(0, frame_list[-1])
    # add white backgroud for video
    render.compose_background(bg_color=[1,1,1,1])
    # set format as mp4, use FFMPEG
    render.do_render( os.path.join(ROOT_DIR, f"./doc/images/animate/rot.mp4"), format='FFMPEG' )

    # !!!!!! NOTE
    # the mp4 video render from blender maybe have some coding problem,
    # the video render in Windows, can't play on Mac, I don't know why,
    # so I recommend you using handBrake to transform the video again
    # https://handbrake.fr/


if __name__ == '__main__':
    main()