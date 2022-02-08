import os
import sys
import copy
import math
import numpy as np

import bpy

# dir = os.path.dirname(bpy.data.filepath)
# if not dir in sys.path:
#     sys.path.append(dir)

# change to your path here
ROOT_DIR = "H:/matlabCode/blender_tools/"
DATA_DIR = os.path.join(ROOT_DIR, "env_data")

sys.path.append(ROOT_DIR)

import importlib

from tools import render
from tools import material
from tools import models
from tools import lighter
from tools import viewer
from tools import scene

importlib.reload(render)
importlib.reload(material)
importlib.reload(models)
importlib.reload(lighter)
importlib.reload(viewer)
importlib.reload(scene)

def main():
    # clean all object
    scene.clear_all()

    # set render
    width = 800
    height = 600
    render.set_render(width, height)

    scene.switch_to_collection('Cam_and_light')
    # add cam
    viewer.set_camera("cam", pos=[0.5, 0.2, 0.5], angles=[56, 0, 111])

    # add light
    lighter.set_sun(0.5, angles=[0, 30, 0])
    lighter.set_hdr_background( os.path.join(DATA_DIR, "background_and_light/artist_workshop_2k.hdr") )

    scene.switch_to_collection('Object')
    # place objects
    plane = models.add_shape("plane", 'plane', [0,0,0], [0,0,0], [1,1,1], [1,1,1,1], None )

    banana = models.add_model(
        os.path.join(DATA_DIR, "model/banana/from_dae.obj"), 'banana', 
        [0,0,0.02], [0,0,0], [0,0,0,1], 
        texture_path=os.path.join(DATA_DIR, "model/banana/texture_map.png")
    )

    cube = models.add_shape("cube", 'cube', [0, 0.2, 0.2], [0,0,np.deg2rad(-45)], [0.07,0.03,0.03], [0,0,0,1], 
        texture_path=os.path.join(DATA_DIR, "texture/castle_brick_02_red_2k_jpg/castle_brick_02_red_ao_2k.jpg"),
        normal_path=os.path.join(DATA_DIR, "texture/castle_brick_02_red_2k_jpg/castle_brick_02_red_nor_2k.jpg")
        )
    # add axis
    cube_axis = models.add_axis(name='cube', pos=cube.location, rot=cube.rotation_euler, width=0.004, length=0.13 )
    scene.move_obj_to_collection(cube_axis, 'Object')
    

    scene.switch_to_collection('PC')
    
    pointcloud = models.add_pointcloud(
        os.path.join(DATA_DIR, "model/ycb_072-a_toy_airplane_scaled/ycb_072-a_toy_airplane_scaled.ply"),
        None,
        "pc",
        [0, -0.2, 0.1], [0.2, 0.4, 1, 1], 0.001, 0.1 )

    curve = scene.add_path(
        [ [0, -0.044, 0.241], [0.04, 0.024, 0.307], [0.054, 0.06, 0.21] ], 'x-x->', node_gap=3
    )

    # render
    render.do_render( os.path.join(ROOT_DIR, "./demo.png") )

if __name__ == '__main__':
    main()