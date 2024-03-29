import os
import sys
import copy
import math
import numpy as np

import bpy
import trimesh
# dir = os.path.dirname(bpy.data.filepath)
# if not dir in sys.path:
#     sys.path.append(dir)

# change to your path here
ROOT_DIR = "F:/Project/blender_tools"
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

def color_from_height( points ):
    z = points[:, 2].copy()
    min_z = np.min(z)
    max_z = np.max(z)
    z = (z-min_z) / (max_z - min_z)
    colors = material.get_cmap_color( z, cmap_type='jet')
    return colors


def main():
    # clean all object
    scene.clear_all()

    # set render
    reso_x = 1200
    reso_y = 500
    render.set_render(reso_x, reso_y)

    scene.switch_to_collection('Cam_and_light')
    # add cam
    viewer.set_camera("cam", pos=[0.62, 0.248, 0.58], angles=[56, 0, 111])

    # add light
    lighter.set_sun(0.5, angles=[0, 30, 0])
    lighter.set_hdr_background( os.path.join(DATA_DIR, "background_and_light/artist_workshop_2k.hdr") )

    scene.switch_to_collection('Object')
    # place objects
    plane = models.add_shape("plane", 'plane', [0,-0.25,0], [0,0,0], [0.5,0.25,1], [1,1,1,1], None )
    
    plane2 = models.add_shape("plane", 'plane', [0,0.25,0], [0,0,0], [0.5,0.25,1], [1,1,1,1], None )
    plane2.is_shadow_catcher = True

    curve = models.add_curve( 'curve',
        points=[ [0, -0.044, 0.241], [0.04, 0.024, 0.307], [0.054, 0.06, 0.21] ],
        radius=0.002,
        color=material.color_map['purple-blue'],
        curve_style='o-x->', 
        start_radius=0.004,
        end_radius=0.004,
    )
    scene.move_obj_to_collection( curve, 'Object')

    banana = models.add_model(
        os.path.join(DATA_DIR, "model/banana/from_dae.obj"), 'banana', 
        [0,0,0.02], [0,0,0], [0,0,0,1], 
        texture_path=os.path.join(DATA_DIR, "model/banana/texture_map.png")
    )

    cube = models.add_shape("cube", 'cube', [0, 0.25, 0.25], [0,0,np.deg2rad(-45)], [0.07,0.03,0.03], [0,0,0,1], 
        texture_path=os.path.join(DATA_DIR, "texture/castle_brick_02_red_2k_jpg/castle_brick_02_red_ao_2k.jpg"),
        normal_path=os.path.join(DATA_DIR, "texture/castle_brick_02_red_2k_jpg/castle_brick_02_red_nor_2k.jpg") )

    cube_vert_colors = [ material.pure_colors[i] for i in [0,0,1,1,2,2,3,3] ]
    cube2 = models.add_shape("cube", 'cube2', [0, 0.25, 0.1], [0,0,np.deg2rad(-45)], [0.07,0.03,0.03], cube_vert_colors)

    # add axis
    cube_axis = models.add_axis(name='cube', pos=cube.location, rot=cube.rotation_euler, width=0.004, length=0.1 )
    scene.move_obj_to_collection(cube_axis, 'Object')

    # add point cloud
    scene.switch_to_collection('PC')
    
    pointcloud = models.add_pointcloud(
        os.path.join(DATA_DIR, "model/ycb_072-a_toy_airplane_scaled/ycb_072-a_toy_airplane_scaled.ply"), None,
        "pc", [0, -0.2, 0.1], [0.2, 0.4, 1, 1], 0.001, 0.05 )
    
    pc = trimesh.load_mesh( os.path.join(DATA_DIR, "model/ycb_072-a_toy_airplane_scaled/ycb_072-a_toy_airplane_scaled.ply") )

    points = np.array(pc.vertices)
    colors = color_from_height(points)
    pointcloud2 = models.add_pointcloud(
        os.path.join(DATA_DIR, "model/ycb_072-a_toy_airplane_scaled/ycb_072-a_toy_airplane_scaled.ply"), None,
        "pc2", [0, -0.4, 0.1], colors, 0.001, 0.05 )

    # render
    output_img_path = os.path.join(ROOT_DIR, "./doc/images/page.png")
    render.do_render( output_img_path )

if __name__ == '__main__':
    main()



#============----------------   How to use in blender GUI   ----------------============#
'''
import sys
import bpy

sys.path.append("F:/Project/blender_tools")

import demo
import importlib

importlib.reload(demo)
demo.main()

'''