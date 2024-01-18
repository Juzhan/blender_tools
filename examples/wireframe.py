import os
import sys
import numpy as np

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
from tools import modifier

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
    width = 700
    height = 150
    render.set_render(width, height)

    scene.switch_to_collection('Cam_and_light')
    # add cam
    viewer.set_camera("cam", pos=[0.41, 0, 0.29], angles=[56, 0, 90])

    # add light
    lighter.set_hdr_background( os.path.join(DATA_DIR, "background_and_light/artist_workshop_2k.hdr") )

    scene.switch_to_collection('Object')

    # add 5 cube
    all_cubes = []
    for i in range(5):
        cube = models.add_shape("cube", f'cube_{i}', [ 0, -0.2 + i * 0.1, 0.02], [0, 0, np.deg2rad(-0)], [0.03,0.03,0.03], [0.3, 0.3, 0.3, 1])
        all_cubes.append(cube)
    
    red = [0.5, 0, 0, 1]
    color_mat = material.rgb_material('red', red[:3])

    #============----------------   exmaple 1   ----------------============#    
    wire_mat = material.wireframe_material('cube_1', wire_color=red, base_color=[1,1,1,1], wire_thickness=0.004)
    material.set_object_mat(all_cubes[1], wire_mat)
    
    #============----------------   exmaple 2   ----------------============#    
    wire_mat2 = material.wireframe_material('cube_2', wire_color=red, base_color=None, wire_thickness=0.004)
    material.set_object_mat(all_cubes[2], wire_mat2)

    #============----------------   exmaple 3   ----------------============#    
    material.set_object_mat(all_cubes[3], color_mat, clear_pre_mats=False)
    modifier.wireframe( all_cubes[3], wire_thickness=0.04, only_wire=True, offset=1)

    #============----------------   exmaple 4   ----------------============#    
    material.set_object_mat(all_cubes[4], color_mat, clear_pre_mats=False)
    modifier.wireframe( all_cubes[4], wire_thickness=0.04, only_wire=False, offset=1)

    # render
    render.do_render( os.path.join(ROOT_DIR, "./doc/images/wireframe.png") )

if __name__ == '__main__':
    main()