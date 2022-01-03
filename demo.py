import os
import sys
import copy
import math
import numpy as np

import bpy

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)

# change to your path here
ROOT_DIR = "F:/Project/blender_tools"
DATA_DIR = os.path.join(ROOT_DIR, "env_data")

sys.path.append(ROOT_DIR)

import render
import material
import models

def clean_all():
    # remove all materials
    [bpy.data.materials.remove(mat) for mat in bpy.data.materials]
    [bpy.data.meshes.remove(mesh) for mesh in bpy.data.meshes]
    # remove all MESH objects
    # [bpy.data.objects.remove(obj) for obj in bpy.data.objects if obj.type == "MESH"]
    # remove all objects
    [bpy.data.cameras.remove(cam) for cam in bpy.data.cameras]
    [bpy.data.lights.remove(light) for light in bpy.data.lights]
    [bpy.data.objects.remove(obj) for obj in bpy.data.objects]


# clean all object
clean_all()

# add cam
render.set_camera("cam", [0.5, 0.2, 0.5], [0,0,0.1])

# add light
render.set_sun(0.5, [0, np.deg2rad(30), 0])
render.set_hdr_background( os.path.join(DATA_DIR, "background_and_light/artist_workshop_2k.hdr") )

# set render
render.set_render()

# place objects
plane = models.add_shape("plane", 'plane', [0,0,0], [0,0,0], [1,1,1], [1,1,1,1], None )

banana = models.add_model(
    os.path.join(DATA_DIR, "model/banana/from_dae.obj"), 'banana', [0,0,0,1], [0,0,0.02], [0,0,0], 
    os.path.join(DATA_DIR, "model/banana/texture_map.png")
)

cube = models.add_shape("cube", 'cube', [0, 0.2, 0.2], [0,0,np.deg2rad(45)], [0.1,0.05,0.05], [0,0,0,1], None,
    os.path.join(DATA_DIR, "texture/castle_brick_02_red_2k_jpg/castle_brick_02_red_ao_2k.jpg"),
    os.path.join(DATA_DIR, "texture/castle_brick_02_red_2k_jpg/castle_brick_02_red_nor_2k.jpg"),
    # material.hex_to_rgba( material.colors[2], 0.4 ) 
    )

pointcloud = models.add_pointcloud(
    os.path.join(DATA_DIR, "model/ycb_072-a_toy_airplane_scaled/ycb_072-a_toy_airplane_scaled.ply"),
    "pc", [0, -0.2, 0.1], [0.1, 0.1, 0.1], [0.2, 0.4, 1, 1], 0.001 )

# render
render.do_render( os.path.join(ROOT_DIR, "./demo.png") )