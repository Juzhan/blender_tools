import os
import sys
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

import bpy

from inspect import getcallargs

from scipy.spatial.transform import Rotation

import tools.models as models

color_map = {
    'purple': [0.503, 0.471, 0.965, 1],
    'deep_purple': [0.141, 0.173, 0.639, 1],
    'gray': [0.1, 0.1, 0.1, 1],
    'red': [0.97, 0.0, 0.0, 1],
    'light-red': [1, 0.1, 0.1, 1],
    'blue': [0., 0.363, 1, 1],
    'light-blue': [0.117, 0.515, 1, 1],
    'pruple-blue': [0.038, 0.208, 860, 1],
    'black': [0, 0, 0, 1],
    'white': [1, 1, 1, 1],
}


#============----------------   Scene   ----------------============#


def clear_all():
    [bpy.data.materials.remove(mat) for mat in bpy.data.materials]
    [bpy.data.meshes.remove(mesh) for mesh in bpy.data.meshes]
    [bpy.data.curves.remove(curve) for curve in bpy.data.curves]
    
    [bpy.data.collections.remove(col) for col in bpy.data.collections]
    [bpy.data.cameras.remove(cam) for cam in bpy.data.cameras]
    [bpy.data.images.remove(img) for img in bpy.data.images]
    [bpy.data.lights.remove(light) for light in bpy.data.lights]
    
    [bpy.data.objects.remove(obj) for obj in bpy.data.objects]


def scene_transform( scene_name, scene_pos, scene_rot ):

    for obj in bpy.data.objects:
        if obj.type == 'EMPTY':
            if obj.name.startswith(scene_name):
                obj.rotation_mode = 'XYZ'
                obj.rotation_euler = scene_rot
                obj.location = scene_pos


#============----------------   Collection basic   ----------------============#

def get_collection(collection_name):
    for c in bpy.data.collections:
        if collection_name == c.name:
            return c
    return c

def has_collection(collection_name):
    return collection_name in bpy.data.collections.keys()

def switch_to_collection(collection_name):
    
    # create now collection
    collection = bpy.data.collections.get(collection_name)
    if collection is None:
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
    
    bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[collection_name]

def move_obj_to_collection(obj, collection_name):
    for col in obj.users_collection:
        col.objects.unlink(obj)
    
    collection = get_collection(collection_name)
    collection.objects.link(obj)


def clear_collection(collection_name):
    if has_collection(collection_name):
        collection = bpy.data.collections.get(collection_name)
        objects = collection.objects
        for o in objects:
            mesh = o.data
            if mesh is not None:
                if type(mesh) == bpy.types.Curve:
                    bpy.data.curves.remove(mesh)
                if type(mesh) == bpy.types.Mesh:
                    bpy.data.meshes.remove(mesh)
            else:
                bpy.data.objects.remove(o)
        
        if 'Hand' in collection_name:
            mat = bpy.data.materials.get(collection_name)
            bpy.data.materials.remove(mat)

        bpy.data.collections.remove(collection)


def get_object_in_collection(collection_name):
    objects = []
    links_name = []
    if has_collection(collection_name):
        collection = bpy.data.collections[collection_name]

        for o in collection.objects:
            if o.type == 'EMPTY': continue
            objects.append(o)
            obj_name = o.name[ len(collection_name) + 1 : ]
            links_name.append(obj_name)

    return objects, links_name

def add_model_in_collection(func):
    def wrapper(*args, **kwargs):
        
        ret = func(*args, **kwargs)
        
        # link all model in that colleciton on a empty link
        all_args = getcallargs(func, *args, **kwargs)
        collection_name = all_args.get('collection_name')

        switch_to_collection(collection_name)
        collection = bpy.data.collections.get(collection_name)
        # create empty obj
        collection_empty_name = '%s_Empty' % collection_name
        empty = bpy.data.objects.get(collection_empty_name)
        if empty == None:
            bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
            empty = bpy.context.active_object
            empty.name = collection_empty_name
            empty.hide_viewport = True

        # get no parent object
        for o in collection.objects:
            if o.parent == None and o != empty:
                o.parent = empty
        return ret

    return wrapper


def hide_collection(collection_name, hide):
    if has_collection(collection_name):
        col = bpy.data.collections.get(collection_name)
        col.hide_render = hide


#============----------------   Path   ----------------============#

def sample_dics(points, min_dist):
    point_len = len(points)
    ret = []
    ret_index = []
    current_p = points[0]
    next_p = points[1]

    for i in range(1, point_len-1):
        if np.linalg.norm(current_p - next_p) > min_dist:
            ret.append(current_p)
            ret_index.append(i)
            current_p = points[i]
        next_p = points[i+1]

    current_p = points[-1]
    ret.append(current_p)
    ret_index.append(point_len-1)
    
    ret_index = np.array(ret_index)
    ret = np.array(ret)

    sort_ids = np.argsort(ret_index)
    ret = ret[sort_ids]
    ret_index = ret_index[sort_ids]
    
    return ret, ret_index
    
def insert_points(points, density=5):
    points_num = len(points)
    ret = []
    for i in range(0, points_num-1):
        new_points = np.linspace(points[i], points[i+1], density)
        for p in new_points:
            ret.append(p)
    return np.array(ret)


@add_model_in_collection
def add_path( points, curve_style='o-o-o', min_dist=0.009, collection_name='PATH', node_gap = 30 ):

    clear_collection(collection_name)
    switch_to_collection(collection_name)

    # insert points
    points = insert_points(points, density=50)
    
    if len(points) == 0: return
    points, points_ids = sample_dics(points, min_dist)
    print(' sample points num: %d' % (len(points)))

    if len(points) == 0: return
    # insert for average gap points
    points = insert_points(points, density=1)
    print(' insert points num: %d' % (len(points)))

    curve_name = "%s_curve" % collection_name

    curve_radius = 0.0025
    start_radius = 0.010
    end_radius = 0.007
    node_radius = 0.007

    curve_color = color_map['purple']
    start_color = color_map['deep_purple']
    curve_color[-1] = 0.4
    start_color[-1] = 0.6
    end_color = None

    # node_gap = 30
    node_points = None

    if 'O' in curve_style:
        node_radius = 0.04

        curve_color = color_map['light-red']
        curve_color[-1] = 0.2

    curve = models.add_curve( curve_name, points, curve_radius, curve_color, \
        curve_style=curve_style, \
        start_radius=start_radius, end_radius=end_radius, node_radius=node_radius, \
        start_color=start_color, end_color=end_color, \
        node_gap=node_gap, node_points=node_points )
    
    if curve is not None and bpy.data.collections.get(collection_name).objects.get(curve_name) == None:
        bpy.data.collections.get(collection_name).objects.link(curve)


#============----------------   Curve   ----------------============#

# https://zhidao.baidu.com/question/437469857900795604.html
def s_curve(rate, k=2, a=0.1, b=0.5):
    # scale_rate = (end_frame) / (end_index - start_index)
    return (2*b)/(1+math.exp( 4 * k * (a - rate)))

