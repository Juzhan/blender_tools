import os
import sys
import copy
import math
import numpy as np
    
import bpy
import bmesh

from space_view3d_point_cloud_visualizer import *

from scipy.spatial.transform import Rotation

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)

from tools.material import *
from tools.transform import *
# 
def set_visible_property(object, diffuse=True, glossy=True, shadow=True):
    if bpy.app.version[0] == 3:
        object.visible_diffuse = diffuse
        object.visible_glossy = glossy
        object.visible_shadow = shadow
    else:    
        object.cycles_visibility.diffuse = diffuse
        object.cycles_visibility.glossy = glossy
        object.cycles_visibility.shadow = shadow

# https://github.com/itsumu/point_cloud_renderer
class PointCloudMaker():
    def __init__(self, active_object, type='ico_sphere'):
        self.type = type
        self.active_object = bpy.context.active_object
        bpy.app.debug_value = 0
        self.instancers = []

        self.controller = PCVControl(active_object)

    # Utility function to generate numpy points from ascii point cloud files
    def generate_points_from_pts(self, filename):
        points = np.loadtxt(filename)
        return points

    def load_points(self, filename):
        # import open3d as o3d
        # pc = o3d.io.read_point_cloud(filename)
        # return np.array(pc.points)

        import trimesh
        pc = trimesh.load_mesh(filename)
        return np.array(pc.vertices)

    # Convert points to spheres
    def convert_to_spheres(self, points=None, name='pc', color=[1,0,0,1], sphere_radius=0.01):

        if self.type == 'ico_sphere':
            bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=sphere_radius)
        elif self.type == 'cube':
            bpy.ops.mesh.primitive_cube_add(size=sphere_radius)
        template_sphere = bpy.context.active_object
        template_sphere.name = name + "_pc"

        uc_key = template_sphere.data.vertex_layers_float.new(name='uc')
        num = len(uc_key.data)
        for i, v in enumerate(uc_key.data):
            v.value = i / num

        bpy.ops.mesh.primitive_plane_add()
        instancer = bpy.context.active_object
        instancer.name = name + "_pc_parent"
        instancer_mesh = instancer.data
        bm = bmesh.new()
        for i in range(points.shape[0]):
            vert = bm.verts.new()
            vert.co = points[i, :]

        uc_key = bm.verts.layers.float.new('uc')
        num = len(bm.verts)
        for i, v in enumerate(bm.verts):
            v[uc_key] = i / num

        bm.to_mesh(instancer_mesh)

        uc_key = instancer.data.vertex_layers_float.new(name='uc')
        template_sphere.parent = instancer
        instancer.instance_type = 'VERTS'

        if len(color) <= 4:
            color_material(template_sphere, template_sphere.name, color=color, shadow_mode='NONE')
        # if True:
        else:
            vcolors = template_sphere.data
            
            vertex_color_name = '%s_pc_col' % name
            if not vcolors.vertex_colors:
                vcolors.vertex_colors.new(name=vertex_color_name)
                # [  vcolors.vertex_colors.remove(vc) for vc in vcolors.vertex_colors ]
            

            loops = vcolors.loops

            if self.type == 'ico_sphere':
                num = 642
            elif self.type == 'cube':
                num = 8

            max_vi = 0
            for loop_i, loop in enumerate(loops):
                vi = loop.vertex_index
                li = loop.index
                
                if max_vi < vi: max_vi = vi
                
                c = color[vi]

                vcolors.vertex_colors[vertex_color_name].data[loop_i].color = c

            rgb_color_material(template_sphere, template_sphere.name, vertex_color_name)
        
        set_visible_property(template_sphere, False, False, False)
        self.instancers.append(instancer)
        bm.free()
        return instancer

    # Points should be generated first
    def convert_to_sphere_rgb(self, object_name, mesh_size):
        # start_time = time.time()

        if not object_name:
            object_name = self.object_name

        pcv = self.active_object.point_cloud_visualizer
        pcv.filepath = object_name + '.pts' # Used for object id
        # pcv.filepath = filename

        pcv.mesh_size = mesh_size
        pcv.mesh_type = 'INSTANCER'
        pcv.mesh_base_sphere_subdivisions = 4
        bpy.ops.point_cloud_visualizer.convert()

        # Rotate to normal pose
        target_name = '{}-instancer'.format(object_name)
        pc = bpy.data.objects[target_name]
        set_visible_property(pc, False, False, False)
        target = bpy.data.objects['Icosphere']
        set_visible_property(target, False, False, False)

        return pc


    def draw(self, points, colors):
        self.controller.draw(vs=points, cs=colors)


    # Convert points to spheres
    def convert_to_spheres_old(self, points=None, name='pc', color=[1,0,0,1], sphere_radius=0.01):
        # start_time = time.time()

        if self.type == 'ico_sphere':
            bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=sphere_radius)
        elif self.type == 'cube':
            bpy.ops.mesh.primitive_cube_add(size=sphere_radius)
        template_sphere = bpy.context.active_object
        template_sphere.name = name + "_pc"

        bpy.ops.mesh.primitive_plane_add()
        instancer = bpy.context.active_object
        instancer.name = name + "_pc_parent"
        instancer_mesh = instancer.data
        bm = bmesh.new()
        for i in range(points.shape[0]):
            bm.verts.new().co = points[i, :]
        bm.to_mesh(instancer_mesh)
        template_sphere.parent = instancer
        instancer.instance_type = 'VERTS'
        color_material(template_sphere, template_sphere.name, color=color, shadow_mode='NONE')

        # display shadow on cycles mode
        
        set_visible_property(template_sphere, False, False, False)


        self.instancers.append(instancer)

    # Clear generated instancers (point spheres)
    def clear_instancers(self):
        active_object = bpy.context.active_object
        active_object_name = active_object.name
        active_object.select_set(False)

        for instancer in self.instancers:
            instancer.select_set(True)
            for child in instancer.children:
                child.select_set(True)

        bpy.ops.object.delete()  # Delete selected objects
        active_object = bpy.context.scene.objects[active_object_name]
        active_object.select_set(True)

    # Reselect active object
    def post_process(self):
        view_layer = bpy.context.view_layer
        bpy.ops.object.select_all(action='DESELECT')
        self.active_object.select_set(True)
        view_layer.objects.active = self.active_object


# Clear intermediate stuff
def reset(pcv_controller=None, clear_instancers=False, clear_database=False):
    #start_time = time.time()

    if pcv_controller != None:
        pcv_controller.reset()
    if (clear_instancers):
        # Clear materials
        for material in bpy.data.materials:
            if 'PCParticles' in material.name or 'material' in material.name or 'Material' in material.name:
                bpy.data.materials.remove(material)
        # Delete previously created instancers
        def delete_hierarchy(obj):
            for child in obj.children:
                delete_hierarchy(child)
            obj.select_set(True)
            bpy.ops.object.delete()

        active_object = bpy.context.active_object
        for object in bpy.data.objects:
            if '-instancer' in object.name:
                active_target_name = str(active_object.name)
                active_object.select_set(False)
                delete_hierarchy(object)
                active_object = bpy.data.objects[active_target_name]

    
    if clear_database:
        # Clear meshes
        for mesh in bpy.data.meshes:
            if (not 'Cube' in mesh.name) and (not 'Cone' in mesh.name) \
                    and (not 'Cylinder' in mesh.name):
                bpy.data.meshes.remove(mesh)
            
        # Clear images
        for image in bpy.data.images:
            bpy.data.images.remove(image)



# code from https://github.com/ianhuang0630/blender_render
def pc_rgb( pc_name, points, colors, obj_pos, obj_scale, point_radius=0.01, \
    diffuse=False, glossy=False, shadow=False ):

    points *= obj_scale
    points += obj_pos

    sphere_mesh = bpy.data.meshes.new('sphere')
    sphere_bmesh = bmesh.new()

    if bpy.app.version[0] == 3:
        bmesh.ops.create_icosphere(sphere_bmesh, subdivisions=1, radius=point_radius)
    else:
        bmesh.ops.create_icosphere(sphere_bmesh, subdivisions=2, diameter=point_radius*2)
    
    sphere_bmesh.to_mesh(sphere_mesh)
    sphere_bmesh.free()
    
    sphere_verts = np.array([[v.co.x, v.co.y, v.co.z] for v in sphere_mesh.vertices])
    sphere_faces = np.array([[p.vertices[0], p.vertices[1], p.vertices[2]] for p in sphere_mesh.polygons])
    # 
    verts = (np.expand_dims(sphere_verts, axis=0) + np.expand_dims(points, axis=1)).reshape(-1, 3)
    faces = (np.expand_dims(sphere_faces, axis=0) + (np.arange(points.shape[0]) * sphere_verts.shape[0]).reshape(-1, 1, 1)).reshape(-1, 3)

    vert_colors = np.repeat(colors, sphere_verts.shape[0], axis=0).astype(dtype='float64')
    vert_colors = vert_colors[faces.reshape(-1), :]
    # 
    verts[:, 2] -= verts.min(axis=0)[2]
    # 
    verts = verts.tolist()
    faces = faces.tolist()
    vert_colors = vert_colors.tolist()
    # 
    mesh = bpy.data.meshes.new(pc_name)
    mesh.from_pydata(verts, [], faces)
    mesh.validate()

    vertex_color_name = pc_name + '_Col'
    mesh.vertex_colors.new(name= vertex_color_name)
    mesh_vert_colors = mesh.vertex_colors[ vertex_color_name]
    
    for i, c in enumerate(mesh_vert_colors.data):
        c.color = vert_colors[i]  + [1.0]

    obj = bpy.data.objects.new(pc_name, mesh)
    bpy.context.collection.objects.link(obj)

    # add material
    vertex_color_material(obj, pc_name, vertex_color_name )
    
    set_visible_property(obj, diffuse, glossy, shadow)

#=============================#
#=                           =#
#=        Add model          =#
#=                           =#
#=============================#

def add_model( filename, obj_name, obj_pos, obj_rot, obj_color, obj_scale=1, \
        texture_path=None, normal_path=None, \
        diffuse=False, glossy=False, shadow=True, use_auto_smooth=False, mat_name=None ):
    '''
    Args:
        filename: str
        obj_name: str
        obj_color: list[float] [4]
        obj_rot: list[float] [3]
        obj_pos: list[float] [3]
    
    Returns:
        object: bpy.data.object
    '''

    if filename.endswith('obj'):
        bpy.ops.import_scene.obj( filepath=filename )
    elif filename.endswith('ply'):
        bpy.ops.import_mesh.ply( filepath=filename )
    elif filename.endswith('stl'):
        bpy.ops.import_mesh.stl( filepath=filename )
    
    object = bpy.context.selected_objects[0]
    object.name = obj_name
    object.rotation_mode = 'XYZ'
    object.rotation_euler = [0, 0, 0]
    object.scale = [obj_scale, obj_scale, obj_scale]

    mesh = object.data
    if mesh is not None:
        mesh.name = obj_name

    if obj_rot is not None:
        if len(obj_rot) == 3:
            object.rotation_mode = 'XYZ'
            object.rotation_euler = obj_rot
            object.location = obj_pos
        elif len(obj_rot) == 4:
            object.rotation_mode = 'QUATERNION'
            object.rotation_quaternion = obj_rot
            object.location = obj_pos

    set_visible_property(object, diffuse, glossy, shadow)

    object.data.use_auto_smooth = use_auto_smooth

    if mat_name is None:
        mat_name = obj_name

    if texture_path is not None:
        image_material(object, mat_name, obj_color, texture_path, normal_path)
    elif obj_color is not None:
        color_material(object, mat_name, obj_color)
    return object

def add_pointcloud( filename, obj_name, obj_pos, sphere_color, sphere_radius, obj_scale=1 ):
    '''
    Args:
        filename: str
        obj_name: str
        sphere_color: list[float] [4] 
            in [0,1]
        sphere_radius: float
    
    Returns:
        None
    '''	
    # Generate point cloud from file
    pcm = PointCloudMaker(bpy.context.active_object)
    
    if filename.endswith('obj'):
        pcd = pcm.load_points(filename)
    elif filename.endswith('ply'):
        pcd = pcm.load_points(filename)
    else:
        pcd = pcm.generate_points_from_pts(filename)
    
    pcd *= obj_scale

    pcd += np.array(obj_pos)
    
    # Create spheres from point clouds
    return pcm.convert_to_spheres(points=pcd, name = obj_name, color=sphere_color, sphere_radius=sphere_radius)


def draw_pointcloud( filename, obj_name, obj_pos, sphere_colors, sphere_radius, obj_scale=1 ):
    '''
    Args:
        filename: str
        obj_name: str
        sphere_colors: list[float] [n x 4] 
            in [0,1]
        sphere_radius: float
    
    Returns:
        None
    '''	
    # Generate point cloud from file
    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))

    active_object = bpy.context.active_object
    active_object.name = obj_name + '_EMPTY'
    
    # Generate point cloud from file
    pcm = PointCloudMaker(active_object)

    # pcm.controller.reset()
    reset(pcv_controller=pcm.controller, clear_instancers=True)
    
    if filename.endswith('obj'):
        pcd = pcm.load_points(filename)
    elif filename.endswith('ply'):
        pcd = pcm.load_points(filename)
    else:
        pcd = pcm.generate_points_from_pts(filename)
    
    pcd *= obj_scale

    pcd += np.array(obj_pos)
    
    # Create spheres from point clouds
    pcm.draw(points=pcd, colors=sphere_colors )
    ret = pcm.convert_to_sphere_rgb(obj_name, sphere_radius)
    pcm.post_process()

    return ret


def add_shape(obj_type, obj_name, obj_pos, obj_rot, obj_size, obj_color=[0.8,0.8,0.8,1], \
        shape_setting=None, \
        texture_path=None, normal_path=None, \
        diffuse=False, glossy=False, shadow=True, mat_name=None ):
    '''
    Args:
        obj_type: str
            >>> 'cube'
            >>> 'sphere'
            >>> 'cone'
            >>> 'plane'
            >>> 'cylinder'
            >>> 'torus'

        obj_name: str
        obj_pos: list[float] [3]
        obj_rot: list[float] [3]
        obj_color: list[float] [4]
        shape_setting: dict
            specify for shape
    
    Returns:
        object: bpy.data.object
    '''
    if obj_type == 'cylinder':
        bpy.ops.mesh.primitive_cylinder_add(vertices=shape_setting['vertices'])
        init_name = 'Cylinder'

    elif obj_type == 'plane':
        bpy.ops.mesh.primitive_plane_add()
        init_name = 'Plane'

    elif obj_type == 'cube':
        bpy.ops.mesh.primitive_cube_add()
        init_name = 'Cube'

    elif obj_type == 'sphere':
        seg = 32
        rings = 32
        if shape_setting:
            seg = shape_setting['segments']
            rings = shape_setting['rings']
        bpy.ops.mesh.primitive_uv_sphere_add(segments=seg, ring_count=rings)
        init_name = 'Sphere'
        
    elif obj_type == 'cone':
        bpy.ops.mesh.primitive_cone_add(radius1=shape_setting['radius1'], radius2=shape_setting['radius2'], depth=shape_setting['depth'])
        init_name = 'Cone'

    elif obj_type == 'torus':
        major_segments = 48
        minor_segments = 8
        minor_radius = 0.1
        major_radius = 0.6
        if shape_setting is not None:
            if 'minor_segments' in shape_setting:
                minor_segments = shape_setting['minor_segments']
            if 'major_segments' in shape_setting:
                major_segments = shape_setting['major_segments']
            if 'minor_radius' in shape_setting:
                minor_radius = shape_setting['minor_radius']
            if 'major_radius' in shape_setting:
                major_radius = shape_setting['major_radius']

        bpy.ops.mesh.primitive_torus_add(
            major_segments=major_segments,
            minor_segments=minor_segments,
            minor_radius=minor_radius,
            major_radius=major_radius)

        init_name = 'Torus'

    for o in bpy.data.objects:
        if o.name == init_name:
            object = o
            break
    object = bpy.context.active_object

    object.name = obj_name
    
    object.data.name = obj_name

    object.scale = obj_size
    object.rotation_mode = 'XYZ'
    object.rotation_euler= obj_rot
    object.location= obj_pos

    set_visible_property(object, diffuse, glossy, shadow)

    if mat_name is None:
        mat_name = obj_name

    if texture_path is not None:
        image_material(object, mat_name, obj_color, texture_path, normal_path)
    elif obj_color is not None:
        color_material(object, mat_name, obj_color)

    return object


def add_line(start, end, width, color, with_arrow, name='arrow'):
    start = np.array(start)
    end = np.array(end)
    cam_dir = end - start
    length = np.linalg.norm(end-start)
    
    if with_arrow:
        cone = add_shape('cone', 'arrow_cone_' + name, end, vec_to_euler(-cam_dir), [1,1,1], None,
        {
            'radius1': width,
            'radius2': 0,
            'depth': width*2,
        } )
    
    center = (end+start)/2
    
    
    cylinder = add_shape( 'cylinder', 'arrow_cylinder_' + name, center, \
        vec_to_euler(cam_dir), [width*0.4, width*0.4, length/2 - width/2 ], None,  { 'vertices': 10 } )

    if with_arrow:
        objects = [cylinder, cone]
    else:
        objects = [cylinder]

    obj = join_objects(objects, name)

    mat_name = 'arrow_' + name
    color_material(obj, mat_name, color)

def add_axis(pos=[0,0,0], rot=[0,0,0], quat=None, width=0.02, length=0.5, name='origin'):
    pos = np.array(pos)

    if rot:
        rot = np.array(rot)
        mat = Rotation.from_euler('XYZ', rot).as_matrix()
    elif quat:
        quat = np.array(quat)
        mat = Rotation.from_quat(quat).as_matrix()

    mx = mat[:,0]
    my = mat[:,1]
    mz = mat[:,2]
    
    # add_line( pos, pos+mx * length, width, [0.8,0,0,1], True, name + '_x' )
    # add_line( pos, pos+my * length, width, [0,0.8,0,1], True, name + '_y' )
    # add_line( pos, pos+mz * length, width, [0,0,0.8,1], True, name + '_z' )

    x = add_curve(name + '_x', [pos, pos+mx * length], width, [0.8,0,0,1], end_radius=width*2, curve_style='x-x->' )
    y = add_curve(name + '_y', [pos, pos+my * length], width, [0,0.8,0,1], end_radius=width*2, curve_style='x-x->' )
    z = add_curve(name + '_z', [pos, pos+mz * length], width, [0,0,0.8,1], end_radius=width*2, curve_style='x-x->' )
    return join_objects([x, y, z], name + '_axis')

# code from 
# https://blender.stackexchange.com/questions/153742/curve-from-a-set-of-points
# very super cool 
def add_curve_from_points( name, points, radius, color=[1,1,1,1], scale_factor = 1, radius_factor = 1, mat_name=None ):
    
    # Set the points
    verts = []
    edges = []
    for i in range(len(points)):
        point = points[i]

        verts.append( point * scale_factor )
        if i < len(points)-1:
            edges.append( (i, i+1) )

    # Create a mesh
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, edges, [])

    # Create an object with this mesh
    obj = bpy.data.objects.new(name, mesh)
    
    # Add a subdivision surface (smooth the edges)
    subdivision = obj.modifiers.new( "subdivision", 'SUBSURF' )
    subdivision.render_levels = 1
    subdivision.levels = 1

    # Add a skin modifier in order to have thickness
    skin = obj.modifiers.new( "skin", 'SKIN' )
    skin.branch_smoothing = 0.5
    skin.use_smooth_shade = True
    
    # Smooth again with another subdivision
    subdivision = obj.modifiers.new( "subdivision2", 'SUBSURF' )
    subdivision.render_levels = 2
    subdivision.levels = 2
    
    # Associates the radius to each skin vertex
    for i in range(len(points)):
        s = obj.data.skin_vertices[''].data[i]
        
        if type(radius) != float:
            r = radius[i]
        else:
            r = radius
        rd = r * radius_factor
        s.radius = (rd, rd)

    # # Link object to the scene
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.modifier_apply(modifier='subdivision')
    bpy.ops.object.modifier_apply(modifier='skin')
    bpy.ops.object.modifier_apply(modifier='subdivision2')


    if mat_name is None:
        mat_name = name

    color_material(obj, mat_name, color)

    return obj
    
def add_curve_from_points2(name, points, radius, color=[1,1,1,1], scale_factor = 1, radius_factor = 1, mat_name=None ):

    # Create a curve with some bevel depth
    curve = bpy.data.curves.new(name=name, type='CURVE')
    curve.dimensions = '3D'
    curve.bevel_depth = 1

    # Create an object with it    
    obj = bpy.data.objects.new(name, curve)

    # Create a spline for each part
    bezier_curve = curve.splines.new('BEZIER')
    bezier_curve.bezier_points.add(len(points)-1)
    
    # Set the points
    for i in range(len(points)):
        bezier = bezier_curve.bezier_points[i]
        point = points[i]

        if type(radius) != float:
            r = radius[i]
        else:
            r = radius
        bezier.co = point * scale_factor
        bezier.radius = r * radius_factor

    # Link object to the scene
    bpy.context.scene.collection.objects.link(obj)
    # Toggle handle type (faster than doing it point by point)
    obj.select_set( True )
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.editmode_toggle()
    bpy.ops.curve.select_all(action='SELECT')
    bpy.ops.curve.handle_type_set(type='AUTOMATIC')
    bpy.ops.object.editmode_toggle()
    bpy.ops.object.convert(target='MESH')

    if mat_name is None:
        mat_name = name

    color_material(obj, mat_name, color)
    
    return obj

def add_curve(name, points, radius, color=[1,1,1,1], scale_factor = 1, radius_factor = 1,\
    curve_style='o-o-o', \
    start_radius=None, end_radius=None, node_radius=None, \
    node_gap = None, node_points=None, \
    start_color=None, end_color=None, node_color=None, mat_name=None ):
    
    def curve_add_cone(insert_index, radius, mat_name, color=None):
        
        cone_pos = points[insert_index]
        try:
            if insert_index == -1 or insert_index == len(points)-1:
                cone_dir = points[insert_index-1] - points[insert_index]
            else:
                rate = 0.5
                # look_at = points[insert_index] * rate + points[insert_index+1] * (1-rate)
                # cone_dir = points[insert_index-1] - look_at
                cone_dir = points[insert_index-1] - points[insert_index+1]
                cone_pos = points[insert_index] * rate +  (points[insert_index-1] + points[insert_index+1])/2 * (1 - rate)
        except:
            cone_dir = points[insert_index] - [0,0,0]

        cone = add_shape('cone', name + '_%d' % (insert_index), cone_pos, vec_to_euler(cone_dir), [1,1,1], color,
            {
                'radius1': radius,
                'radius2': 0,
                'depth': radius*2,
            }, mat_name=mat_name )
        return cone
    
    def curve_add_sphere(insert_index, radius, mat_name, color=None):
        sphere = add_shape('sphere', name + '_%d' % (insert_index), points[insert_index], [0]*3, [radius]*3, color, shadow=False, \
                shape_setting={'segments': 40, 'rings': 40},  mat_name=mat_name )
        return sphere
    
    def curve_add_node(insert_index, radius, anno_type, mat_name, color=None):
        if anno_type in ['>']:
            ret = curve_add_cone(insert_index, radius, mat_name, color)
        elif anno_type in ['o', 'O', '0']:
            ret = curve_add_sphere(insert_index, radius, mat_name, color)
        else:
            ret = None
        return ret

    def curve_union(curve, node, solver):
        node_copy = node.copy()
        node_copy.data = node.data.copy()
        bpy.context.scene.collection.objects.link(node_copy)
        
        curve_copy = curve.copy()
        curve_copy.data = curve.data.copy()
        bpy.context.scene.collection.objects.link(curve_copy)

        boolean_with(node, curve_copy, solver=solver)
        boolean_with(curve, node_copy, solver=solver)

        remove_object(node_copy, False)
        remove_object(curve_copy, False)

    if mat_name is None:
        mat_name = name

    all_list = []
    if '-' in curve_style:
        obj = add_curve_from_points(name, points, radius, color, scale_factor, radius_factor, mat_name=mat_name)
        all_list.append(obj)
    elif '~' in curve_style:
        # BEZIER curve
        obj = add_curve_from_points2(name, points, radius, color, scale_factor, radius_factor, mat_name=mat_name)
        all_list.append(obj)
    
    start_style = curve_style[0]
    if start_style == 'x':
        start_radius = None

    node_style = curve_style[2]
    if node_style == 'x':
        node_radius = None

    end_style = curve_style[4]
    if end_style == 'x':
        end_radius = None

    # insert node on curve
    curve_nodes = []

    node_list = []
    if node_gap:
        points_num = len(points)
        end_len = points_num
        points_list = np.array([ i for i in range(points_num) ])
        points_list = points_list[node_gap:end_len]
        node_list = points_list[::node_gap]
    elif node_points:
        for npos in node_points:
            all_dist = np.linalg.norm( points - npos, axis=-1 )
            node_list.append(np.argmin(all_dist))

    if start_radius is not None:
        if start_color is None:
            start_color = color
        start = curve_add_node(0, start_radius, start_style, mat_name + '_start', start_color)

        curve_nodes.append(start)
    
    if node_style != 'x':
        for i, insert_id in enumerate(node_list):
            if node_color is None:
                node_color = color

            node = curve_add_node(insert_id, node_radius, node_style, mat_name, node_color)
            curve_nodes.append(node)

    if end_radius is not None:
        if end_color is None:
            end_color = color
        end = curve_add_node(-1, end_radius, end_style, mat_name + '_end', end_color)

        # curve_union(obj, end, 'EXACT')
        curve_nodes.append(end)
    
    all_list += curve_nodes

    if len(all_list) > 0: obj = join_objects( all_list, name )
    else: obj = None

    if obj:
        set_visible_property(obj, False, False, False)

    return obj


def join_objects(objs, obj_name):
    '''
    Args:
        objs: list[bpy.data.objects]
        obj_name: str
    
    Returns:
        obj: bpy.data.objects
    '''
    
    c = {}
    c["object"] = c["active_object"] = objs[0]
    c["selected_objects"] = c["selected_editable_objects"] = objs
    bpy.ops.object.join(c)    
    objs[0].name = obj_name
    return objs[0]


#=============================#
#=                           =#
#=      Add model end        =#
#=                           =#
#=============================#

def remove_object(obj, remove_mat=True):
    if remove_mat:
        [ bpy.data.materials.remove(mat) for mat in obj.data.materials ]
    data = obj.data
    if data is not None:
        if type(data) == bpy.types.Curve:
            bpy.data.curves.remove(data)
        elif type(data) == bpy.types.Mesh:
            bpy.data.meshes.remove(data)
    else:
        bpy.data.objects.remove(obj)


#----------------------------------------------
# modifier begin
#----------------------------------------------

def boolean_with(obj_main, obj_with, operation_type="DIFFERENCE", solver='EXACT'):
    # obj_main.select_set(True)
    bpy.context.view_layer.objects.active = obj_main
    b = obj_main.modifiers.new('Boolean', 'BOOLEAN')
    # b.operation = 'INTERSECT'
    b.operation = operation_type
    b.object = obj_with
    b.solver = solver

    bpy.context.view_layer.objects.active = obj_main

    bpy.ops.object.modifier_apply(modifier='Boolean')

def subdivide(obj, levels=2):
    subdivision = obj.modifiers.new( "sub", 'SUBSURF' )
    subdivision.render_levels = levels
    subdivision.levels = levels
    bpy.ops.object.modifier_apply(modifier='sub')

#----------------------------------------------
# modifier end
#----------------------------------------------


def obj_border(object:bpy.context.object , width, obj_color):
    """
    Attention: really slow !!!
    """

    obj_name = object.name
    
    # obj_scale = object.scale
    # obj_rot = object.rotation_euler
    # obj_pos = object.location
    # rot_mat = eulerAnglesToRotationMatrix(obj_rot)    
    vertices = object.data.vertices

    objs = []
    for edge in object.data.edges:
        v1 = vertices[ edge.vertices[0] ].co #* obj_scale
        v2 = vertices[ edge.vertices[1] ].co #* obj_scale

        vec = np.array(v1 - v2)
        center = np.array((v1+v2)/2)
        
        lenght = np.linalg.norm(vec)

        euler = vec_to_euler(vec)
                
        obj = add_shape( 'cylinder', obj_name+"_edge", center, euler, [width, width, width + lenght/2], obj_color,  { 'vertices': 10 } )
        
        objs.append(obj)

    c = {}
    c["object"] = c["active_object"] = objs[0]
    c["selected_objects"] = c["selected_editable_objects"] = objs
    bpy.ops.object.join(c)
    
    objs[0].parent = object

def ball_pc(radius, grid_num, ball_name, pos, sphere_color, sphere_radius=0.02):

    ball = []

    alpha_list = np.linspace(0, math.pi*2, grid_num+1)
    beta_list = np.linspace(0, math.pi*2, grid_num+1)

    for alpha in alpha_list:
        for beta in beta_list:
            x = math.cos(alpha) * math.cos(beta) * radius
            y = math.sin(alpha) * math.cos(beta) * radius
            z = math.sin(beta) * radius
                
            ball.append( [ x, y, z ] )
        
    ball = np.array(ball) + pos

    pcm = PointCloudMaker()
    pcm.convert_to_spheres(points=ball, \
        name = ball_name, color=sphere_color, sphere_radius=sphere_radius)


class Voxel:
    '''
    voxel class
    
    Args:
        x: x
    
    Functions:
        x -> x
    
    '''
    def __init__(self, voxel_size, voxel_unit, voxel_center):
        self.size = voxel_size
        self.unit = voxel_unit
        self.center = voxel_center
        self.start_pos = self.center - self.unit * self.size / 2

    def get_voxel(self, i,j,k):
        voxel_leftup = self.start_pos + np.array([i,j,k]) * self.unit
        voxel_rightdown = voxel_leftup + self.unit
        return voxel_leftup, voxel_rightdown

    def inside(self, i, p, start):
        return (i*self.unit + start <= p) and ( (i+1)*self.unit + start > p)

    def find_voxel(self, p):
        return np.floor((p - self.start_pos) / self.unit).astype('int')

    def is_valid(self, i,j,k):
        return ( i < self.size ) and ( j < self.size ) and ( k < self.size )
    
    def get_pos(self, i,j,k):
        return self.start_pos + self.unit * (np.array([i,j,k]) + [0.5,0.5,0.5])

    def cut_voxel(self, voxel, cut=[20,-1,-1]):
        if cut[0] >= 0:
            voxel = voxel[cut[0],:,:]
        elif cut[1] >= 0:
            voxel = voxel[:, cut[1], :]
        if cut[2] >= 0:
            voxel = voxel[:,:,cut[2]]
        return voxel

    def divide_voxel_two(self, voxel, axis='y'):

        part_one = np.zeros( [self.size, self.size] )-1
        for i in range(self.size):
            for j in range(self.size):
                # for k in range(self.size):
                if voxel[i,j] == 0:
                    part_one[i,j] = 0
                else:
                    break
        # part_one[:,:7] = -1
        
        part_two = np.zeros( [self.size, self.size] )-1
        s = self.size - 1
        for i in range(self.size):
            for j in range(self.size):
                # for k in range(self.size):
                if voxel[s-i,s-j] == 0:
                    part_two[s-i,s-j] = 0
                else:
                    break
        return part_one, part_two

def voxel_data(data_folder, name, voxel, data_sign):
    def sign(a):
        if a < 0 : return -1
        if a == 0 : return 0
        if a > 0 : return 1
    
    data = np.loadtxt( os.path.join(data_folder, '%s.txt' % name ) )
    pos = data[:,:3]
    obj = data[:,3]
    hand = data[:,4]
    time = data[:,-1]

    print(name, " 1: ", np.sum(time == 1) )
    print(name, " 2: ", np.sum(time == 2) )
    print(name, " x: ", np.sum(time > 2) )

    voxel_dist = np.zeros( [voxel.size, voxel.size, voxel.size] )
    voxel_sign = np.zeros( [voxel.size, voxel.size, voxel.size] )
    voxel_time = np.zeros( [voxel.size, voxel.size, voxel.size] )

    # start 
    for pi, p in enumerate(pos):
        i,j,k = voxel.find_voxel(p)
        voxel_dist[i,j,k] = hand[pi] * data_sign
        voxel_sign[i,j,k] = sign( hand[pi] ) * data_sign
        voxel_time[i,j,k] = time[pi]
    return voxel_dist, voxel_sign, voxel_time

def voxel_cubes(positions, obj_name, obj_size, obj_color):
    obj_list = []
    for i, pos in enumerate(positions):
        obj = add_cube(obj_name+"_%d" % i, pos, [0,0,0], [obj_size, obj_size, obj_size], obj_color)
        obj_border(obj, 0.01, [1,1,1,1])
        obj_list.append(obj)
    return obj_list

def object_bbox( obj ):
    # min_x = 1000
    # max_x = -1000
    # min_y = 1000
    # max_y = -1000
    # min_z = 1000
    # max_z = -1000

    # lowest_pt = min([(obj.matrix_world @ v.co).z for v in obj.data.vertices])

    v = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
    v = np.array(v)

    min_x = min(v[:,0])
    min_y = min(v[:,1])
    min_z = min(v[:,2])
    
    max_x = max(v[:,0])
    max_y = max(v[:,1])
    max_z = max(v[:,2])

    # for vertex in object.data.vertices:
    #     x, y, z = np.array(vertex.co)
    #     if min_x > x: min_x = x
    #     if max_x < x: max_x = x

    #     if min_y > y: min_y = y
    #     if max_y < y: max_y = y

    #     if min_z > z: min_z = z
    #     if max_z < z: max_z = z
    return min_x, max_x, min_y, max_y, min_z, max_z


