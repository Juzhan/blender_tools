import math
import numpy as np
    
import bpy
import bmesh

from scipy.spatial.transform import Rotation
import trimesh

# dir = os.path.dirname(bpy.data.filepath)
# if not dir in sys.path:
#     sys.path.append(dir)

from tools.material import *
from tools.transform import *
# 
def set_visible_property(object, diffuse=True, glossy=True, shadow=True):
    if bpy.app.version[0] >= 3:
        object.visible_diffuse = diffuse
        object.visible_glossy = glossy
        object.visible_shadow = shadow
    else:    
        object.cycles_visibility.diffuse = diffuse
        object.cycles_visibility.glossy = glossy
        object.cycles_visibility.shadow = shadow

def read_verts(mesh):
    mverts_co = np.zeros((len(mesh.vertices)*3), dtype=np.float)
    mesh.vertices.foreach_get("co", mverts_co)
    return np.reshape(mverts_co, (len(mesh.vertices), 3))      

def read_edges(mesh):
    fastedges = np.zeros((len(mesh.edges)*2), dtype=np.int) # [0.0, 0.0] * len(mesh.edges)
    mesh.edges.foreach_get("vertices", fastedges)
    return np.reshape(fastedges, (len(mesh.edges), 2))

def read_norms(mesh):
    mverts_no = np.zeros((len(mesh.vertices)*3), dtype=np.float)
    mesh.vertices.foreach_get("normal", mverts_no)
    return np.reshape(mverts_no, (len(mesh.vertices), 3))

def smooth(obj):
    mesh = obj.data
    mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    obj.data.use_auto_smooth = True


def edit_model(obj):
    obj.select_set( True )
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.editmode_toggle()
    # bpy.ops.object.select_all(action='SELECT')
    # bpy.ops.mesh.remove_doubles()

    # do something here

    bpy.ops.object.editmode_toggle()

# https://github.com/itsumu/point_cloud_renderer
# https://b3d.interplanety.org/en/how-to-create-a-new-mesh-uv-with-the-blender-python-api/
def add_pointcloud(filename, points, name, position, color, radius, scale=1, shadow=True ):
    '''
    Args:
        filename: str
            set to None if points is not None
        points: list[float], [nx3]
            set to None if filename is not None
        name: str
            name of the pointcloud object, just a name
        position: list[float], [3]
            position of the hole pointcloud object
        color: list[float], [4] / [nx4]
            number between [0,1], rgba of the pointcloud / rgba of each point
        radius: float
            radius of the sphere at points
        scale: float
            scale of the hole pointcloud object

    Returns:
        object: bpy.data.object
    '''
    
    if filename:
        pc = trimesh.load_mesh(filename)
        points = np.array(pc.vertices)
    
    points *= scale
    points += np.array(position)

    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=radius)
    tmp_obj = bpy.context.active_object
    tmp_obj.name = name + "_tmp"
    tmp_obj.data.name = name + "_tmp"
    tmp_obj.location = [0,0,0]
    tmp_obj.rotation_euler = [0,0,0]

    bpy.ops.mesh.primitive_plane_add()
    instancer = bpy.context.active_object
    instancer.name = name + "_point_cloud"
    instancer_mesh = instancer.data
    instancer_mesh.name = instancer.name
    instancer.location = [0,0,0]
    instancer.rotation_euler = [0,0,0]

    # add face for each vertices
    scale = radius / 2
    
    img_w = int(np.ceil(np.sqrt(len(points))) + 1)
    bm = bmesh.new()

    offsets = [
        [1, 1, 0],
        [1, -1, 0],
        [-1, -1, 0],
        [-1, 1, 0],
    ]
    offsets = [
        [0, 1, 0],
        [math.sqrt(3)/2, -0.5, 0],
        [-math.sqrt(3)/2, -0.5, 0],
    ]

    # new rectangular
    for i in range(points.shape[0]):
        verts = []
        for j in range(len(offsets)):
            vert = bm.verts.new()
            vert.co = points[i, :] + np.array(offsets[j]) * scale
            verts.append(vert)
        bm.faces.new(verts)

    bm.to_mesh(instancer_mesh)

    # instacer for point cloud
    tmp_obj.parent = instancer
    instancer.instance_type = 'FACES'

    color_mode = 'rgba'

    if len(color) > 4:

        # add vertex color
        vertex_color_name = instancer.name + '_col'
        if not instancer_mesh.vertex_colors:
            instancer_mesh.vertex_colors.new(name=vertex_color_name)
        loops = instancer_mesh.loops

        for loop_i, loop in enumerate(loops):
            instancer_mesh.vertex_colors[vertex_color_name].data[loop_i].color = color[int(loop_i/3)]

        # add texture
        mat = vertex_rgba_material(instancer.name, vertex_color_name)
        set_object_mat(instancer, mat)

        nodes = mat.node_tree.nodes
        image_node = nodes.new('ShaderNodeTexImage')
        img = bpy.data.images.new( vertex_color_name, img_w, img_w )
        image_node.image = img

        # set render 
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'

        # unwrap UV of the mesh
        # each face use 1 pixel
        uv_name = vertex_color_name
        new_uv = instancer.data.uv_layers.new(name=uv_name)
        for loop in bpy.context.active_object.data.loops:
            loop_i = loop.index
            point_index = int(loop_i/3)
            x = (int(point_index / img_w) + 0.5) / (img_w * 1.0)
            y = (point_index % img_w + 0.5) / (img_w * 1.0)
            new_uv.data[loop_i].uv = ( y, x )

        colors = list(np.zeros( img_w * img_w * 4 ))
        colors[ :len(color)*4 ] = color.flatten()
        img.pixels.foreach_set(colors)

        # # add texture for tmp_obj

        if color_mode == 'rgba':
            mat = vertex_rgba_material(tmp_obj.name, vertex_color_name)
        else:
            mat = vertex_rgb_material(tmp_obj.name, vertex_color_name)
        
        set_object_mat(tmp_obj, mat)

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        image_node = nodes.new('ShaderNodeTexImage')
        image_node.image = img
        
        uv_node = nodes.new('ShaderNodeTexCoord')
        links.new( uv_node.outputs['UV'], image_node.inputs['Vector'] )
        if color_mode == 'rgba':
            diffuse_node = nodes['Principled BSDF']
            links.new( image_node.outputs['Color'], diffuse_node.inputs['Base Color'] )
        else:
            diffuse_node = nodes['Diffuse BSDF']
            links.new( image_node.outputs['Color'], diffuse_node.inputs['Color'] )
        uv_node.from_instancer = True
    else:
        mat = rgba_material(tmp_obj.name, color)
        set_object_mat(tmp_obj, mat)

    set_visible_property(instancer, shadow=shadow)

    bm.free()

    return instancer

#=============================#
#=                           =#
#=        Add model          =#
#=                           =#
#=============================#

def add_model_by_vertex(vertices, edges, faces, name):
    new_mesh = bpy.data.meshes.new('new_mesh')
    new_mesh.from_pydata(vertices, edges, faces)
    # must set calc_edges=True, or it would crash when the object is in edit mode in windows
    new_mesh.update(calc_edges=True)
    # new_mesh.validate(verbose=True)
    new_mesh.name = name
    # make object from mesh
    new_object = bpy.data.objects.new(name, new_mesh)
    bpy.context.collection.objects.link(new_object)    
    return new_object

def add_model( filename, name, position, rotation, color, scale=1, \
        texture_path=None, normal_path=None, \
        diffuse=False, glossy=False, shadow=True, use_auto_smooth=False, mat_name=None ):
    '''
    Args:
        filename: str
        name: str
        position: list[float], [3]
        rotation: list[float], [3]
        color: list[float], [4] / [nx4]
        scale: float
    
    Returns:
        object: bpy.data.object
    '''

    if filename.endswith('off'):
        mesh = trimesh.load_mesh(filename)
        object = add_model_by_vertex(mesh.vertices, mesh.edges, mesh.faces, name)
        object.select_set(True)
        bpy.context.view_layer.objects.active = object
    else:
        if filename.endswith('obj'):
            if bpy.app.version[0] >= 4:
                bpy.ops.wm.obj_import( filepath=filename )
            else:
                bpy.ops.import_scene.obj( filepath=filename )
        elif filename.endswith('ply'):
            if bpy.app.version[0] >= 4:
                bpy.ops.wm.ply_import( filepath=filename )
            else:
                bpy.ops.import_mesh.ply( filepath=filename )
        elif filename.endswith('stl'):
            bpy.ops.import_mesh.stl( filepath=filename )
            
        object = bpy.context.selected_objects[0]
    object.name = name
    object.rotation_mode = 'XYZ'
    object.rotation_euler = [0, 0, 0]
    object.scale = [scale, scale, scale]

    mesh = object.data
    if mesh is not None:
        mesh.name = name

    if rotation is not None:
        if len(rotation) == 3:
            object.rotation_mode = 'XYZ'
            object.rotation_euler = rotation
            object.location = position
        elif len(rotation) == 4:
            object.rotation_mode = 'QUATERNION'
            object.rotation_quaternion = rotation
            object.location = position

    set_visible_property(object, diffuse, glossy, shadow)

    object.data.use_auto_smooth = use_auto_smooth

    if mat_name is None:
        mat_name = name

    mat = None
    if texture_path is not None:
        mat = image_material(mat_name, texture_path, normal_path)
    elif color is not None:
        if len(color) == 4:
            mat = rgba_material(mat_name, color)
        else:            
            set_vertex_color(object, colors, f'{name}_vert')
            mat = vertex_rgba_material(f'{name}_vert_mat', f'{name}_vert')

    if mat is not None:
        set_object_mat(object, mat)

    return object

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

    mat = None
    if texture_path is not None:
        mat = image_material(mat_name, texture_path, normal_path)
    elif obj_color is not None:
        if len(obj_color) == 4:
            mat = rgba_material(mat_name, obj_color)
        else:
            set_vertex_color(object, obj_color, f'{obj_name}_vert')
            mat = vertex_rgba_material(f'{obj_name}_vert_mat', f'{obj_name}_vert')

    if mat is not None:
        set_object_mat(object, mat)

    return object



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

    mat = rgba_material(mat_name, color)
    set_object_mat(obj, mat)

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

    mat = rgba_material(mat_name, color)
    set_object_mat(obj, mat)

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

    points = np.array(points)

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

    if bpy.app.version[0] >= 4:
        with bpy.context.temp_override( object = c['object'], \
                                       active_object = c['active_object'], \
                                       selected_objects = c['selected_objects'],\
                                        selected_editable_objects = c['selected_editable_objects'] ):
            bpy.ops.object.join()
    else:
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
