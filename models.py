import os
import sys
import copy
import math
import numpy as np
    
import bpy
import bmesh

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)

from material import *
from transform import *
# 


# https://github.com/itsumu/point_cloud_renderer
class PointCloudMaker():
    def __init__(self, type='ico_sphere'):
        self.type = type
        self.active_object = bpy.context.active_object
        bpy.app.debug_value = 0
        self.instancers = []

    # Utility function to generate numpy points from ascii point cloud files
    def generate_points_from_pts(self, filename):
        points = np.loadtxt(filename)
        return points

    def load_points(self, filename):
        # import open3d as o3d
        import trimesh
        pc = trimesh.load_mesh(filename)
        # pc = o3d.io.read_point_cloud(filename)
        # return np.array(pc.points)
        return np.array(pc.vertices)

    # Convert points to spheres
    def convert_to_spheres(self, points=None, name='pc', color=[1,0,0,1], sphere_radius=0.01):
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
        
        template_sphere.cycles_visibility.diffuse = False
        template_sphere.cycles_visibility.glossy = False
        template_sphere.cycles_visibility.shadow = False

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
def reset(pcm=None, clear_instancers=False, clear_database=False):
    # start_time = time.time()

    if (clear_instancers):
        # Clear materials
        for material in bpy.data.materials:
            if ('Material' in material.name) or ('material' in material.name):
                bpy.data.materials.remove(material)

    if pcm != None:
        pcm.clear_instancers()
    if clear_database:
        # Clear meshes
        for mesh in bpy.data.meshes:
            if (not 'Cube' in mesh.name) and (not 'Cone' in mesh.name) \
                    and (not 'Cylinder' in mesh.name):
                bpy.data.meshes.remove(mesh)

        # Clear images
        for image in bpy.data.images:
            bpy.data.images.remove(image)
    # print('reset time: ', time.time() - start_time)


#=============================#
#=                           =#
#=        Add model          =#
#=                           =#
#=============================#

def add_model( filename, obj_name, obj_color, obj_pos, obj_rot, \
        texture_path=None, normal_path=None, \
        diffuse=False, glossy=False, shadow=True, use_auto_smooth=False ):
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
    
    name = filename.split('/')[-1].split('\\')[-1].split('.')[0][:10]

    mesh = None
    for obj in bpy.data.meshes:
        if name in obj.name[:10]:
            mesh = obj
            break

    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            if name == obj.name[:10]:
                object = obj
                break

    if mesh is not None:
        mesh.name = obj_name
    object.name = obj_name
    object.rotation_mode = 'XYZ'
    object.rotation_euler = [0, 0, 0]

    if obj_rot is not None:
        if len(obj_rot) == 3:
            object.rotation_mode = 'XYZ'
            object.rotation_euler = obj_rot
            object.location = obj_pos
        elif len(obj_rot) == 4:
            object.rotation_mode = 'QUATERNION'
            object.rotation_quaternion = obj_rot
            object.location = obj_pos
    
    object.cycles_visibility.diffuse = diffuse
    object.cycles_visibility.glossy = glossy
    object.cycles_visibility.shadow = shadow

    object.data.use_auto_smooth = use_auto_smooth

    if texture_path is not None:
        image_material(object, obj_name, obj_color, texture_path, normal_path)
    else:
        color_material(object, obj_name, obj_color)
    return object

def add_pointcloud( filename, obj_name, obj_pos, obj_scale, sphere_color, sphere_radius, pc_ids=[] ):
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
    pcm = PointCloudMaker()
    if filename.endswith('obj'):
        pcd = pcm.load_points(filename)
    elif filename.endswith('ply'):
        pcd = pcm.load_points(filename)
    else:
        pcd = pcm.generate_points_from_pts(filename)
    
    pcd[:,0] *= obj_scale[0]
    pcd[:,1] *= obj_scale[1]
    pcd[:,2] *= obj_scale[2]

    pcd += np.array(obj_pos)
    if len(pc_ids) > 0:
        pcd = pcd[pc_ids]
    
    # Create spheres from point clouds
    pcm.convert_to_spheres(points=pcd, name = obj_name, color=sphere_color, sphere_radius=sphere_radius)

def add_shape(obj_type, obj_name, obj_pos, obj_rot, obj_size, obj_color, \
        shape_setting=None, \
        texture_path=None, normal_path=None, \
        diffuse=False, glossy=False, shadow=True ):
    '''
    Args:
        obj_type: str
            >>> 'cube'
            >>> 'sphere'
            >>> 'cone'
            >>> 'plane'
            >>> 'cylinder'

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
    elif obj_type == 'plane':
        bpy.ops.mesh.primitive_plane_add()
    elif obj_type == 'cube':
        bpy.ops.mesh.primitive_cube_add()
    elif obj_type == 'sphere':
        bpy.ops.mesh.primitive_uv_sphere_add()
    elif obj_type == 'cone':
        bpy.ops.mesh.primitive_cone_add(radius1=shape_setting['radius1'], radius2=shape_setting['radius2'], depth=shape_setting['depth'])

    object = bpy.context.active_object
    object.name = obj_name
    object.scale = obj_size
    object.rotation_mode = 'XYZ'
    object.rotation_euler= obj_rot
    object.location= obj_pos
    
    object.cycles_visibility.diffuse = diffuse
    object.cycles_visibility.glossy = glossy
    object.cycles_visibility.shadow = shadow

    if texture_path is not None:
        image_material(object, obj_name, obj_color, texture_path, normal_path)
    else:
        color_material(object, obj_name, obj_color)
    return object

def add_line(start, end, width, color, with_arrow, name='arrow'):
    start = np.array(start)
    end = np.array(end)
    cam_dir = end - start
    length = np.linalg.norm(end-start)
    
    if with_arrow:
        cone = add_shape('cone', 'arrow_cone_' + name, end, vec_to_euler( -cam_dir ), [1,1,1], color,
        {
            'radius1': width,
            'radius2': 0,
            'depth': width*2,
        }
         )

        # bpy.ops.mesh.primitive_cone_add(radius1=width, radius2=0, depth=width*2, \
        # enter_editmode=False, align='WORLD', location=end, )
        # cone = bpy.context.active_object
        # cone.rotation_mode = 'XYZ'
        # cone.rotation_euler = vec_to_euler( -cam_dir )
        # cone.cycles_visibility.shadow = False
        # cone.name = 'arrow_cone_' + name

        # if len(color) == 3:
        #     pure_color_material(cone, name, color )
        # else:
        #     color_material(cone, name, color )
    
    cylinder = add_shape( 'cylinder', 'arrow_cylinder_' + name, (end+start)/2, vec_to_euler(cam_dir), [1,1,1], color, 
    {
        'vertices': 10
    }
      )
    # bpy.ops.mesh.primitive_cylinder_add(radius=width/3, depth=length, \
    #     enter_editmode=False, align='WORLD', location=(end+start)/2, )
    # cylinder = bpy.context.active_object

    # cylinder.rotation_mode = 'XYZ'
    # cylinder.rotation_euler = vec_to_euler(cam_dir)
    # cylinder.cycles_visibility.shadow = False
    # cylinder.name = 'arrow_cylinder_' + name

    # if len(color) == 3:
    #     pure_color_material(cylinder, name, color )
    # else:
    #     color_material(cylinder, name, color )

    if with_arrow:
        return [cylinder, cone]
    else:
        return cylinder

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
                
        obj = add_cylinder(obj_name+"_edge", center, euler, [width, width, width + lenght/2], obj_color)
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


# def color_obj_by_part(mesh_labels):
#     for label in labels:
#         mesh_color = hex_to_rgba(colors[label])
#         # add materials
#         object.data.materials.append( new_color_material( 'ibs_part_%d' % label, mesh_color, shadow_mode='NONE' ) ) 

#     for polygon_idx, polygon in enumerate(object.data.polygons):
#         polygon.material_index = mesh_labels[ polygon_idx ]

