import bpy

def select_object(obj):
    obj.select_set( True )
    bpy.context.view_layer.objects.active = obj

#----------------------------------------------
# Modifier
#----------------------------------------------

def boolean_with(obj_main, obj_with, operation_type="DIFFERENCE", solver='EXACT', apply_mod=False):
    # obj_main.select_set(True)
    bpy.context.view_layer.objects.active = obj_main
    b = obj_main.modifiers.new('Boolean', 'BOOLEAN')
    # b.operation = 'INTERSECT'
    b.operation = operation_type
    b.object = obj_with
    b.solver = solver
    bpy.context.view_layer.objects.active = obj_main

    if apply_mod:
        bpy.ops.object.modifier_apply(modifier='Boolean')

def subdivide(obj, levels=2, apply_mod=False):
    subdivision = obj.modifiers.new( "sub", 'SUBSURF' )
    subdivision.render_levels = levels
    subdivision.levels = levels
    if apply_mod:
        bpy.ops.object.modifier_apply(modifier='sub')

def smooth(obj, use_auto_smooth=False):
    # mesh = obj.data
    # mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
    if use_auto_smooth:
        obj.data.use_auto_smooth = True
    else:
        obj.data.shade_smooth()


def wireframe(obj, wire_thickness=0.04, only_wire=False, offset=0):
    if obj.name in obj.modifiers:
        wire = obj.modifiers[obj.name]
    else:
        wire = obj.modifiers.new( obj.name, 'WIREFRAME' )
    wire.thickness = wire_thickness
    wire.material_offset = offset
    wire.use_replace = only_wire
    return wire

def recalculate_normal(obj):
    obj.select_set( True )
    bpy.context.view_layer.objects.active = obj

    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.mesh.select_all(action='SELECT')

    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.mesh.set_normals_from_faces()

    bpy.ops.object.editmode_toggle()


def clean_double_faces(obj, just_remove_doubles=False):
    
    obj.select_set( True )
    bpy.context.view_layer.objects.active = obj
    
    if not just_remove_doubles:
        modifier_name = obj.name + '_mod'
        if obj.name in obj.modifiers:
            mod = obj.modifiers[modifier_name]
        else:
            mod = obj.modifiers.new( modifier_name, 'EDGE_SPLIT' )
        bpy.ops.object.modifier_apply(modifier= modifier_name)

    # merge meshes
    bpy.ops.object.mode_set(mode = 'EDIT') 
    bpy.ops.mesh.select_mode(type="FACE")
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles()
    bpy.ops.object.mode_set(mode = 'OBJECT') 


def mark_freestyle_edge(obj):
    # special use for render_with_lines('wireframe')
    obj.select_set( True )
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_mode(type='EDGE')

    bpy.ops.mesh.mark_freestyle_edge(clear=False)
    # bpy.ops.object.mode_set(mode = 'OBJECT') 
    bpy.ops.object.editmode_toggle()



def render_with_lines( line_type, line_thickness=2, line_color=[1,0,0,1], target_collection=None, target_obj=None):
    '''
    Args:
        line_type: str
            1. wireframe: wireframe of object surface, but you have to use *set_edge_line_mask()* first
            2. all_border: total border of the objects in scene
            3. each_border: border of each object
            4. intersection: intersection lines between objects 
        line_thickness: int
        line_color: list[4]
    '''

    bpy.ops.object.gpencil_add(align='WORLD', location=(0, 0, 0), scale=(1, 1, 1), type='LRT_SCENE')
    obj = bpy.context.selected_objects[0]

    mod = obj.grease_pencil_modifiers["Line Art"]

    if target_obj is not None:
        mod.source_type = 'OBJECT'
        mod.source_object = target_obj
    elif target_collection is not None:
        mod.source_type = 'COLLECTION'
        mod.source_collection = target_collection
    else:
        mod.source_type = 'SCENE'

    # clean setting
    mod.use_crease = False
    mod.use_intersection = False
    mod.use_material = False
    mod.use_edge_mark = False
    mod.use_loose = False
    mod.use_contour = False

    if line_type == 'wireframe':
        mod.use_edge_mark = True
    
    if line_type == 'all_border':
        mod.silhouette_filtering = 'GROUP'
        mod.use_contour = True
    elif line_type == 'each_border':
        mod.silhouette_filtering = 'INDIVIDUAL'
        mod.use_contour = True
    
    if line_type == 'intersection':
        mod.use_intersection = True

    # mod.silhouette_filtering = 'NONE'
    # mod.use_invert_silhouette = False

    mod.thickness = line_thickness
    obj.data.materials[0].grease_pencil.color = line_color



#----------------------------------------------
# Geometry nodes only support 3.0+
#----------------------------------------------

def show_intersection_lines(object_list, line_name, line_thickness=0.0015, line_color=[1,0,0,1] ):
    #  show the intersection lines between two objects
    # or you can use another function:
    #     render_with_lines('intersection')
    
    from tools import material

    bpy.ops.mesh.primitive_plane_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))

    obj = bpy.context.active_object
    obj.name = line_name
    

    mat = material.rgba_material(line_name, line_color)
    material.set_object_mat(obj, mat)

    mod_name = line_name + '_geo'

    # if mod_name in obj.modifiers:
    #     mod = obj.modifiers[mod_name]
    # else:
    #     mod = obj.modifiers.new( mod_name, 'NODES' )

    # bpy.ops.object.modifier_add(type='NODES')
    bpy.ops.node.new_geometry_nodes_modifier()

    mod = obj.modifiers[-1]
    mod.name = mod_name

    tree = mod.node_group
    
    nodes = tree.nodes
    nodes.clear()

    OUT = nodes.new('NodeGroupOutput')

    mesh_boolean_node = nodes.new('GeometryNodeMeshBoolean')
    mesh_2_curve_node = nodes.new('GeometryNodeMeshToCurve')
    try:
        bpy.data.node_groups["Geometry Nodes.021"].nodes["Group"].inputs[1].default_value

        smooth_curve_node = nodes.new('GeometryNodeSmoothHairCurves')
    except:
        smooth_curve_node = None
    curve_circle_node = nodes.new('GeometryNodeCurvePrimitiveCircle')
    curve_2_mesh_node = nodes.new('GeometryNodeCurveToMesh')
    
    set_mat_node = nodes.new('GeometryNodeSetMaterial')


    mesh_boolean_node.operation = 'INTERSECT'
    curve_circle_node.inputs[0].default_value = 5
    curve_circle_node.inputs[4].default_value = line_thickness

    set_mat_node.inputs[2].default_value = mat

    object_nodes = []

    for union_obj in object_list:
        obj_node = nodes.new('GeometryNodeObjectInfo')
        obj_node.inputs[0].default_value = union_obj
        obj_node.transform_space = 'RELATIVE'
        object_nodes.append(obj_node)
        
        tree.links.new(obj_node.outputs['Geometry'], mesh_boolean_node.inputs[1])

    tree.links.new(mesh_boolean_node.outputs['Mesh'], mesh_2_curve_node.inputs['Mesh'])
    tree.links.new(mesh_boolean_node.outputs['Intersecting Edges'], mesh_2_curve_node.inputs['Selection'])

    if smooth_curve_node is not None:
        smooth_curve_node.inputs[6].default_value = False
        tree.links.new(mesh_2_curve_node.outputs['Curve'], smooth_curve_node.inputs['Geometry'])
        tree.links.new(smooth_curve_node.outputs['Geometry'], curve_2_mesh_node.inputs['Curve'])
    else:
        tree.links.new(mesh_2_curve_node.outputs['Curve'], curve_2_mesh_node.inputs['Curve'])

    tree.links.new(curve_circle_node.outputs['Curve'], curve_2_mesh_node.inputs['Profile Curve'])
    tree.links.new(curve_2_mesh_node.outputs['Mesh'], set_mat_node.inputs['Geometry'])
    tree.links.new(curve_2_mesh_node.outputs['Mesh'], set_mat_node.inputs['Geometry'])

    tree.links.new(set_mat_node.outputs['Geometry'], OUT.inputs['Geometry'])

    for node in nodes:
        node.location[1] = 0

    OUT.location[0] = 600
    set_mat_node.location[0] = 400
    curve_2_mesh_node.location[0] = 200
    mesh_2_curve_node.location[0] = 0
    curve_circle_node.location[0] = 0
    curve_circle_node.location[1] = -130
    mesh_boolean_node.location[0] = -200

    for oi, obj_node in enumerate(object_nodes):
        obj_node.location[0] = -400
        obj_node.location[1] = oi * -220


def curve_to_line(curve_obj, line_name, line_thickness=0.0015, line_color=[1,0,0,1] ):
    #  visualize edge / curve to line mesh

    bpy.ops.mesh.primitive_plane_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))

    obj = bpy.context.active_object
    obj.name = line_name
    
    try:
        from tools import material
        mat = material.rgba_material(line_name, line_color)
        material.set_object_mat(obj, mat)
    except:
        mat = None
    
    mod_name = line_name + '_geo'

    bpy.ops.node.new_geometry_nodes_modifier()

    mod = obj.modifiers[-1]
    mod.name = mod_name

    tree = mod.node_group
    
    nodes = tree.nodes
    nodes.clear()

    OUT = nodes.new('NodeGroupOutput')
    # IN = nodes.new('NodeGroupInput')

    mesh_2_curve_node = nodes.new('GeometryNodeMeshToCurve')
    try:
        bpy.data.node_groups["Geometry Nodes.021"].nodes["Group"].inputs[1].default_value

        smooth_curve_node = nodes.new('GeometryNodeSmoothHairCurves')
    except:
        smooth_curve_node = None
    curve_circle_node = nodes.new('GeometryNodeCurvePrimitiveCircle')
    curve_2_mesh_node = nodes.new('GeometryNodeCurveToMesh')
    
    set_mat_node = nodes.new('GeometryNodeSetMaterial')


    curve_circle_node.inputs[0].default_value = 5
    curve_circle_node.inputs[4].default_value = line_thickness

    set_mat_node.inputs[2].default_value = mat

    obj_node = nodes.new('GeometryNodeObjectInfo')
    obj_node.inputs[0].default_value = curve_obj
    obj_node.transform_space = 'RELATIVE'
    

    if curve_obj.type == 'CURVE':
        tree.links.new(obj_node.outputs['Geometry'], curve_2_mesh_node.inputs['Curve'])
    else:
        tree.links.new(obj_node.outputs['Geometry'], mesh_2_curve_node.inputs['Mesh'])

        if smooth_curve_node is not None:
            smooth_curve_node.inputs[6].default_value = False
            tree.links.new(mesh_2_curve_node.outputs['Curve'], smooth_curve_node.inputs['Geometry'])
            tree.links.new(smooth_curve_node.outputs['Geometry'], curve_2_mesh_node.inputs['Curve'])
        else:
            tree.links.new(mesh_2_curve_node.outputs['Curve'], curve_2_mesh_node.inputs['Curve'])

    tree.links.new(curve_circle_node.outputs['Curve'], curve_2_mesh_node.inputs['Profile Curve'])
    tree.links.new(curve_2_mesh_node.outputs['Mesh'], set_mat_node.inputs['Geometry'])
    tree.links.new(curve_2_mesh_node.outputs['Mesh'], set_mat_node.inputs['Geometry'])

    tree.links.new(set_mat_node.outputs['Geometry'], OUT.inputs['Geometry'])

    for node in nodes:
        node.location[1] = 0

    OUT.location[0] = 600
    set_mat_node.location[0] = 400
    curve_2_mesh_node.location[0] = 200
    mesh_2_curve_node.location[0] = 0
    curve_circle_node.location[0] = 0
    curve_circle_node.location[1] = -130

    obj_node.location[0] = -400
    obj_node.location[1] = 0


