import bpy

def select_object(obj):
    obj.select_set( True )
    bpy.context.view_layer.objects.active = obj


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
