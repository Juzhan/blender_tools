import bpy
from inspect import getcallargs


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
