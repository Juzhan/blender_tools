import bpy

def insert_keys(obj, frame_list, value_list, data_path, index=-1):
    '''
    Args:
        obj: bpy.data.objects[xxx]
        frame_list: list[int], the key frame you want to insert
        value_list: list, the value you want to set in each frame
        data_path: str
            the path of obj porperty
            example: we want to change the 'obj.rotation_euler', so the data_path is 'rotation_euler'
        index: int
            array index of the property to key. 
            Defaults to -1 which will key all indices or a single channel if the property is not an array.
    '''
    
    assert len(frame_list) == len(value_list), "the len of frame_list and value_list must be the same"

    for i in range(len(frame_list)):
        key_frame = frame_list[i]
        value = value_list[i]
        
        if index != -1:
            current_value = getattr(obj, data_path)
            current_value[index] = value
            setattr(obj, data_path, current_value)
            
        else:
            setattr(obj, data_path, value)
        
        obj.keyframe_insert(data_path=data_path, index=index, frame=key_frame)


