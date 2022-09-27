import copy
import math
import numpy as np

# some bugs, please use scipy.spatial.transform for correct results

#============----------------   vec / euler   ----------------============#

def vec_cos_theta(vec_a, vec_b):
    """
    Args:
        vec_a: list/np.array [3]
        vec_b: list/np.array [3]
    Returns:
        cos_theta: float
    """
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)
    
    len_a = np.linalg.norm(vec_a)
    len_b = np.linalg.norm(vec_b)
    return (np.dot(vec_a, vec_b) / ( len_a * len_b ))

def vec_euler_to_matrix( vec, theta ):
    """
    Args:
        vec: list/np.array [3]
        theta: float
    Returns:
        rot: np.array [3, 3]
    """
    rot = np.zeros((3,3))
    x, y, z = vec

    xx = x**2
    yy = y**2
    zz = z**2

    xy = x*y
    xz = x*z
    
    yz = z*y

    cost = np.math.cos(theta)    
    sint = np.math.sin(theta)    

    rot[0,0] = xx*(1-cost) + cost
    rot[0,1] = xy*(1-cost) + z*sint
    rot[0,2] = xz*(1-cost) - y*sint

    rot[1,0] = xy*(1-cost) - z*sint
    rot[1,1] = yy*(1-cost) + cost
    rot[1,2] = yz*(1-cost) + x*sint

    rot[2,0] = xz*(1-cost) + y*sint
    rot[2,1] = yz*(1-cost) - x*sint
    rot[2,2] = zz*(1-cost) + cost
    return rot

def vec_to_euler(vec):
    '''
    
    Args:
        vec: list | np.array
    
    Returns:
        xyz_euler: list
    '''
    
    y_cam_dir = copy.deepcopy(vec)
    y_cam_dir[1] = 0

    x_cam_dir = copy.deepcopy(vec)
    x_cam_dir[0] = 0

    z_cam_dir = copy.deepcopy(vec)
    z_cam_dir[2] = 0

    x_axis = np.array([1,0,0])
    y_axis = np.array([0,1,0])
    z_axis = np.array([0,0,1])

    x_rot_degree = math.acos(vec_cos_theta( vec, -z_axis ))

    z_rot_degree = math.acos(vec_cos_theta( z_cam_dir, y_axis ))
    if vec[0] > 0:
        z_rot_degree = -z_rot_degree
    
    if np.isnan(z_rot_degree):
        z_rot_degree = 0

    return [ x_rot_degree, 0, z_rot_degree ]

def euler_to_matrix(euler) :
    '''
    Args:
        euler: list[float]
    
    Returns:
        matrix: [4 x 4]
    '''
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(euler[0]), -math.sin(euler[0]) ],
                    [0,         math.sin(euler[0]), math.cos(euler[0])  ]
                    ])

    R_y = np.array([[math.cos(euler[1]),    0,      math.sin(euler[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(euler[1]),   0,      math.cos(euler[1])  ]
                    ])

    R_z = np.array([[math.cos(euler[2]),    -math.sin(euler[2]),    0],
                    [math.sin(euler[2]),    math.cos(euler[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

#============----------------   matrix   ----------------============#

def matrix_to_quaternion(mat):
    q = np.zeros(4)
    tr = mat[0,0] + mat[1,1] + mat[2,2]

    w = np.sqrt(tr+1) / 2
    q[0] = (mat[2,1] - mat[1,2]) / (4 * w)
    q[1] = (mat[0,2] - mat[2,0]) / (4 * w)
    q[2] = (mat[1,0] - mat[0,1]) / (4 * w)
    q[3] = w
    return q


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def matrix_to_euler(R) :
    '''
    Args:
        R: np.array [4 x 4]
    
    Returns:
        euler: np.array [3]
    '''
    
    # assert(is_rotation_matrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

# Checks if a matrix is a valid rotation matrix.
def is_rotation_matrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

#============----------------   quaternion   ----------------============#

def quaternion_multi(q, r):
    qv = q[:3]
    qw = q[-1]

    rv = r[:3]
    rw = r[-1]

    res = q.copy()
    res[:3] = np.cross( qv, rv ) + rw * qv + qw * rv
    res[-1] = qw * rw - np.dot(qv, rv)

    return res

def quaternion_inv(q):
    q[:3] = -q[:3]
    q = q / (np.linalg.norm(q))**2
    return q

def quaternion_to_matrix(q):
    x, y, z, w = q
    mat = np.identity(3)

    xx = x**2
    yy = y**2
    zz = z**2

    xy = x*y
    xz = x*z
    xw = x*w
    
    yz = z*y
    yw = w*y

    zw = w*z

    mat[0,0] = 1 - 2 * (yy+zz)
    mat[0,1] = 2 * (xy - zw)
    mat[0,2] = 2 * (xz + yw)
    
    mat[1,0] = 2 * (xy + zw)
    mat[1,1] = 1 - 2 * ( xx + zz)
    mat[1,2] = 2 * (yz - xw)
    
    mat[2,0] = 2 * (xz - yw)
    mat[2,1] = 2 * (yz - xw)
    mat[2,2] = 1 - 2 * (xx + yy)

    return mat

#============----------------   transform   ----------------============#

def transform_by_matrix( points, mat, is_vec=False, is_point_image=False ):
    """
    Args:
        points: np.array [N, 3]
        mat: np.array [4, 4]
        is_vec: bool
        
    Returns:
        trans_points: np.array [N, 3]
    """
    rot = mat[:3, :3]

    w, h = mat.shape
    if w == 3 and h == 3:
        m = np.identity(4)
        m[:3,:3] = rot
        mat = m

    if is_point_image:
        trans_points = np.einsum('ij,abj->abi', rot, points )
    else:
        trans_points = np.einsum('ij,aj->ai', rot, points )

    if not is_vec:
        trans = mat[:3, 3]
        trans_points += trans
    
    return trans_points

def depth_to_pc_fast(depth, cam_intrinsic, cam_to_world, with_noise=False):
    """
    Args:
        depth: np.array [w, h, 3]
        cam_intrinsic: np.array [3, 3]
        cam_to_world: np.array [3, 3]
        with_noise: bool
    
    Returns:
        pointcloud: np.array [w, h, 3]
    """
    
    depth = depth.transpose(1, 0)
    w, h = depth.shape

    u0 = cam_intrinsic[0,2]
    v0 = cam_intrinsic[1,2]
    fx = cam_intrinsic[0, 0]
    fy = cam_intrinsic[1, 1]
    
    v, u = np.meshgrid( range(h), range(w) )
    z = depth
    x = (u - u0) * z / fx
    y = (v - v0) * z / fy

    z = z.reshape(w, h, 1)
    x = x.reshape(w, h, 1)
    y = y.reshape(w, h, 1)

    depth = depth.transpose(1, 0)
    # 640 * 480 * 3
    ret = np.concatenate([x,y,z], axis=-1)

    # translate to world coordinate
    ret = transform_by_matrix(ret, cam_to_world, is_point_image=True)

    ret = ret.transpose(1, 0, 2)

    if with_noise:
        shape = ret.shape
        # ret = ret + (np.random.rand( shape[0], shape[1], shape[2] ) - 0.5 ) * 0.005
        ret = ret + (np.random.normal( size=shape  ) - 0.5 ) * 0.005

    return ret