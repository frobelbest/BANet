import numpy as np
from PIL import Image
from minieigen import MatrixX,Vector3
from .datatypes import *

def convert_array_to_colorimg(inp):
    """Returns the img as PIL images"""
    image_arr = inp.copy()
    if image_arr.dtype == np.float32:
        image_arr += 0.5
        image_arr *= 255
        image_arr = image_arr.astype(np.uint8)
    image_arr = image_arr[0:3,:,:]
    image_arr = np.rollaxis(image_arr,0,3)
    return Image.fromarray(image_arr)


def convert_array_to_grayimg(inp):
    """Convert single channel array to grayscale PIL image. """
    arr = inp.copy()
    arr[np.isinf(arr)] = np.nan
    norm_factor = [np.nanmin(arr), np.nanmax(arr)-np.nanmin(arr)]
    
    if norm_factor[1] == 0:
        raise RuntimeError('Cannot convert array.')
    else:
        arr -= norm_factor[0]
        arr /= norm_factor[1]
        arr *= 255
        arr[np.isnan(arr)] = 0
        arr[np.isinf(arr)] = 0
        arr = arr.astype(np.uint8)
    return Image.fromarray(arr)



def convert_between_c2w_w2c(inp):
    """Converts camera-to-world pose to world-to-camera pose or vice verse
    
    inp: Pose
    """
    
    T = MatrixX.Identity(4,4)
    rot = inp.R
    trans = inp.t
    
    T[0,0] = rot[0,0]
    T[0,1] = rot[0,1]
    T[0,2] = rot[0,2]
    T[1,0] = rot[1,0]
    T[1,1] = rot[1,1]
    T[1,2] = rot[1,2]
    T[2,0] = rot[2,0]
    T[2,1] = rot[2,1]
    T[2,2] = rot[2,2]

    T[0,3] = trans[0]
    T[1,3] = trans[1]
    T[2,3] = trans[2]
    
    T_inv = T.inverse()
    
    rot_inv = rot.transpose()
    trans_inv = Vector3(T_inv[0,3], T_inv[1,3], T_inv[2,3])
    return Pose(R=rot_inv,t=trans_inv)