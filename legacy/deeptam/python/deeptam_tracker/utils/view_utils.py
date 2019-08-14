from .datatypes import View
from PIL import Image
from skimage.transform import resize
import numpy as np

def safe_crop_image(image, box, fill_value):
    """crops an image and adds a border if necessary
    
    image: PIL.Image

    box: 4 tuple
        (x0,y0,x1,y1) tuple

    fill_value: color value, scalar or tuple

    Returns the cropped image
    """
    x0, y0, x1, y1 = box
    if x0 >=0 and y0 >= 0 and x1 < image.width and y1 < image.height:
        return image.crop(box)
    else:
        crop_width = x1-x0
        crop_height = y1-y0
        tmp = Image.new(image.mode, (crop_width, crop_height), fill_value)
        safe_box = (
            max(0,min(x0,image.width-1)),
            max(0,min(y0,image.height-1)),
            max(0,min(x1,image.width)),
            max(0,min(y1,image.height)),
            )
        img_crop = image.crop(safe_box)
        x = -x0 if x0 < 0 else 0
        y = -y0 if y0 < 0 else 0
        tmp.paste(image, (x,y))
        return tmp


def safe_crop_array2d(arr, box, fill_value):
    """crops an array and adds a border if necessary
    
    arr: numpy.ndarray with 2 dims

    box: 4 tuple
        (x0,y0,x1,y1) tuple. x is the column and y is the row!

    fill_value: scalar

    Returns the cropped array
    """
    x0, y0, x1, y1 = box
    if x0 >=0 and y0 >= 0 and x1 < arr.shape[1] and y1 < arr.shape[0]:
        return arr[y0:y1,x0:x1]
    else:
        crop_width = x1-x0
        crop_height = y1-y0
        tmp = np.full((crop_height, crop_width), fill_value, dtype=arr.dtype)
        safe_box = (
            max(0,min(x0,arr.shape[1]-1)),
            max(0,min(y0,arr.shape[0]-1)),
            max(0,min(x1,arr.shape[1])),
            max(0,min(y1,arr.shape[0])),
            )
        x = -x0 if x0 < 0 else 0
        y = -y0 if y0 < 0 else 0
        safe_width = safe_box[2]-safe_box[0]
        safe_height = safe_box[3]-safe_box[1]
        tmp[y:y+safe_height,x:x+safe_width] = arr[safe_box[1]:safe_box[3],safe_box[0]:safe_box[2]]
        return tmp

def adjust_intrinsics(view, K_new, width_new, height_new):
    """Creates a new View with the specified intrinsics and image dimensions.
    The skew parameter K[0,1] will be ignored.
    
    view: View namedtuple
        The view tuple
        
    K_new: numpy.ndarray
        3x3 calibration matrix with the new intrinsics
        
    width_new: int
        The new image width
        
    height_new: int
        The new image height
        
    Returns a View tuple with adjusted image, depth and intrinsics
    """


    #original parameters
    fx = view.K[0,0]
    fy = view.K[1,1]
    cx = view.K[0,2]
    cy = view.K[1,2]
    width = view.image.width
    height = view.image.height
    
    #target param
    fx_new = K_new[0,0]
    fy_new = K_new[1,1]
    cx_new = K_new[0,2]
    cy_new = K_new[1,2]
    
    scale_x = fx_new/fx
    scale_y = fy_new/fy
    
    #resize to get the right focal length
    width_resize = int(width*scale_x)
    height_resize = int(height*scale_y)
    # principal point position in the resized image
    cx_resize = cx*scale_x
    cy_resize = cy*scale_y
    
    img_resize = view.image.resize((width_resize, height_resize), Image.BILINEAR if scale_x > 1 else Image.LANCZOS)
    if not view.depth is None:
        max_depth    = np.max(view.depth)
        depth_resize = view.depth / max_depth
        depth_resize[depth_resize < 0.] = 0.
        depth_resize = resize(depth_resize, (height_resize,width_resize), 0,mode='constant') * max_depth
    else:
        depth_resize = None
    
    #crop to get the right principle point and resolution
    x0 = int(round(cx_resize - cx_new))
    y0 = int(round(cy_resize - cy_new))
    x1 = x0 + int(width_new)
    y1 = y0 + int(height_new)

    if x0 < 0 or y0 < 0 or x1 > width_resize or y1 > height_resize:
        print('Warning: Adjusting intrinsics adds a border to the image')
        img_new = safe_crop_image(img_resize,(x0,y0,x1,y1),(127,127,127))
        if not depth_resize is None:
            depth_new = safe_crop_array2d(depth_resize,(x0,y0,x1,y1),0).astype(np.float32)
        else:
            depth_new = None
    else:
        img_new = img_resize.crop((x0,y0,x1,y1))
        if not depth_resize is None:
            depth_new = depth_resize[y0:y1,x0:x1].astype(np.float32)
        else:
            depth_new = None
    
    return View(R=view.R, t=view.t, K=K_new, image=img_new, depth=depth_new, depth_metric=view.depth_metric)

