from collections import namedtuple

# depth always stores the absolute depth values (not inverse depth)
# image is a PIL.Image with the same dimensions as depth
# depth_metric should always be 'camera_z'
# K corresponds to the width and height of image/depth
# R, t is the world to camera transform
View = namedtuple('View',['R','t','K','image','depth','depth_metric'])


# stores a camera pose
# R, t is the world to camera transform
Pose = namedtuple('Pose',['R','t'])

def Pose_identity():
    """Returns the identity pose"""
    from minieigen import Matrix3, Vector3
    return Pose(R = Matrix3.Identity, t = Vector3.Zero)

