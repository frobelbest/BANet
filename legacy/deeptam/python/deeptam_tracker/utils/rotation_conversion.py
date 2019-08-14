from minieigen import Vector3, Matrix3, Quaternion
import numpy as np

def numpy_to_Vector3(arr):
    tmp = arr.astype(np.float64)
    return Vector3(tmp[0],tmp[1],tmp[2])

def angleaxis_to_angle_axis(aa, epsilon=1e-6):
    """Converts the angle axis vector with angle encoded as magnitude to 
    the angle axis representation with seperate angle and axis.

    aa: minieigen.Vector3
        axis angle with angle as vector magnitude

    epsilon: minimum angle in rad
        If the angle is smaller than epsilon
        then 0,(1,0,0) will be returned

    returns the tuple (angle,axis)
    """
    angle = aa.norm()
    if angle < epsilon:
        angle = 0
        axis = Vector3(1,0,0)
    else:
        axis = aa.normalized()
    return angle, axis


def angleaxis_to_quaternion(aa, epsilon=1e-6):
    """Converts the angle axis vector with angle encoded as magnitude to 
    the quaternion representation.

    aa: minieigen.Vector3
        axis angle with angle as vector magnitude

    epsilon: minimum angle in rad
        If the angle is smaller than epsilon
        then 0,(1,0,0) will be returned

    returns the unit quaternion
    """
    angle, axis = angleaxis_to_angle_axis(aa,epsilon)
    return Quaternion(angle,axis)



def angleaxis_to_rotation_matrix(aa, epsilon=1e-6):
    """Converts the angle axis vector with angle encoded as magnitude to 
    the rotation matrix representation.

    aa: minieigen.Vector3 or np.array
        axis angle with angle as vector magnitude

    epsilon: minimum angle in rad
        If the angle is smaller than epsilon
        then 0,(1,0,0) will be returned

    returns the 3x3 rotation matrix as numpy.ndarray
    """
    if not isinstance(aa,Vector3):
        _tmp = np.squeeze(aa).astype(np.float64)
        _aa = Vector3(_tmp[0], _tmp[1], _tmp[2])
    else:
        _aa = Vector3(aa)
    q = angleaxis_to_quaternion(_aa,epsilon)
    tmp = q.toRotationMatrix()
    return np.array(tmp)


def rotation_matrix_to_angleaxis(R):
    """Converts the rotation matrix to an angle axis vector with the angle 
    encoded as the magnitude.

    R: minieigen.Matrix3 or np.array

    returns an angle axis vector as np.array
    """
    angle,axis = Quaternion(R).toAngleAxis()
    aa = angle*axis
    return np.array(aa)


    
