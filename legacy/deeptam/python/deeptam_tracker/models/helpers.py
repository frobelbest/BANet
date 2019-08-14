import lmbspecialops as sops
import tensorflow as tf
import math
from minieigen import Quaternion
import numpy as np


def convert_NCHW_to_NHWC(inp):
    """Convert the tensor from caffe format BCHW into tensorflow format BHWC
        
        inp: tensor 
    """
    return tf.transpose(inp,[0,2,3,1])

def convert_NHWC_to_NCHW(inp):
    """Convert the tensor from tensorflow format BHWC into caffe format BCHW 
        
        inp: tensor 
    """
    return tf.transpose(inp,[0,3,1,2])

def myLeakyRelu(x):
    """Leaky ReLU with leak factor 0.1"""
    # return tf.maximum(0.1*x,x)
    return sops.leaky_relu(x, leak=0.1)
    #return tf.nn.leaky_relu(x, alpha=0.1)


def default_weights_initializer():
    return tf.variance_scaling_initializer(scale=2)


def conv2d(inputs, num_outputs, kernel_size, data_format, padding=None, **kwargs):
    """Convolution with 'same' padding"""

    if padding is None:
        padding='same'
    return tf.layers.conv2d(
        inputs=inputs,
        filters=num_outputs,
        kernel_size=kernel_size,
        kernel_initializer=default_weights_initializer(),
        padding=padding,
        data_format=data_format,
        **kwargs,
        )


def convrelu(inputs, num_outputs, kernel_size, data_format, activation=None, **kwargs):
    """Shortcut for a single convolution+relu 
    
    See tf.layers.conv2d for a description of remaining parameters
    """
    if activation is None:
        activation=myLeakyRelu
    return conv2d(inputs, num_outputs, kernel_size, data_format, activation=activation, **kwargs)


def convrelu2(inputs, num_outputs, kernel_size, name, stride, data_format, padding=None, activation=None, **kwargs):
    """Shortcut for two convolution+relu with 1D filter kernels 
    
    num_outputs: int or (int,int)
        If num_outputs is a tuple then the first element is the number of
        outputs for the 1d filter in y direction and the second element is
        the final number of outputs.
    """
    if isinstance(num_outputs,(tuple,list)):
        num_outputs_y = num_outputs[0]
        num_outputs_x = num_outputs[1]
    else:
        num_outputs_y = num_outputs
        num_outputs_x = num_outputs

    if isinstance(kernel_size,(tuple,list)):
        kernel_size_y = kernel_size[0]
        kernel_size_x = kernel_size[1]
    else:
        kernel_size_y = kernel_size
        kernel_size_x = kernel_size

    if padding is None:
        padding='same'

    if activation is None:
        activation=myLeakyRelu

    tmp_y = tf.layers.conv2d(
        inputs=inputs,
        filters=num_outputs_y,
        kernel_size=[kernel_size_y,1],
        strides=[stride,1],
        padding=padding,
        activation=activation,
        kernel_initializer=default_weights_initializer(),
        data_format=data_format,
        name=name+'y',
        **kwargs,
    )
    return tf.layers.conv2d(
        inputs=tmp_y,
        filters=num_outputs_x,
        kernel_size=[1,kernel_size_x],
        strides=[1,stride],
        padding=padding,
        activation=activation,
        kernel_initializer=default_weights_initializer(),
        data_format=data_format,
        name=name+'x',
        **kwargs,
    )

def fcrelu(inputs, name, num_outputs, weights_regularizer=None, activation=None, **kwargs):
    """Shortcut for fully_connected layer + relu 
    
    See tf.layers.dense for a description of remaining parameters

    num_outputs: int 

    """
    if activation is None:
        activation=myLeakyRelu

    return tf.layers.dense(
            inputs=inputs,
            units=num_outputs,
            activation=activation,
            kernel_initializer=default_weights_initializer(),
            name=name,
            **kwargs,
            )

def scale_tensor(inp, scale_factor, data_format='NCHW'):
    """ Down/Up scale the tensor by a factor using nearest neighbor
    
        inp: tensor in BCHW
        
        scale_factor: signed int (+: upscale, -: downscale)
        
        data_format: str
        
    """
    if data_format == 'NCHW':
        if scale_factor == 0:
            return inp
        else:
            data_shape = inp.get_shape().as_list()
            channel_num = data_shape[1]
            height_org = data_shape[2]
            width_org = data_shape[3]
            height_new = int(height_org*math.pow(2,scale_factor))
            width_new = int(width_org*math.pow(2,scale_factor))
            
            inp_tmp = convert_NCHW_to_NHWC(inp)
            resize_shape_tensor = tf.constant([height_new, width_new], tf.int32)
            inp_resize = tf.image.resize_nearest_neighbor(inp_tmp,resize_shape_tensor)

            return convert_NHWC_to_NCHW(inp_resize)
    else: 
        raise Exception('scale_tensor does not support {0} format now.'.format(data_format))

def resize_nearest_neighbor_NCHW(inp, size):
    """ shortcut for resizing with NCHW format
        
        inp: Tensor:
        size: list with height and width
    """
    if inp.get_shape().as_list()[-2:] == list(size):
        return inp
    else:
        return convert_NHWC_to_NCHW(tf.image.resize_nearest_neighbor(convert_NCHW_to_NHWC(inp),size,align_corners=True))

def resize_area_NCHW(inp, size):
    """ shortcut for resizing with NCHW format

    inp: Tensor
    size: list with height and width
    """
    if inp.get_shape().as_list()[-2:] == list(size):
        return inp
    else:
        return convert_NHWC_to_NCHW(tf.image.resize_area(convert_NCHW_to_NHWC(inp),size,align_corners=True))


def apply_motion_increment(R_prev, t_prev, R_inc, t_inc):
    """Apply motion increment to previous motion
    
        R_next = R_inv*R_prev
        t_next = t_inc + R_inc*t_prev
        
        
    """
    R_matrix_prev = sops.angle_axis_to_rotation_matrix(R_prev)
    R_matrix_inc = sops.angle_axis_to_rotation_matrix(R_inc)
    R_matrix_next = tf.matmul(R_matrix_inc, R_matrix_prev)
    R_angleaxis_next = sops.rotation_matrix_to_angle_axis(R_matrix_next)
    t_next = tf.add(t_inc, tf.squeeze(tf.matmul(R_matrix_inc, tf.expand_dims(t_prev,2)),[2,]))
    
    return R_angleaxis_next, t_next



