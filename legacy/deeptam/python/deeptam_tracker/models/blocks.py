from .helpers import *
import tensorflow as tf
import string


def create_flow_inputs_and_gt(
        key_image, 
        current_image, 
        intrinsics, 
        depth, 
        rotation, 
        translation, 
        suffix=None):
    """Creates the input for the flow block and the corresponding ground truth

    key_image: Tensor
        keyframe image on NCHW format.
        key_image must have the same shape as current_image.

    current_image: Tensor
        current image in NCHW format
        current_image must have the same shape as key_image.

    depth: Tensor
        The predicted depth map.
        depth must have have the same shape as key_image and current_image.

    intrinsics: Tensor
        The normalized intrinsics

    rotation: Tensor
        The predicted rotation

    translation: Tensor
        The predicted translation

    suffix: str
        Suffix for the tensorboard summaries.
    """
    if suffix is None:
        suffix = ""


    rendered_image_near, rendered_depth_near = sops.transfer_key_frame2(
            image = key_image, 
            depth = depth, 
            intrinsics = intrinsics, 
            rotation = rotation,
            translation = translation,
            inverse_depth=True,
            pass_near=True)

    rendered_image_far, rendered_depth_far = sops.transfer_key_frame2(
            image = key_image, 
            depth = depth, 
            intrinsics = intrinsics, 
            rotation = rotation,
            translation = translation,
            inverse_depth=True,
            pass_near=False)

    rendered_image_near2, _ = sops.transfer_key_frame2(
            image = key_image, 
            depth = 1.2*depth, 
            intrinsics = intrinsics, 
            rotation = rotation,
            translation = translation,
            inverse_depth=True,
            pass_near=True)


    flow_input = tf.concat([rendered_image_near,
                            rendered_image_far,
                            rendered_image_near2,
                            rendered_depth_near,
                            rendered_depth_far,
                            current_image,
                            ],
                            axis=1)

    flow_input = tf.stop_gradient(sops.replace_nonfinite(flow_input))

    result = {
        'flow_input' : flow_input,
        'rendered_image_near': sops.replace_nonfinite(rendered_image_near),
        'rendered_depth_near_far': sops.replace_nonfinite(tf.concat([rendered_depth_near, rendered_depth_far],axis=1)),
    }

    return result

def motion_block(block_inputs, weights_regularizer=None, resolution_level=0, sigma_epsilon=0.1, data_format='channels_first'):
    """Creates a motion network
    
    block_inputs: [(Tensor, num_outputs)]
        List of input tensors and the target number of output channels for each feature before concatenation
        The tensor format is NCHW.

    weights_regularizer: function
        A function returning a weight regularizer
    
    Returns predict rotation and translation
    """
    conv_params = {'kernel_regularizer': weights_regularizer, 'data_format': data_format}
    fc_params = {'kernel_regularizer': weights_regularizer}
    padding='valid'

    conv1_kernel_size = {2: 3, 1: 5, 0: 5}
    conv1_stride = {2: 2, 1: 4, 0:4}
    ks = conv1_kernel_size[resolution_level]
    s = conv1_stride[resolution_level]
            
    # motion prediction part
    with tf.variable_scope('motion'):
        
        features = []
        for i, inp_num_outputs in enumerate(block_inputs):
            inp, num_outputs = inp_num_outputs
            name = 'motion_conv1_{0}'.format(string.ascii_lowercase[i])
            conv1 = convrelu2(name=name, inputs=inp, num_outputs=num_outputs, kernel_size=ks, stride=s, padding=padding, **conv_params)
            name = 'motion_conv2_{0}'.format(string.ascii_lowercase[i])
            conv2 = convrelu2(name=name, inputs=conv1, num_outputs=num_outputs, kernel_size=3, stride=1, padding=padding, **conv_params)
            features.append(conv2)
            print(conv1)
            print(conv2)
        
        motion_conv2 = tf.concat(features,axis=1,name='motion_conv2',)
        print(motion_conv2,flush=True)
  
        conv3_kernel_size = {2: 3, 1: 3, 0: 5}
        conv3_stride = {2: 2, 1: 2, 0: 4}
        ks = conv3_kernel_size[resolution_level]
        s = conv3_stride[resolution_level]
        motion_conv3 = convrelu2(name='motion_conv3', inputs=motion_conv2, num_outputs=(96,96), kernel_size=ks, stride=s, padding=padding, **conv_params)
        print(motion_conv3)
        
        motion_conv4 = convrelu2(name='motion_conv4', inputs=motion_conv3, num_outputs=(128,128), kernel_size=3, stride=2, padding=padding, **conv_params)
        print(motion_conv4)
        
        motion_conv5 = convrelu2(name='motion_conv5', inputs=motion_conv4, num_outputs=(256,256), kernel_size=3, stride=2, padding=padding, **conv_params)
        print(motion_conv5)



        motion_conv5_shape =  motion_conv5.get_shape().as_list()
        batch_size = motion_conv5_shape[0]
        num_predictions = 64
        values_per_prediction = (motion_conv5_shape[1]*motion_conv5_shape[2]*motion_conv5_shape[3])//num_predictions
        
        motion_fc1 = fcrelu(name='motion_fc1', inputs=tf.contrib.layers.flatten(motion_conv5), num_outputs=values_per_prediction*num_predictions, **fc_params)
        
        motion_fc1_reshaped = tf.reshape(motion_fc1, [batch_size,values_per_prediction,num_predictions,1])
        motion_conv6 = convrelu(name='motion_conv6', inputs=motion_fc1_reshaped, num_outputs=values_per_prediction, kernel_size=1, padding='valid', **conv_params)
        motion_conv7 = convrelu(name='motion_conv7', inputs=motion_conv6, num_outputs=16, kernel_size=1, padding='valid', **conv_params)
        predict_motion = conv2d(name='motion_predict', inputs=motion_conv7, num_outputs=6, kernel_size=1, padding='valid', **conv_params)
        
        tf.summary.histogram('predict_motion', predict_motion)


        scale_motion = 0.1
        predict_motion = scale_motion*tf.reshape(predict_motion, [batch_size, 6, num_predictions])
        mean_prediction = tf.reduce_mean(predict_motion, axis=-1, keep_dims=True) # [N,6,1]
        tf.summary.histogram('mean_prediction', mean_prediction)

        deviations = predict_motion-mean_prediction # [N,6,num_predictions]
        tf.summary.histogram('deviations', deviations)
        sigma = tf.matmul(deviations, deviations, transpose_b=True)/num_predictions
        tf.summary.histogram('sigma_without_eps', sigma)
        sigma = sigma + sigma_epsilon*tf.eye(6, 6, batch_shape=[batch_size], dtype=sigma.dtype)


        mean_prediction = tf.squeeze(mean_prediction,axis=[-1])

        predict_rotation, predict_translation = tf.split(value=mean_prediction, num_or_size_splits=[3,3], axis=1)

        predict_rotation_samples, predict_translation_samples = tf.split(predict_motion, num_or_size_splits=[3,3], axis=1)
        
        print(motion_conv3)
        print(motion_conv4)
        print(motion_conv5)
        print(motion_fc1)
        print(motion_conv6)
        print(motion_conv7)
        print(predict_motion)
        print(predict_rotation,predict_translation)
        print(predict_rotation_samples,predict_translation_samples)
 
    return{
            'motion_conv5': motion_conv5,
            'motion_fc1': motion_fc1,
            'predict_rotation': predict_rotation,
            'predict_translation': predict_translation,
            'predict_motion_sigma': sigma,
            'predict_rotation_samples': predict_rotation_samples,
            'predict_translation_samples': predict_translation_samples,
            'predict_motion_samples': predict_motion,
            'num_samples': num_predictions,
          }


def _predict_flow(inp, channels=2, **kwargs ):
    """Generates a tensor for optical flow prediction
    
    inp: Tensor
    """

    tmp = convrelu(
        inputs=inp,
        num_outputs=24,
        kernel_size=3,
        strides=1,
        name="conv1",
        **kwargs,
    )
    
    output = conv2d(
        inputs=tmp,
        num_outputs=channels,
        kernel_size=3,
        strides=1,
        name="conv2",
        **kwargs,
    )
    
    return output


def _upsample_prediction(inp, num_outputs, **kwargs ):
    """Upconvolution for upsampling predictions
    
    inp: Tensor 
        Tensor with the prediction
        
    num_outputs: int
        Number of output channels. 
        Usually this should match the number of channels in the predictions
    """
    output = tf.layers.conv2d_transpose(
        inputs=inp,
        filters=num_outputs,
        kernel_size=4,
        strides=2,
        padding='same',
        activation=None,
        kernel_initializer=default_weights_initializer(),
        name="upconv",
        **kwargs,
    )
    return output


def _refine(inp, num_outputs, data_format, upsampled_prediction=None, features_direct=None, **kwargs):
    """ Generates the concatenation of 
         - the previous features used to compute the flow/depth
         - the upsampled previous flow/depth
         - the direct features that already have the correct resolution

    inp: Tensor
        The features that have been used before to compute flow/depth

    num_outputs: int 
        number of outputs for the upconvolution of 'features'

    upsampled_prediction: Tensor
        The upsampled flow/depth prediction

    features_direct: Tensor
        The direct features which already have the spatial output resolution
    """
    upsampled_features = tf.layers.conv2d_transpose(
        inputs=inp,
        filters=num_outputs,
        kernel_size=4,
        strides=2,
        padding='same',
        activation=myLeakyRelu,
        kernel_initializer=default_weights_initializer(),
        data_format=data_format,
        name="upconv",
        **kwargs,
    )
   
    inputs = [upsampled_features, features_direct, upsampled_prediction]
    valid_inputs = [ x for x in inputs if not x is None ]

    # deal with possible shape mismatches due to odd input dimensions
    if data_format == 'channels_first':
        input_shapes_HW = [ x.get_shape().as_list()[-2:] for x in valid_inputs ]
    else:
        input_shapes_HW = [ x.get_shape().as_list()[1:3] for x in valid_inputs ]
    common_shape_HW = list(np.min(input_shapes_HW, axis=0))
    
    concat_inputs = []
    for x in valid_inputs:
        original_shape = x.get_shape().as_list()
        if data_format == 'channels_first':
            new_shape = original_shape[:2] + common_shape_HW
        else:
            new_shape = original_shape[0:1] + common_shape_HW + [original_shape[-1]]
        if original_shape == new_shape:
            concat_inputs.append(x)
        else:
            print(original_shape, new_shape)
            concat_inputs.append(tf.slice(x,begin=[0,0,0,0], size=new_shape))
            
    
    if data_format == 'channels_first':
        return tf.concat(concat_inputs, axis=1)
    else: # NHWC
        return tf.concat(concat_inputs, axis=3)


def flow_block(block_inputs, weights_regularizer=None, data_format='channels_first' ):
    """Creates a flow block
    """
    conv_params = {'kernel_regularizer': weights_regularizer, 'data_format': data_format}
    fc_params = {'kernel_regularizer': weights_regularizer,}
    
    with tf.variable_scope('flowdepth'):
        # contracting part
        conv0 = convrelu2(name='conv0', inputs=block_inputs, num_outputs=(24,24), kernel_size=3, stride=1, **conv_params)
        conv0_1 = convrelu2(name='conv0_1', inputs=conv0, num_outputs=24, kernel_size=3, stride=1, **conv_params)
        
        conv1 = convrelu2(name='conv1', inputs=conv0_1, num_outputs=(24,32), kernel_size=3, stride=2, **conv_params)
        conv1_1 = convrelu2(name='conv1_1', inputs=conv1, num_outputs=32, kernel_size=3, stride=1, **conv_params)
        
        conv2 = convrelu2(name='conv2', inputs=conv1_1, num_outputs=(48,64), kernel_size=3, stride=2, **conv_params)
        conv2_1 = convrelu2(name='conv2_1', inputs=conv2, num_outputs=64, kernel_size=3, stride=1, **conv_params)
        
        conv3 = convrelu2(name='conv3', inputs=conv2_1, num_outputs=(96,128), kernel_size=3, stride=2, **conv_params)
        conv3_1 = convrelu2(name='conv3_1', inputs=conv3, num_outputs=128, kernel_size=3, stride=1, **conv_params)
        
        
        print(conv0, conv0_1)
        print(conv1, conv1_1)
        print(conv2, conv2_1)
        print(conv3, conv3_1)

        # expanding part
        with tf.variable_scope('predict_flow3'):
            predict_flow3 = _predict_flow(conv3_1, channels=4, **conv_params)
        
        with tf.variable_scope('upsample_flow3to2'):
            predict_flow3to2 = _upsample_prediction(predict_flow3, 4, data_format=data_format)
    
        print(predict_flow3, predict_flow3to2)


        with tf.variable_scope('refine2'):
            concat2 = _refine(
                inp=conv3_1, 
                num_outputs=64, 
                upsampled_prediction=predict_flow3to2, 
                features_direct=conv2_1,
                data_format=data_format,
            )
            print(concat2)
            
            
        with tf.variable_scope('refine1'):
            concat1 = _refine(
                inp=concat2, 
                num_outputs=48,
                features_direct=conv1_1,
                data_format=data_format,
            )
            print(concat1)
            
        with tf.variable_scope('refine0'):
            concat0 = _refine(
                inp=concat1, 
                num_outputs=32, 
                features_direct=conv0_1,
                data_format=data_format,
            )
            print(concat0)
            
        with tf.variable_scope('predict_flow0'):
            predict_flow0 = _predict_flow(concat0, channels=4,  **conv_params)
            
    scale_flow = 0.01
    return {
            'predict_flow0_near': scale_flow*predict_flow0[:,0:2,:,:],
            'predict_flow0_far': scale_flow*predict_flow0[:,2:,:,:],
            'predict_flow3_near': scale_flow*predict_flow3[:,0:2,:,:],
            'predict_flow3_far': scale_flow*predict_flow3[:,2:4,:,:],
            'conv3_1': conv3_1,
            'concat2': concat2,
            'concat1': concat1,
            'concat0': concat0,
            }

