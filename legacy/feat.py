
import tensorflow as tf
import numpy as np

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-5

reuse_variables=False

def batch_norm(name,inputs,is_training,data_format,use_center=True,use_scale=True,active_batchnorm=False):
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,center=use_center,
        scale=use_scale,training=(is_training and (not fixed_batchnorm)) or active_batchnorm,trainable=is_training,fused=True,name=name,reuse=reuse_variables)
    return inputs

def batch_norm_relu(name, inputs, is_training, data_format,):
    inputs = batch_norm(name,inputs,is_training,data_format)
    inputs = tf.nn.relu(inputs)
    return inputs

def batch_norm_selu(name, inputs, is_training, data_format):
    inputs = batch_norm(name,inputs,is_training,data_format,active_batchnorm=True)
    inputs = tf.nn.selu(inputs)
    return inputs

def reflect_padding(inputs,padding,data_format):
    with tf.name_scope('padding'):
        if data_format == 'channels_first':
            padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [padding, padding], [padding, padding]],mode='SYMMETRIC')
        else:
            padded_inputs = tf.pad(inputs, [[0, 0], [padding, padding], [padding, padding], [0, 0]],mode='SYMMETRIC')
    return padded_inputs

#should do reflect padding!zero padding is stupid!
def conv2d(name,inputs,filters,kernel_size,strides=1,padding=1,dilation=1,data_format='channels_first',is_training=False,use_bias=False,activation=None):

    if kernel_size > 1 and padding > 0:
        inputs = reflect_padding(inputs,padding,data_format)

    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding='valid', use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.he_normal(),activation=activation,
        data_format=data_format,name=name,dilation_rate=(dilation,dilation),trainable=is_training)

def projection_shortcut(inputs,filters,stride,is_training,data_format,use_bn=True,use_bias=False):
    with tf.variable_scope('downsample'):
        inputs = conv2d('0',inputs=inputs, filters=filters, kernel_size=1, strides=stride,data_format=data_format,is_training=is_training,use_bias=use_bias)
        if use_bn:
            inputs = batch_norm('1',inputs,is_training,data_format)
        return inputs

class _block:

    def __init__(self,name,filters,strides,downsample,dilation,residual,is_training,data_format):
        pass

    def inference(self,inputs):
        return None

class building_block(_block):
    
    expansion = 1
    
    def __init__(self,name,filters,strides=1,downsample=None,dilation=(1,1),residual=True,is_training=False,data_format='channels_first'):

        
        self.name=name

        #origin parameters
        self.filters=filters
        self.strides=strides
        self.downsample=downsample
        self.dilation=dilation
        self.residual=residual

        #tensorflow parameters
        self.is_training=is_training
        self.data_format=data_format


    def inference(self,inputs):

        with tf.variable_scope(self.name):

            if self.residual:
                if self.downsample is None:
                    shortcut = inputs
                else:
                    shortcut = self.downsample(inputs,self.filters*self.expansion,self.strides,self.is_training,self.data_format)
    
            inputs = conv2d('conv1',inputs=inputs, filters=self.filters, kernel_size=3, strides=self.strides,
                            padding=self.dilation[0],dilation=self.dilation[0],data_format=self.data_format,is_training=self.is_training)
            inputs = batch_norm_relu('bn1',inputs,self.is_training,self.data_format)
            
            inputs = conv2d('conv2',inputs=inputs, filters=self.filters, kernel_size=3, strides=1,
                            padding=self.dilation[1],dilation=self.dilation[1],data_format=self.data_format,is_training=self.is_training)
            inputs = batch_norm('bn2',inputs,self.is_training,self.data_format)

            if self.residual:
                return tf.nn.relu(inputs + shortcut)
            else:
                return tf.nn.relu(inputs)


class bottleneck_block(_block):

    expansion = 4
    
    def __init__(self,name,filters,strides=1,downsample=None,dilation=(1,1),residual=True,is_training=False,data_format='channels_first'):
        
        self.name=name

        #origin parameters
        self.filters=filters
        self.strides=strides
        self.downsample=downsample
        self.dilation=dilation
        self.residual=residual

        #tensorflow parameters
        self.is_training=is_training
        self.data_format=data_format

    def inference(self,inputs):
        with tf.variable_scope(self.name):

            if self.downsample is None:
                shortcut = inputs
            else:
                shortcut = self.downsample(inputs,self.filters*self.expansion,self.strides,self.is_training,self.data_format)

            inputs = conv2d('conv1',inputs=inputs, filters=self.filters, kernel_size=1, strides=1,data_format=self.data_format,is_training=self.is_training)
            inputs = batch_norm_relu('bn1',inputs,self.is_training,self.data_format)

            inputs = conv2d('conv2',inputs=inputs, filters=self.filters, kernel_size=3, strides=self.strides,
                            padding=self.dilation[1],dilation=self.dilation[1],data_format=self.data_format,is_training=self.is_training)
            inputs = batch_norm_relu('bn2',inputs,self.is_training,self.data_format)

            inputs = conv2d('conv3',inputs=inputs, filters=self.expansion*self.filters, kernel_size=1, strides=1,data_format=self.data_format,is_training=self.is_training)
            inputs = batch_norm('bn3',inputs,self.is_training,self.data_format)

            return tf.nn.relu(inputs+shortcut)


class DRN:

    def __init__(self,block=bottleneck_block,layers=[1, 1, 3, 4, 6, 3, 1, 1],
                 channels=(16, 32, 64, 128, 256, 512, 512, 512),reuse_variables=False,is_training=False):

        self.channels=channels
        self.layers=layers
        self.block=block
        self.is_training=is_training
        self.reuse_variables=reuse_variables

        #fixed format
        self.data_format='channels_first'
        self.layer_list={}

    def layer(self,name,inputs,block,filters,blocks,stride=1, dilation=1,new_level=True,residual=True):
        with tf.variable_scope(name):
            assert dilation == 1 or dilation % 2 == 0
            if stride != 1 or inputs.get_shape().as_list()[1] != filters * block.expansion:
                downsample=projection_shortcut
            else:
                downsample=None

            outputs=block('0',filters,stride,downsample,dilation=(1,1) if dilation==1 else (dilation//2 if new_level else dilation, dilation),residual=residual,is_training=self.is_training,data_format=self.data_format).inference(inputs)
            for i in range(1,blocks):
                outputs=block(str(i),filters,residual=residual,dilation=(dilation,dilation),is_training=self.is_training,data_format=self.data_format).inference(outputs)
            self.layer_list[name]=outputs
            return outputs

    def conv_layers(self,name,inputs,filters,convs,stride=1,dilation=1):
        with tf.variable_scope(name):
            outputs=inputs
            for i in range(convs):
                outputs=conv2d(str(2*i),inputs=outputs,filters=filters,padding=dilation,dilation=dilation,kernel_size=3,strides=stride if i==0 else 1,data_format=self.data_format,is_training=self.is_training)
                outputs=batch_norm_relu(str(2*i+1),inputs=outputs,is_training=self.is_training,data_format=self.data_format)
            self.layer_list[name]=outputs
            return outputs

    def drn54_no_dilation(self,inputs):
        inputs=tf.nn.batch_normalization(inputs/255.0,tf.constant([0.485, 0.456, 0.406]),tf.constant([0.229*0.229,0.224*0.224,0.225*0.225]),offset=None,scale=None,variance_epsilon=0.0)
        if self.data_format=='channels_first':
            inputs=tf.transpose(inputs,[0,3,1,2])
        with tf.variable_scope('DRN',reuse=self.reuse_variables):
            with tf.variable_scope('layer0'):
                self.conv1=conv2d('0',inputs=inputs,filters=self.channels[0],kernel_size=7,strides=1,padding=3,is_training=self.is_training,data_format=self.data_format)
                self.layer0 = batch_norm_relu('1',inputs=self.conv1,is_training=self.is_training,data_format=self.data_format)
            self.layer_list['layer0']=self.layer0
            self.layer1 = self.conv_layers('layer1',inputs=self.layer0,filters=self.channels[0],convs=self.layers[0], stride=1)
            self.layer2 = self.conv_layers('layer2',inputs=self.layer1,filters=self.channels[1],convs=self.layers[1], stride=2)
            self.layer3 = self.layer('layer3',inputs=self.layer2,block=self.block,filters=self.channels[2],blocks=self.layers[2],stride=2)
            self.layer4 = self.layer('layer4',inputs=self.layer3,block=self.block,filters=self.channels[3],blocks=self.layers[3],stride=2)
            #self.layer5 = self.layer('layer5',inputs=self.layer4,block=self.block,filters=self.channels[4],blocks=self.layers[4],new_level=False,stride=2)
            #self.layer6 = self.layer('layer6',inputs=self.layer5,block=self.block,filters=self.channels[5],blocks=self.layers[5],new_level=False,stride=2)
        return self.layer_list

class Pyramid:

    def __init__(self,filters,is_training=False,reuse_variables=False):

        self.filters=filters
        self.is_training=is_training
        self.reuse_variables=reuse_variables

        self.data_format='channels_first'
        self.layer_dict={}

    def upsample(self,inputs,name):
        with tf.variable_scope(name):

            kernel=[[[[ 0.0625,  0.1875,  0.1875,  0.0625],
                      [ 0.1875,  0.5625,  0.5625,  0.1875],
                      [ 0.1875,  0.5625,  0.5625,  0.1875],
                      [ 0.0625,  0.1875,  0.1875,  0.0625]]]]

            filters=int(inputs.get_shape()[1])
            filter_kernel=np.tile(kernel,(filters,1,1,1))
            filter_kernel=np.transpose(filter_kernel,(2,3,0,1)).astype(np.float32)
            tf_kernel=tf.get_variable("conv2d",initializer=filter_kernel,trainable=self.is_training)
            shape =inputs.get_shape()
            inputs=tf.pad(inputs,[[0,0],[0,0],[1,1],[1,1]],'SYMMETRIC')

            output=tf.nn.depthwise_conv2d_native_backprop_input([shape[0],shape[1],2*shape[2]+4,2*shape[3]+4],tf_kernel,inputs,strides=(1,1,2,2),padding='SAME',data_format='NCHW')
            output=output[:,:,2:-2,2:-2]
            output=batch_norm_selu('group_norm',output,self.is_training,self.data_format)
        return output

    def aggregation(self,input1,input2,filters,name):
        with tf.variable_scope(name):
            inputs=tf.concat([input1,input2],axis=1)
            output=conv2d('conv2d',inputs,filters,3,is_training=self.is_training)
            output=batch_norm_selu('group_norm',output,self.is_training,self.data_format)
            return output

    def pyramid(self,layers):
        
        with tf.variable_scope('PYRAMID',reuse=self.reuse_variables):
            
            shape =layers['layer1'].get_shape()
            nbatch=int(shape[0])

            with tf.variable_scope("layer3"):

                _layer4=self.upsample(layers['layer4'],'upsample')
                layer3 =self.aggregation(_layer4,layers['layer3'],256,'aggregate')
                layer3 =conv2d('conv2d_1',layer3,128,3,is_training=self.is_training)
                layer3 =batch_norm_selu('group_norm_1',layer3,self.is_training,self.data_format)
                layer3 =conv2d('conv2d_2',layer3,128,3,is_training=self.is_training)
                layer3 =batch_norm_selu('group_norm_2',layer3,self.is_training,self.data_format)

            with tf.variable_scope("layer2"):
                _layer3=self.upsample(layer3,'upsample')
                layer2=self.aggregation(_layer3,layers['layer2'],256,'aggregate')
                layer2=conv2d('conv2d_1',layer2,128,3,is_training=self.is_training)
                layer2=batch_norm_selu('group_norm_1',layer2,self.is_training,self.data_format)
                layer2=conv2d('conv2d_2',layer2,128,3,is_training=self.is_training)
                layer2=batch_norm_selu('group_norm_2',layer2,self.is_training,self.data_format)

            with tf.variable_scope("layer1"):
                _layer2=self.upsample(layer2,'upsample')
                layer1=self.aggregation(_layer2,layers['layer1'],256,'aggregate')
                layer1=conv2d('conv2d_1',layer1,128,3,is_training=self.is_training)
                layer1=batch_norm_selu('group_norm_1',layer1,self.is_training,self.data_format)
                layer1=conv2d('conv2d_2',layer1,128,3,is_training=self.is_training)
                layer1=batch_norm_selu('group_norm_2',layer1,self.is_training,self.data_format)
            return [tf.transpose(layer3,[0,2,3,1]),tf.transpose(layer2,[0,2,3,1]),tf.transpose(layer1,[0,2,3,1])] 