
import tensorflow as tf
import numpy as np

_BATCH_NORM_DECAY = 0.95
_BATCH_NORM_EPSILON = 1e-5

fixed_batchnorm=False

def batch_norm(name,inputs,is_training,data_format):
    batch_norm_training=(is_training and (not fixed_batchnorm))
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,center=True,
        scale=True,training=batch_norm_training,trainable=batch_norm_training,fused=True,name=name,reuse=tf.AUTO_REUSE)
    return inputs

def batch_norm_relu(name,inputs,is_training,data_format):
    inputs = batch_norm(name,inputs,is_training,data_format)
    inputs = tf.nn.relu(inputs)
    return inputs

def symmetric_padding(inputs,padding,data_format):
    with tf.name_scope('padding'):
        if data_format == 'channels_first':
            padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [padding, padding], [padding, padding]],mode='SYMMETRIC')
        else:
            padded_inputs = tf.pad(inputs, [[0, 0], [padding, padding], [padding, padding], [0, 0]],mode='SYMMETRIC')
    return padded_inputs

#zero padding is stupid
def conv2d(name,inputs,filters,kernel_size,strides=1,padding=1,dilation=1,data_format='channels_first',is_training=False,use_bias=False,activation=None):

    if kernel_size > 1 and padding > 0:
        inputs = symmetric_padding(inputs,padding,data_format)

    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding='VALID', use_bias=use_bias,
        kernel_initializer=tf.initializers.he_normal(),activation=activation,
        data_format=data_format,name=name,dilation_rate=(dilation,dilation),trainable=is_training)

def projection_shortcut(inputs,filters,stride,is_training,data_format):
    with tf.variable_scope('downsample'):
        inputs = conv2d('0',inputs=inputs, filters=filters, kernel_size=1, strides=stride,data_format=data_format,is_training=is_training)
        inputs = batch_norm('1',inputs,is_training,data_format)
        return inputs

class _block:

    def __init__(self,name,filters,strides,downsample,dilation,residual,is_training,data_format,activation=None):
        pass

    def inference(self,inputs):
        return None

class building_block(_block):
    
    expansion = 1
    
    def __init__(self,name,filters,strides=1,downsample=None,dilation=(1,1),residual=True,is_training=False,data_format='channels_first',activation=tf.nn.relu):

        
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
        self.activation=activation


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
                return self.activation(inputs + shortcut)
            else:
                return self.activation(inputs)


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

    def __init__(self,block=None,layers=None,
                 channels=(16, 32, 64, 128, 256, 512)):

        self.channels=channels
        self.layers=layers
        self.block=block
        self.data_format='channels_first'
    
    def layer(self,name,inputs,block,filters,blocks,stride=1,dilation=1,new_level=True,residual=True,is_training=None):
        with tf.variable_scope(name):
            assert dilation == 1 or dilation % 2 == 0
            if stride != 1 or inputs.get_shape().as_list()[1] != filters * block.expansion:
                downsample=projection_shortcut
            else:
                downsample=None

            if stride==2:
                inputs =tf.nn.avg_pool(inputs,[1,1,2,2],[1,1,2,2],'VALID',data_format='NCHW')

            outputs=block('0',filters,1,downsample,dilation=(1,1) if dilation==1 else (dilation//2 if new_level else dilation, dilation),residual=residual,is_training=is_training,data_format=self.data_format).inference(inputs)
            for i in range(1,blocks):
                outputs=block(str(i),filters,residual=residual,dilation=(dilation,dilation),is_training=is_training,data_format=self.data_format).inference(outputs)
            return outputs

    def conv_layers(self,name,inputs,filters,convs,stride=1,dilation=1,is_training=None):
        with tf.variable_scope(name):
            
            outputs=inputs
            if stride==2:
                outputs=tf.nn.avg_pool(outputs,[1,1,2,2],[1,1,2,2],'VALID',data_format='NCHW')

            for i in range(convs):
                outputs=conv2d(str(2*i),inputs=outputs,filters=filters,padding=dilation,dilation=dilation,kernel_size=3,strides=1,data_format=self.data_format,is_training=is_training)
                outputs=batch_norm_relu(str(2*i+1),inputs=outputs,is_training=is_training,data_format=self.data_format)
            return outputs

    def drn22_no_dilation(self,inputs,is_trainings=None,reuse_variables=False):

        self.block =building_block
        self.layers=[1,1,2,2,2,2]

        inputs=tf.nn.batch_normalization(inputs/255.0,tf.constant([0.485, 0.456, 0.406]),tf.constant([0.229*0.229,0.224*0.224,0.225*0.225]),offset=None,scale=None,variance_epsilon=0.0)
        if self.data_format=='channels_first':
            inputs=tf.transpose(inputs,[0,3,1,2])

        with tf.variable_scope('DRN',reuse=reuse_variables):
            with tf.variable_scope('layer0'):
                self.conv1  = conv2d('0',inputs=inputs,filters=self.channels[0],kernel_size=7,strides=1,padding=3,is_training=is_trainings[-1],data_format=self.data_format)
                self.layer0 = batch_norm_relu('1',inputs=self.conv1,is_training=is_trainings[-1],data_format=self.data_format)
            self.layer1 = self.conv_layers('layer1',inputs=self.layer0,filters=self.channels[0],convs=self.layers[0],stride=1,is_training=is_trainings[-1])
            self.layer2 = self.conv_layers('layer2',inputs=self.layer1,filters=self.channels[1],convs=self.layers[1],stride=2,is_training=is_trainings[-1])
            self.layer3 = self.layer('layer3',inputs=self.layer2,block=self.block,filters=self.channels[2],blocks=self.layers[2],stride=2,is_training=is_trainings[-2])
            self.layer4 = self.layer('layer4',inputs=self.layer3,block=self.block,filters=self.channels[3],blocks=self.layers[3],stride=2,is_training=is_trainings[-3])
            self.layer5 = self.layer('layer5',inputs=self.layer4,block=self.block,filters=self.channels[4],blocks=self.layers[4],stride=2,is_training=is_trainings[-4])
            self.layer6 = self.layer('layer6',inputs=self.layer5,block=self.block,filters=self.channels[5],blocks=self.layers[5],stride=2,is_training=is_trainings[-5])
        return [self.layer6,self.layer5,self.layer4,self.layer3,self.layer2]

    def drn38_no_dilation(self,inputs,is_trainings=None,reuse_variables=False):
        self.block =building_block
        self.layers=[1,1,3,4,6,3]

        inputs=tf.nn.batch_normalization(inputs/255.0,tf.constant([0.485, 0.456, 0.406]),tf.constant([0.229*0.229,0.224*0.224,0.225*0.225]),offset=None,scale=None,variance_epsilon=0.0)
        if self.data_format=='channels_first':
            inputs=tf.transpose(inputs,[0,3,1,2])

        with tf.variable_scope('DRN',reuse=reuse_variables):
            with tf.variable_scope('layer0'):
                self.conv1  = conv2d('0',inputs=inputs,filters=self.channels[0],kernel_size=7,strides=1,padding=3,is_training=is_trainings[-1],data_format=self.data_format)
                self.layer0 = batch_norm_relu('1',inputs=self.conv1,is_training=is_trainings[-1],data_format=self.data_format)
            self.layer1 = self.conv_layers('layer1',inputs=self.layer0,filters=self.channels[0],convs=self.layers[0],stride=1,is_training=is_trainings[-1])
            self.layer2 = self.conv_layers('layer2',inputs=self.layer1,filters=self.channels[1],convs=self.layers[1],stride=2,is_training=is_trainings[-1])
            self.layer3 = self.layer('layer3',inputs=self.layer2,block=self.block,filters=self.channels[2],blocks=self.layers[2],stride=2,is_training=is_trainings[-2])
            self.layer4 = self.layer('layer4',inputs=self.layer3,block=self.block,filters=self.channels[3],blocks=self.layers[3],stride=2,is_training=is_trainings[-3])
            self.layer5 = self.layer('layer5',inputs=self.layer4,block=self.block,filters=self.channels[4],blocks=self.layers[4],stride=2,is_training=is_trainings[-4])
            self.layer6 = self.layer('layer6',inputs=self.layer5,block=self.block,filters=self.channels[5],blocks=self.layers[5],stride=2,is_training=is_trainings[-5])
        return [self.layer6,self.layer5,self.layer4,self.layer3,self.layer2,self.layer1]
    
    def drn54_no_dilation(self,inputs,is_trainings=None,reuse_variables=False):
        self.block =bottleneck_block
        self.layers=[1,1,3,4,6,3]

        inputs=tf.nn.batch_normalization(inputs/255.0,tf.constant([0.485, 0.456, 0.406]),tf.constant([0.229*0.229,0.224*0.224,0.225*0.225]),offset=None,scale=None,variance_epsilon=0.0)
        if self.data_format=='channels_first':
            inputs=tf.transpose(inputs,[0,3,1,2])

        with tf.variable_scope('DRN',reuse=reuse_variables):
            with tf.variable_scope('layer0'):
                self.conv1  = conv2d('0',inputs=inputs,filters=self.channels[0],kernel_size=7,strides=1,padding=3,is_training=is_trainings[-1],data_format=self.data_format)
                self.layer0 = batch_norm_relu('1',inputs=self.conv1,is_training=is_trainings[-1],data_format=self.data_format)
            self.layer1 = self.conv_layers('layer1',inputs=self.layer0,filters=self.channels[0],convs=self.layers[0],stride=1,is_training=is_trainings[-1])
            self.layer2 = self.conv_layers('layer2',inputs=self.layer1,filters=self.channels[1],convs=self.layers[1],stride=2,is_training=is_trainings[-1])
            self.layer3 = self.layer('layer3',inputs=self.layer2,block=self.block,filters=self.channels[2],blocks=self.layers[2],stride=2,is_training=is_trainings[-2])
            self.layer4 = self.layer('layer4',inputs=self.layer3,block=self.block,filters=self.channels[3],blocks=self.layers[3],stride=2,is_training=is_trainings[-3])
            self.layer5 = self.layer('layer5',inputs=self.layer4,block=self.block,filters=self.channels[4],blocks=self.layers[4],stride=2,is_training=is_trainings[-4])
            self.layer6 = self.layer('layer6',inputs=self.layer5,block=self.block,filters=self.channels[5],blocks=self.layers[5],stride=2,is_training=is_trainings[-5])
        return [self.layer6,self.layer5,self.layer4,self.layer3,self.layer2]

    def load_npy(self,data_path,session,ignore_missing=True):
        data_dict = np.load(data_path,allow_pickle=True).item()
        with tf.variable_scope('DRN', reuse=True):
            for op_name in data_dict: 
                try:
                    var = tf.get_variable(op_name)
                    session.run(var.assign(data_dict[op_name]))
                    print op_name,"loaded"
                except ValueError:
                    if not ignore_missing:
                        raise


    