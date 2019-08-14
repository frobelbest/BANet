from enc import *
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops

@ops.RegisterGradient("DepthwiseConv2dNativeBackpropInput")
def _DepthwiseConv2DNativeBackpropInputGrad(op, grad):
  return [None,
          nn_ops.depthwise_conv2d_native_backprop_filter(
              grad,
              array_ops.shape(op.inputs[1]),
              op.inputs[2],
              op.get_attr("strides"),
              op.get_attr("padding"),
              data_format=op.get_attr("data_format")),
          nn_ops.depthwise_conv2d_native(
              grad,
              op.inputs[1],
              op.get_attr("strides"),
              op.get_attr("padding"),
              data_format=op.get_attr("data_format"))]

def upsample(inputs):
    kernel=[[[[ 0.0625,  0.1875,  0.1875,  0.0625],
              [ 0.1875,  0.5625,  0.5625,  0.1875],
              [ 0.1875,  0.5625,  0.5625,  0.1875],
              [ 0.0625,  0.1875,  0.1875,  0.0625]]]]
    filters      =int(inputs.get_shape()[1])
    filter_kernel=np.tile(kernel,(filters,1,1,1))
    filter_kernel=np.transpose(filter_kernel,(2,3,0,1)).astype(np.float32)
    shape        =inputs.get_shape()
    inputs       =tf.pad(inputs,[[0,0],[0,0],[1,1],[1,1]],'SYMMETRIC')
    output       =tf.nn.depthwise_conv2d_native_backprop_input([shape[0],shape[1],2*shape[2]+4,2*shape[3]+4],tf.constant(filter_kernel),inputs,strides=(1,1,2,2),padding='SAME',data_format='NCHW')
    output       =output[:,:,2:-2,2:-2]
    return output

class DLA:

    def __init__(self,filters,is_training=False,reuse_variables=False):

        self.filters=filters
        self.is_training=is_training
        self.reuse_variables=reuse_variables

        #fixed format
        self.data_format='channels_first'
        self.layer_dict={}

    def upsample(self,inputs):
        kernel=[[[[ 0.0625,  0.1875,  0.1875,  0.0625],
                  [ 0.1875,  0.5625,  0.5625,  0.1875],
                  [ 0.1875,  0.5625,  0.5625,  0.1875],
                  [ 0.0625,  0.1875,  0.1875,  0.0625]]]]
        filters      =int(inputs.get_shape()[1])
        filter_kernel=np.tile(kernel,(filters,1,1,1))
        filter_kernel=np.transpose(filter_kernel,(2,3,0,1)).astype(np.float32)
        shape        =inputs.get_shape()
        inputs       =tf.pad(inputs,[[0,0],[0,0],[1,1],[1,1]],'SYMMETRIC')
        output       =tf.nn.depthwise_conv2d_native_backprop_input([shape[0],shape[1],2*shape[2]+4,2*shape[3]+4],tf.constant(filter_kernel),inputs,strides=(1,1,2,2),padding='SAME',data_format='NCHW')
        output       =output[:,:,2:-2,2:-2]
        return output

    def aggregation(self,input1,input2,filters,name,is_training):
        with tf.variable_scope(name):
            inputs=tf.concat([input1,input2],axis=1)
            output=conv2d('conv2d',inputs,filters,1,is_training=is_training)
            output=batch_norm_relu('batch_norm_relu',output,is_training,self.data_format)
            return output


    def depth_basis(self,layers):
        
        with tf.variable_scope('DLA',reuse=self.reuse_variables):

            for l in range(2,7):
                layer_name='layer_4_{0}'.format(l-2)
                self.layer_dict[layer_name]=layers[6-l]

            for level in range(3,-1,-1):
                for scale in range(level+1):
                    
                    node_name='node_{0}_{1}'.format(level,scale)

                    with tf.name_scope(node_name):

                        layer_name_1='layer_{0}_{1}'.format(level+1,scale)
                        input1=self.layer_dict[layer_name_1]
                        
                        layer_name_2='layer_{0}_{1}'.format(level+1,scale+1)
                        input2=self.layer_dict[layer_name_2]

                        with tf.variable_scope(node_name):
                          input2=self.projection_shortcut_norm('downsample',input2,int(input1.get_shape()[1]),stride=1,is_training=self.is_training,data_format=self.data_format,reuse_variables=self.reuse_variables)
                          input2=tf.nn.relu(input2)

                        input2=self.upsample(input2)

                        aggregate_name='aggregation_{0}_{1}'.format(level,scale)
                        aggregated=self.aggregation(input1,input2,int(input1.get_shape()[1]),aggregate_name,self.is_training)

                        layer_name='layer_{0}_{1}'.format(level,scale)
                        self.layer_dict[layer_name]=aggregated
              
            depth       = conv2d('conv2d',self.layer_dict['layer_0_0'],1,1,is_training=self.is_training,use_bias=True)
            depth       = tf.nn.relu(depth)

            _,variance  = tf.nn.moments(self.layer_dict['layer_0_0'],axes=[2,3],keep_dims=True)
            basis       = tf.rsqrt(variance+1e-3)*self.layer_dict['layer_0_0']

            return depth,basis

    def depth_basis_bundle(self,layers,batch_size,height,width):
        with tf.variable_scope('DLA',reuse=self.reuse_variables):

            for l in range(2,7):
                layer_name='layer_4_{0}'.format(l-2)
                self.layer_dict[layer_name]=layers[6-l][:batch_size,:,:,:]

            for level in range(3,-1,-1):
                for scale in range(level+1):
                    
                    node_name='node_{0}_{1}'.format(level,scale)

                    with tf.name_scope(node_name):

                        layer_name_1='layer_{0}_{1}'.format(level+1,scale)
                        input1=self.layer_dict[layer_name_1]
                        
                        layer_name_2='layer_{0}_{1}'.format(level+1,scale+1)
                        input2=self.layer_dict[layer_name_2]

                        with tf.variable_scope(node_name):
                          input2=self.projection_shortcut_norm('downsample',input2,int(input1.get_shape()[1]),stride=1,is_training=self.is_training,data_format=self.data_format,reuse_variables=self.reuse_variables)
                          input2=tf.nn.relu(input2)

                        input2=self.upsample(input2)

                        aggregate_name='aggregation_{0}_{1}'.format(level,scale)
                        aggregated=self.aggregation(input1,input2,int(input1.get_shape()[1]),aggregate_name,self.is_training)

                        layer_name='layer_{0}_{1}'.format(level,scale)
                        self.layer_dict[layer_name]=aggregated
              
            depth       = conv2d('conv2d',self.layer_dict['layer_0_0'],1,1,is_training=self.is_training,use_bias=True)
            depth       = tf.nn.relu(depth)

            _,variance  = tf.nn.moments(self.layer_dict['layer_0_0'],axes=[2,3],keep_dims=True)
            basis       = self.layer_dict['layer_0_0']
            return depth,basis

    def projection_shortcut_norm(self,name,inputs,filters,stride,is_training,data_format,reuse_variables):
      with tf.variable_scope(name,reuse=reuse_variables):
          inputs = conv2d('0',inputs=inputs,filters=filters,kernel_size=1,strides=stride,data_format=data_format,is_training=is_training)
          inputs = batch_norm('1',inputs,is_training,data_format)
          return inputs

    def pyramid(self,_layers):
        layers={}
        layers['layer1']=_layers[-1]#16
        layers['layer2']=_layers[-2]#32
        layers['layer3']=_layers[-3]#64
        layers['layer4']=_layers[-4]#128
        layers['layer5']=_layers[-5]#256

        with tf.variable_scope('PYRAMID',reuse=self.reuse_variables):
            shape =layers['layer1'].get_shape()
            nbatch=int(shape[0])
            height=int(shape[2])
            width =int(shape[3])

            with tf.variable_scope("layer4"):
                _layer5=self.upsample(layers['layer5'],'upsample')
                layer4 =self.aggregation(_layer5,layers['layer4'],384,'aggregate')#256+128=384
                layer4 =conv2d('conv2d_1',layer4,128,3,is_training=False)
                layer4 =batch_norm_selu('norm_1',layer4,False,self.data_format,active_batchnorm=True)

            with tf.variable_scope("layer3"):

                _layer4=self.upsample(layer4,'upsample')
                layer3 =self.aggregation(_layer4,layers['layer3'],192,'aggregate')#128+64=192
                layer3 =conv2d('conv2d_1',layer3,128,3,is_training=False)
                layer3 =batch_norm_selu('group_norm_1',layer3,False,self.data_format,active_batchnorm=True)

            with tf.variable_scope("layer2"):
                _layer3=self.upsample(layer3,'upsample')
                layer2=self.aggregation(_layer3,layers['layer2'],160,'aggregate')#128+32=160
                layer2=conv2d('conv2d_1',layer2,128,3,is_training=False)
                layer2=batch_norm_selu('norm_1',layer2,False,self.data_format,active_batchnorm=True)

            with tf.variable_scope("layer1"):
                _layer2=self.upsample(layer2,'upsample')
                layer1=self.aggregation(_layer2,layers['layer1'],144,'aggregate')#128+16
                layer1=conv2d('conv2d_1',layer1,128,3,is_training=False)
                layer1=batch_norm_selu('norm_1',layer1,False,self.data_format,active_batchnorm=True)

            return [tf.transpose(layer4,[0,2,3,1]),tf.transpose(layer3,[0,2,3,1]),tf.transpose(layer2,[0,2,3,1]),tf.transpose(layer1,[0,2,3,1])]  














