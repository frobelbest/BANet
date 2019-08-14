import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops

def rotation2quaternion(R,name=None):
    with ops.name_scope(name,"RotationMatrixToQuaternion"):
        diag=1.0+R[:,0,0]+R[:,1,1]+R[:,2,2]
        q0=tf.sqrt(diag)/2.0
        q1=(R[:,2,1]-R[:,1,2])/(4.0*q0)
        q2=(R[:,0,2]-R[:,2,0])/(4.0*q0)
        q3=(R[:,1,0]-R[:,0,1])/(4.0*q0)
        _q=tf.stack([q0,q1,q2,q3],axis=1)
        q=tf.nn.l2_normalize(_q,axis=1)
        return q

def AngleaAxisRotation(wx,wy,wz,name=None):
    with tf.name_scope("RotationMatrix"):
        ones =tf.ones(wx.get_shape())
        theta=tf.maximum(tf.sqrt(wx*wx+wy*wy+wz*wz),1e-6)
        wx=wx/theta
        wy=wy/theta
        wz=wz/theta

        costheta = tf.cos(theta);
        sintheta = tf.sin(theta);

        rotationMatrix=tf.stack([costheta+wx*wx*(ones-costheta),
                                 wz*sintheta+wx*wy*(ones-costheta),
                                -wy*sintheta+wx*wz*(ones-costheta),
                                 wx*wy*(ones-costheta)-wz*sintheta,
                                 costheta+wy*wy*(ones-costheta),
                                 wx*sintheta+wy*wz*(ones-costheta),
                                 wy*sintheta+wx*wz*(ones-costheta),
                                -wx*sintheta+wy*wz*(ones-costheta),
                                 costheta+wz*wz*(ones-costheta)],axis=-1)
        return tf.transpose(tf.reshape(rotationMatrix,[-1,3,3]),[0,2,1])

def VMatrix(wx,wy,wz,name=None):
    with tf.name_scope("skew"):
        theta    = tf.sqrt(wx*wx+wy*wy+wz*wz);
        costheta = tf.cos(theta);
        sintheta = tf.sin(theta);
        zero=tf.zeros(wx.get_shape())
        skew_matrix=tf.reshape(tf.stack([zero,-wz,wy,wz,zero,-wx,-wy,wx,zero]),[-1,3,3])
        return tf.eye(3,3,batch_shape=[1])+((1-costheta)/tf.square(theta))*skew_matrix+((theta-sintheta)/tf.pow(theta,3))*tf.matmul(skew_matrix,skew_matrix)


def CameraJacobianMatrix(x,y,Z,fx,fy,name=None):
    with tf.name_scope(name):
        xy_z2= tf.multiply(x,y)
        xx_z2= -1.0-tf.square(x)
        x_z2 = tf.div(x,Z)
        yy_z2= 1.0+tf.square(y)
        y_z2 = tf.div(y,Z)
        _Z   = 1.0/Z
        zeros= tf.zeros(xy_z2.get_shape())
        dx   = tf.expand_dims(fx,axis=-1)*tf.stack([xy_z2, xx_z2, y,-_Z,zeros,x_z2],axis=2)
        dy   = tf.expand_dims(fy,axis=-1)*tf.stack([yy_z2,-xy_z2,-x,zeros,-_Z,y_z2],axis=2)
        jacobianMatrixGeometry=-tf.stack([dx,dy],axis=2)
    return jacobianMatrixGeometry

def DepthJacobianMatrix(rx,ry,rz,x,y,Z,fx,fy,name=None):
    with tf.name_scope(name):
        rx = tf.squeeze(rx,axis=1)
        ry = tf.squeeze(ry,axis=1)
        rz = tf.squeeze(rz,axis=1)

        dx = fx*tf.div(rx-rz*x,Z)
        dy = fy*tf.div(ry-rz*y,Z)
        # print fx.get_shape(),rx.get_shape(),rz.get_shape(),x.get_shape(),Z.get_shape()
        # print dx.get_shape()
        jacobianMatrixGeometry=tf.stack([dx,dy],axis=2)
    return jacobianMatrixGeometry

util=tf.load_op_library('./utils.so')
equation_construction=util.equation_construction
equation_construction_grad=util.equation_construction_grad
@ops.RegisterGradient("EquationConstruction")
def equation_construction_gradient(op,*grad):
    jacobian_grad,gradient_grad,difference_grad=equation_construction_grad(op.inputs[0],op.inputs[1],op.inputs[2],grad[0],grad[1])
    return jacobian_grad,gradient_grad,difference_grad



class BundleNet:

    def __init__(self,is_training=True,reuse_variables=None):
        self.is_training    = is_training
        self.reuse_variables=reuse_variables

    def grad_fixed(self,input,name=None):
        with tf.name_scope(name):
            shape = input.get_shape()
            height= int(shape[1])
            width = int(shape[2])
            paded = tf.pad(input,[[0,0],[1,1],[1,1],[0,0]],mode='REFLECT')
            gradx = 0.5*(paded[:,1:height+1,2:width+2,:]-paded[:,1:height+1,0:width,:])
            grady = 0.5*(paded[:,2:height+2,1:width+1,:]-paded[:,0:height,1:width+1,:])
            return tf.concat([gradx,grady],axis=-1)

    def conv1d(self, x, num_out_layers,name,activation=tf.nn.elu):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            num_in_layers=int(x.get_shape().as_list()[-1])
            filters = tf.get_variable(shape=[1,num_in_layers,num_out_layers],initializer=tf.keras.initializers.he_normal(),name=name+"_filters")
            biases  = tf.get_variable(shape=[num_out_layers],name=name + "_biases",initializer=tf.zeros_initializer())
            conv = tf.nn.conv1d(x,filters,1,padding='SAME')
            bias = tf.nn.bias_add(conv,biases)
            relu = activation(bias)
        return relu

    def computeCoordinates(self,points2d,fx,fy,ox,oy):
        nbatch =int(points2d.get_shape()[0])
        npixels=int(points2d.get_shape()[1])
        x   =tf.expand_dims((points2d[:,:,0]-ox)/fx,axis=1)
        y   =tf.expand_dims((points2d[:,:,1]-oy)/fy,axis=1)
        ones=tf.ones([nbatch,1,npixels],dtype=tf.float32)
        p   =tf.concat([x,y,ones],axis=1)
        p   =tf.nn.l2_normalize(p,axis=1)
        return p

    def CameraIteration(self,conv1,conv2,fx,fy,ox,oy,p,D,R,T,l2_regularizer_base=None,level=None):
        N,height,width,_=conv2.get_shape()
        with tf.name_scope("initialization"):

            conv1_shape=conv1.get_shape()
            nbatch    =int(conv1_shape[0])
            npixels   =int(conv1_shape[1])
            
            nchannels1=int(conv1_shape[2])
            nchannels2=int(2*nchannels1)
            nchannels3=nchannels1+nchannels2

        with tf.name_scope("warp_compute"): 

            Rp = tf.matmul(R,p)
            RP = tf.multiply(Rp,tf.tile(tf.transpose(D,[0,2,1]),[1,3,1]))
            RPT= tf.add(RP,tf.tile(T,[1,1,npixels]))

            Z  = RPT[:,2,:]
            X  = RPT[:,0,:]
            Y  = RPT[:,1,:]

            x  = tf.div(X,Z)
            y  = tf.div(Y,Z)

            px=fx*x+ox
            py=fy*y+oy

        with tf.name_scope("warp_conv"): 

            with tf.name_scope("weighting"):

                _conv2 = tf.contrib.resampler.resampler(conv2,tf.stack([px,py],axis=-1))
                _mask  = tf.to_float(tf.logical_not(tf.reduce_any(tf.stack([px<0,px>float(int(width)-1),py<0,py>float(int(height)-1)],axis=-1),axis=-1,keepdims=False)))
                _mask  = tf.expand_dims(_mask,axis=-1)
                mask   = tf.expand_dims(_mask,axis=-1)
                _diff  = tf.expand_dims(conv1-_conv2[:,:,0:nchannels1],-1)
                _gradx = tf.expand_dims(_conv2[:,:,nchannels1:nchannels2],-1)
                _grady = tf.expand_dims(_conv2[:,:,nchannels2:nchannels3],-1)

                diff   = tf.matmul(_diff,mask)
                grad   = tf.concat([tf.matmul(_gradx,mask),tf.matmul(_grady,mask)],-1)

        with tf.name_scope("lambda_prediction"):

            avg_residual     =tf.reduce_mean(tf.abs(tf.squeeze(diff,axis=-1)),axis=1,keep_dims=True)
            lambda_conv1     =self.conv1d(avg_residual,2*nchannels1,name="lambda_"+level+"_1",activation=tf.nn.selu)
            lambda_conv2     =self.conv1d(lambda_conv1,4*nchannels1,name="lambda_"+level+"_2",activation=tf.nn.selu)
            lambda_conv3     =self.conv1d(lambda_conv2,2*nchannels1,name="lambda_"+level+"_3",activation=tf.nn.selu)
            lambda_conv4     =self.conv1d(lambda_conv3,nchannels1,name="lambda_"+level+"_4",activation=tf.nn.selu)
            lambda_conv5     =self.conv1d(lambda_conv4,1,name="lambda_"+level+"_5",activation=tf.nn.tanh)
            lambda_prediction=tf.pow(tf.norm(avg_residual,axis=-1,keepdims=True),2.0+lambda_conv5)

        with tf.name_scope("FirstEstimation"):

            with tf.name_scope("Solve"):

                jacobianMatrixGeometry=CameraJacobianMatrix(x,y,Z,fx,fy)
                AtA,Atb=equation_construction(jacobian=jacobianMatrixGeometry,gradient=grad,difference=diff)
                diag   = tf.matrix_diag_part(AtA)
                AtA    = AtA+tf.matrix_diag(tf.squeeze(tf.matmul(tf.expand_dims(diag,axis=-1)+1e-5,lambda_prediction),axis=-1))
                motion = tf.matrix_solve(AtA,Atb)
            with tf.name_scope("update"):
                wx,wy,wz,tx,ty,tz=tf.split(motion,num_or_size_splits=6,axis=1)
                dr=AngleaAxisRotation(wx,wy,wz)
                dv=VMatrix(wx,wy,wz)
                dt=tf.concat([tx,ty,tz],axis=1)
                updatedR=tf.matmul(dr,R)
                updatedT=tf.add(tf.matmul(dv,dt),tf.matmul(dr,T))
        return updatedR,updatedT

    def BundleIteration(self,conv1,conv2,fx,fy,ox,oy,p,D,B,R,T,W,l2_regularizer_base=None,level=None):

        N,height,width,_=conv2.get_shape()
        with tf.name_scope("initialization"):

            conv1_shape=conv1.get_shape()
            nbatch    =int(conv1_shape[0])
            npixels   =int(conv1_shape[1])
            
            nchannels1=int(conv1_shape[2])
            nchannels2=int(2*nchannels1)
            nchannels3=nchannels1+nchannels2

        with tf.name_scope("warp_compute"): 
            #print B.get_shape(),W.get_shape()
            D  = D+tf.matmul(B,W)
            Rp = tf.matmul(R,p)
            rx,ry,rz= tf.split(Rp,3,axis=1)
            RP = tf.multiply(Rp,tf.tile(tf.transpose(D,[0,2,1]),[1,3,1]))
            

            RPT= tf.add(RP,tf.tile(T,[1,1,npixels]))

            Z  = RPT[:,2,:]
            X  = RPT[:,0,:]
            Y  = RPT[:,1,:]

            x  = tf.div(X,Z)
            y  = tf.div(Y,Z)

            px=fx*x+ox
            py=fy*y+oy

        with tf.name_scope("warp_conv"): 

            with tf.name_scope("weighting"):

                _conv2 = tf.contrib.resampler.resampler(conv2,tf.stack([px,py],axis=-1))
                _mask  = tf.to_float(tf.logical_not(tf.reduce_any(tf.stack([px<0,px>float(int(width)-1),py<0,py>float(int(height)-1)],axis=-1),axis=-1,keepdims=False)))
                _mask  = tf.expand_dims(_mask,axis=-1)
                mask   = tf.expand_dims(_mask,axis=-1)
                _diff  = tf.expand_dims(conv1-_conv2[:,:,0:nchannels1],-1)
                _gradx = tf.expand_dims(_conv2[:,:,nchannels1:nchannels2],-1)
                _grady = tf.expand_dims(_conv2[:,:,nchannels2:nchannels3],-1)

                diff   = tf.matmul(_diff,mask)
                grad   = tf.concat([tf.matmul(_gradx,mask),tf.matmul(_grady,mask)],-1)

        with tf.name_scope("lambda_prediction"):

            avg_residual     =tf.reduce_mean(tf.abs(tf.squeeze(diff,axis=-1)),axis=1,keep_dims=True)
            lambda_conv1     =self.conv1d(avg_residual,2*nchannels1,name="lambda_"+level+"_1",activation=tf.nn.selu)
            lambda_conv2     =self.conv1d(lambda_conv1,4*nchannels1,name="lambda_"+level+"_2",activation=tf.nn.selu)
            lambda_conv3     =self.conv1d(lambda_conv2,2*nchannels1,name="lambda_"+level+"_3",activation=tf.nn.selu)
            lambda_conv4     =self.conv1d(lambda_conv3,nchannels1,name="lambda_"+level+"_4",activation=tf.nn.selu)
            lambda_conv5     =self.conv1d(lambda_conv4,1,name="lambda_"+level+"_5",activation=tf.nn.tanh)
            lambda_prediction=tf.pow(tf.norm(avg_residual,axis=-1,keepdims=True),2.0+lambda_conv5)
            print "lambda_shape",lambda_prediction.get_shape()
            
            if l2_regularizer_base is not None:
                lambda_prediction=l2_regularizer_base*lambda_prediction

        with tf.name_scope("BundleEstimation"):

            with tf.name_scope("Solve"):

                jacobianMatrixCamera  = CameraJacobianMatrix(x,y,Z,fx,fy)
                jacobianMatrixDepth   = tf.matmul(tf.expand_dims(DepthJacobianMatrix(rx,ry,rz,x,y,Z,fx,fy),axis=-1),tf.expand_dims(B,axis=-2))
                jacobianMatrixGeometry= tf.concat([jacobianMatrixCamera,jacobianMatrixDepth],axis=-1)

                AtA,Atb  = equation_construction(jacobian=jacobianMatrixGeometry,gradient=grad,difference=diff)
                diag     = tf.matrix_diag_part(AtA)

                AtA      = AtA+tf.matrix_diag(tf.concat([diag[:,:-1]+1e-5,tf.zeros([nbatch,1])],axis=-1)*tf.squeeze(lambda_prediction,axis=-1))
                solution = tf.matrix_solve(AtA,Atb)

            with tf.name_scope("update"):
                wx,wy,wz,tx,ty,tz=tf.split(solution[:,:6,:],num_or_size_splits=6,axis=1)
                dr=AngleaAxisRotation(wx,wy,wz)
                dv=VMatrix(wx,wy,wz)
                dt=tf.concat([tx,ty,tz],axis=1)
                updatedR=tf.matmul(dr,R)
                updatedT=tf.add(tf.matmul(dv,dt),tf.matmul(dr,T))
                updatedW=tf.add(W,solution[:,6:,:])

        return updatedR,updatedT,updatedW

    def CameraResize(self,intrisic,layers,points,_depths,reuse_variables=False):

        with tf.name_scope("prepare"):

            self.reuse_variables=reuse_variables
            x,y=tf.split(points,axis=-1,num_or_size_splits=2)
            x=320*(x-4)/(312)
            y=256*(y-4)/(232)
            _points=tf.concat([x,y],axis=-1)
            depths=tf.stop_gradient(_depths)
            d=tf.contrib.resampler.resampler(depths,_points/2)

        with tf.name_scope("feature_pyramid"):

            shape  =layers[-1].get_shape()
            nbatch =int(shape[0])
            npixels=int(points.get_shape()[1])

            self.fx=40.0*tf.tile(intrisic[:,0],[1,npixels])/39.0
            self.fy=32.0*tf.tile(intrisic[:,1],[1,npixels])/29.0

            self.ox=(40.0*tf.tile(intrisic[:,2],[1,npixels])/39.0)-(160.0/39.0)
            self.oy=(32.0*tf.tile(intrisic[:,3],[1,npixels])/29.0)-(128.0/29.0)

            p      = self.computeCoordinates(_points,self.fx,self.fy,self.ox,self.oy)

        rotations=[tf.eye(3,3,batch_shape=[nbatch])]
        translations=[tf.zeros([nbatch,3,1])]

        for level in range(0,4):
            
            scale  = 2**(3-level)
            fx     = self.fx/scale
            fy     = self.fy/scale
            ox     = self.ox/scale
            oy     = self.oy/scale
            points1= _points/scale

            # layer1=tf.contrib.resampler.resampler(layers[level-1],points1)
            # layer2=tf.concat([layers[level-1][nbatch/2:nbatch,:,:,:],layers[level-1][0:nbatch/2,:,:,:]],axis=0)
            layer1=tf.contrib.resampler.resampler(layers[level],points1)
            layer2=tf.concat([layers[level][nbatch/2:nbatch,:,:,:],layers[level][0:nbatch/2,:,:,:]],axis=0)
            
            grad2 =self.grad_fixed(layer2)
            layer2=tf.concat([layer2,grad2],axis=-1)
            for iters in range(1):
                R,T=self.CameraIteration(layer1,layer2,fx,fy,ox,oy,p,d,rotations[-1],translations[-1],1.0,str(level))
                rotations.append(R)
                translations.append(T)
        return rotations[1:],translations[1:]


    def BundleResize(self,intrisic,layers,points,basis,init_depth,init_rotation=None,init_translation=None,reuse_variables=False):

        with tf.name_scope("prepare"):

            self.reuse_variables=reuse_variables
            x,y=tf.split(points,axis=-1,num_or_size_splits=2)
            x=320*(x-4)/(312)
            y=256*(y-4)/(232)
            _points=tf.concat([x,y],axis=-1)
            depths=tf.stop_gradient(init_depth)

            d=tf.contrib.resampler.resampler(depths,_points/2)
            b=tf.contrib.resampler.resampler(basis,_points/2)

        with tf.name_scope("feature_pyramid"):

            shape  =layers[-1].get_shape()
            nbatch =int(shape[0])
            npixels=int(points.get_shape()[1])
            nbasis =int(basis.get_shape()[-1])

            #adjust camera parameter if the image is cropped
            self.fx=40.0*tf.tile(intrisic[:,0],[1,npixels])/39.0
            self.fy=32.0*tf.tile(intrisic[:,1],[1,npixels])/29.0
            self.ox=(40.0*tf.tile(intrisic[:,2],[1,npixels])/39.0)-(160.0/39.0)
            self.oy=(32.0*tf.tile(intrisic[:,3],[1,npixels])/29.0)-(128.0/29.0)
            p      = self.computeCoordinates(_points,self.fx,self.fy,self.ox,self.oy)

        output_rotations   =[]
        output_translations=[]
        output_depths      =[]

        if init_rotation==None:
            R=tf.eye(3,3,batch_shape=[nbatch]) 
        else: 
            R=init_rotation

        if init_translation==None:
            T=tf.zeros([nbatch,3,1])
        else:
            T=init_translation

        W=tf.zeros([nbatch,nbasis,1])

        for level in range(2,4):

            scale  = 2**(3-level)
            fx     = self.fx/scale
            fy     = self.fy/scale
            ox     = self.ox/scale
            oy     = self.oy/scale
            points1= _points/scale

            layer1=tf.contrib.resampler.resampler(layers[level],points1)
            layer2=tf.concat([layers[level][nbatch/2:nbatch,:,:,:],layers[level][0:nbatch/2,:,:,:]],axis=0)
            
            grad2 =self.grad_fixed(layer2)
            layer2=tf.concat([layer2,grad2],axis=-1)

            for iters in range(1):
                
                R,T,W=self.BundleIteration(layer1,layer2,fx,fy,ox,oy,p,d,b,R,T,W,1000.0,str(level))

                output_rotations.append(R)
                output_translations.append(T)
                output_depths.append(init_depth+tf.reshape(tf.matmul(tf.reshape(basis,[nbatch,-1,nbasis]),W),[nbatch,256//2,320//2,1]))

        return output_rotations,output_translations,output_depths

    def lossR(self,predQ,gtQ):
        with tf.name_scope("accuracyR_Quaterninon"):
            accR=tf.losses.cosine_distance(predQ,gtQ,axis=1)
            return accR

    def lossT(self,predT,gtT):
        with tf.name_scope("accuracyT_Angle"):
            angT=tf.losses.cosine_distance(tf.nn.l2_normalize(predT,axis=1),tf.nn.l2_normalize(gtT,axis=1),axis=1)
            return angT

    def lossT(self,predT,gtT):
        with tf.name_scope("accuracyT_Euclidean"):
            return tf.reduce_mean(tf.abs(predT-gtT))

    def lossF(self,intrisic,depth,mask,predR,predT,gtR,gtT):
        
        def flow(p,fx,fy,ox,oy,depth,rotation,translation,nbatch,npixels):
            Rp = tf.matmul(rotation,p)
            D  = tf.reshape(depth,[nbatch,1,npixels])
            RP = tf.multiply(Rp,tf.tile(D,[1,3,1]))
            RPT= tf.add(RP,tf.tile(translation,[1,1,npixels]))
            X = RPT[:,0,:]
            Y = RPT[:,1,:]
            Z = RPT[:,2,:]
            x  = tf.div(X,Z)
            y  = tf.div(Y,Z)
            px=fx*x+ox
            py=fy*y+oy
            return px,py

        with tf.name_scope("accuracyF"):

            predT=tf.expand_dims(predT,axis=-1)
            gtT  =tf.expand_dims(gtT,axis=-1)

            shape  =depth.get_shape()
            nbatch =int(shape[0])
            height =int(shape[1])
            width  =int(shape[2])
            npixels=height*width
            mask   =tf.reshape(mask,[nbatch,npixels])

            fx=40.0*tf.tile(intrisic[:,0],[1,npixels])/39.0
            fy=32.0*tf.tile(intrisic[:,1],[1,npixels])/29.0

            ox=(40.0*tf.tile(intrisic[:,2],[1,npixels])/39.0)-(160.0/39.0)
            oy=(32.0*tf.tile(intrisic[:,3],[1,npixels])/29.0)-(128.0/29.0)

            x,y=tf.meshgrid(tf.range(0,width),tf.range(0,height))
            x  =tf.reshape(tf.cast(x,tf.float32),[height*width,1])
            y  =tf.reshape(tf.cast(y,tf.float32),[height*width,1])
            p  =tf.concat([x,y],axis=-1)
            p  =tf.tile(tf.expand_dims(p,axis=0),[nbatch,1,1])
            p  =self.computeCoordinates(p,fx,fy,ox,oy)

            flowx_pred,flowy_pred= flow(p,fx,fy,ox,oy,depth,predR,predT,nbatch,npixels)
            flowx_gt,flowy_gt    = flow(p,fx,fy,ox,oy,depth,gtR,gtT,nbatch,npixels)

            validcount=tf.reduce_sum(mask)
            totalcount=tf.to_float(npixels*nbatch)

            return (totalcount/validcount)*(tf.reduce_mean(tf.multiply(tf.abs(flowx_pred-flowx_gt),mask))/width
                                           +tf.reduce_mean(tf.multiply(tf.abs(flowy_pred-flowy_gt),mask))/width)


