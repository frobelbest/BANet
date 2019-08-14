import feat as Features
import tensorflow as tf
import utils_python as utils

early_termination=True
angle_change=0.002*(3.14/180.0)
translation_change=0.0002
residual_ratio=1.0
qr=True

util=tf.load_op_library('./utils.so')
equation_construction=util.equation_construction
jacobian_construction=util.jacobian_construction

class Tracker:

    def grad_fixed(self,input,name=None):
        with tf.name_scope(name):
            shape = input.get_shape()
            height= int(shape[1])
            width = int(shape[2])
            paded = tf.pad(input,[[0,0],[1,1],[1,1],[0,0]],mode='REFLECT')
            gradx = 0.5*(paded[:,1:height+1,2:width+2,:]-paded[:,1:height+1,0:width,:])
            grady = 0.5*(paded[:,2:height+2,1:width+1,:]-paded[:,0:height,1:width+1,:])
            return tf.concat([gradx,grady],axis=-1)

    def computeCoordinates(self,points2d,fx,fy,ox,oy):
        nbatch =int(points2d.get_shape()[0])
        npixels=int(points2d.get_shape()[1])
        x   =tf.expand_dims((points2d[:,:,0]-ox)/fx,axis=1)
        y   =tf.expand_dims((points2d[:,:,1]-oy)/fy,axis=1)
        ones=tf.ones([nbatch,1,npixels],dtype=tf.float32)
        p   =tf.concat([x,y,ones],axis=1)
        return p

    def CameraJacobianMatrix(self,x,y,Z,fx,fy,name=None):
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
            jacobianMatrixGeometry=tf.stack([dx,dy],axis=2)
        return jacobianMatrixGeometry


    def VMatrix(self,wx,wy,wz,name=None):
        with tf.name_scope("skew"):
            theta    =tf.sqrt(wx*wx+wy*wy+wz*wz);
            costheta = tf.cos(theta);
            sintheta = tf.sin(theta);
            zero=tf.zeros(wx.get_shape())
            skew_matrix=tf.reshape(tf.stack([zero,-wz,wy,wz,zero,-wx,-wy,wx,zero]),[-1,3,3])
            return tf.eye(3,3,batch_shape=[1])+((1-costheta)/tf.square(theta))*skew_matrix+((theta-sintheta)/tf.pow(theta,3))*tf.matmul(skew_matrix,skew_matrix)

    def AngleaAxisRotation(self,wx,wy,wz,name=None):
        with tf.name_scope("RotationMatrix"):
            ones=tf.ones(wx.get_shape())
            theta=tf.sqrt(wx*wx+wy*wy+wz*wz);
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



        
    def trackTF(self,intrisic,layers,points,d,initR,initT,level_iters):

        with tf.name_scope("feature_pyramid"):

            shape  =layers[-1].get_shape()
            nbatch =int(shape[0])/2
            npixels=int(points.get_shape()[1])

            self.fx=tf.tile(intrisic[:,0],[1,npixels])
            self.fy=tf.tile(intrisic[:,1],[1,npixels])

            self.ox=tf.tile(intrisic[:,2],[1,npixels])
            self.oy=tf.tile(intrisic[:,3],[1,npixels])
            p      = self.computeCoordinates(points,self.fx,self.fy,self.ox,self.oy)


        rotations=[initR]
        translations=[initT]
        R=initR
        T=initT

        for level in range(1,4):
            
            scale  = 2**(3-level)
            fx     = self.fx/scale
            fy     = self.fy/scale
            ox     = self.ox/scale
            oy     = self.oy/scale
            points1= points/scale

            layer1=utils.interpolate2d2(layers[level-1][0:1,:,:,:],points1)
            layer2=layers[level-1][1:2,:,:,:]
            grad2 =self.grad_fixed(layer2)
            layer2=tf.concat([layer2,grad2],axis=-1)

            if not early_termination:
                for iters in range(level_iters[level-1]):
                    R,T,ratio=self.CameraIteration(layer1,layer2,fx,fy,ox,oy,p,d,rotations[-1],translations[-1])
                    rotations.append(R)
                    translations.append(T)
            else:

                iters =0
                update_w=1.0
                update_t=1.0
                ratio=1.0

                def cond(iters,update_w,update_t,*argv):
                    return tf.reduce_all([tf.less(iters,level_iters[level-1]),tf.less(angle_change,update_w),tf.less(translation_change,update_t)])

                def body(iters,update_w,update_t,R,T,ratio):
                    R,T,update_w,update_t,ratio=self.CameraIteration2(layer1,layer2,fx,fy,ox,oy,p,d,R,T,str(level))
                    iters=iters+1
                    return iters,update_w,update_t,R,T,ratio

                _,_,_,R,T,ratio=tf.while_loop(cond,body,[iters,update_w,update_t,R,T,ratio],back_prop=False, parallel_iterations=1)

        if not early_termination:
            return rotations[1:],translations[1:],ratio
        else:
            return R,T,ratio


    def CameraIteration(self,conv1,conv2,fx,fy,ox,oy,p,D,R,T):

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
                _conv2,_mask=utils.interpolate2d(conv2,px,py)
                mask   = tf.expand_dims(_mask,axis=-1)
                _diff  = tf.expand_dims(_conv2[:,:,0:nchannels1]-conv1,-1)
                _gradx = tf.expand_dims(_conv2[:,:,nchannels1:nchannels2],-1)
                _grady = tf.expand_dims(_conv2[:,:,nchannels2:nchannels3],-1)

                diff   = tf.matmul(_diff,mask)
                grad   = tf.concat([tf.matmul(_gradx,mask),tf.matmul(_grady,mask)],-1)

        with tf.name_scope("lambda_prediction"):

            avg_residual     =tf.reduce_mean(tf.abs(tf.squeeze(diff,axis=-1)),axis=1,keep_dims=True)
            lambda_prediction=tf.pow(tf.norm(avg_residual,axis=-1,keepdims=True),2.0)

        with tf.name_scope("FirstEstimation"):

            with tf.name_scope("Solve"):

                jacobianMatrixGeometry=self.CameraJacobianMatrix(x,y,Z,fx,fy)
                AtA=tf.reduce_sum(tf.matmul(jacobianMatrixGeometry,tf.matmul(tf.matmul(grad,grad,transpose_a=True),jacobianMatrixGeometry),transpose_a=True),axis=1)
                Atb=tf.reduce_sum(tf.matmul(jacobianMatrixGeometry,tf.matmul(grad,diff,transpose_a=True),transpose_a=True),axis=1)

                diag   = tf.matrix_diag_part(AtA)
                AtA    = AtA+tf.matrix_diag(tf.squeeze(tf.matmul(tf.expand_dims(diag,axis=-1)+1e-5,lambda_prediction),axis=-1))
                if not qr:
                    motion = tf.matmul(tf.matrix_inverse(AtA),Atb)
                else:
                    q_full,r_full=tf.qr(AtA,full_matrices=True)
                    motion=tf.linalg.solve(r_full,tf.matmul(q_full,Atb,transpose_a=True))

            with tf.name_scope("Update"):
                wx,wy,wz,tx,ty,tz=tf.unstack(motion,num=6,axis=1)
                dr=self.AngleaAxisRotation(wx,wy,wz)
                dt=tf.stack([tx,ty,tz],axis=1)
                updatedR=tf.matmul(dr,R)
                updatedT=tf.add(dt,tf.matmul(dr,T))
        return updatedR,updatedT,tf.reduce_sum(mask)/npixels

    def conv1d(self, x, num_out_layers,name,activation=tf.nn.elu):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
            num_in_layers=int(x.get_shape().as_list()[-1])
            filters = tf.get_variable(shape=[1,num_in_layers,num_out_layers],initializer=tf.keras.initializers.he_normal(),name=name+"_filters")
            biases  = tf.get_variable(shape=[num_out_layers],name=name + "_biases",initializer=tf.zeros_initializer())
            conv = tf.nn.conv1d(x,filters,1,padding='SAME')
            bias = tf.nn.bias_add(conv,biases)
            relu = activation(bias)
        return relu

    def CameraIteration2(self,conv1,conv2,fx,fy,ox,oy,p,D,R,T,level):

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
                _conv2,_mask=utils.interpolate2d(conv2,px,py)
                num_valid=npixels/tf.reduce_sum(_mask,axis=1,keepdims=True)
                mask   = tf.expand_dims(_mask,axis=-1)
                _diff  = tf.expand_dims(_conv2[:,:,0:nchannels1]-conv1,-1)
                _gradx = tf.expand_dims(_conv2[:,:,nchannels1:nchannels2],-1)
                _grady = tf.expand_dims(_conv2[:,:,nchannels2:nchannels3],-1)

                diff   = tf.matmul(_diff,mask)
                grad   = tf.concat([tf.matmul(_gradx,mask),tf.matmul(_grady,mask)],-1)

        with tf.name_scope("lambda_prediction"):

            avg_residual     =num_valid*tf.reduce_mean(tf.abs(tf.squeeze(diff,axis=-1)),axis=1,keep_dims=True)
            lambda_conv1     =self.conv1d(avg_residual,2*nchannels1,name="lambda_"+level+"_1",activation=tf.nn.selu)
            lambda_conv2     =self.conv1d(lambda_conv1,4*nchannels1,name="lambda_"+level+"_2",activation=tf.nn.selu)
            lambda_conv3     =self.conv1d(lambda_conv2,2*nchannels1,name="lambda_"+level+"_3",activation=tf.nn.selu)
            lambda_conv4     =self.conv1d(lambda_conv3,nchannels1,name="lambda_"+level+"_4",activation=tf.nn.selu)
            lambda_conv5     =self.conv1d(lambda_conv4,1,name="lambda_"+level+"_5",activation=tf.nn.tanh)
            lambda_prediction=tf.pow(tf.norm(avg_residual,axis=-1,keepdims=True),1.0+lambda_conv5)
            avg_residual     =tf.reduce_mean(avg_residual)

        with tf.name_scope("FirstEstimation"):

            with tf.name_scope("Solve"):

                jacobianMatrixGeometry=self.CameraJacobianMatrix(x,y,Z,fx,fy)
                AtA=tf.reduce_sum(tf.matmul(jacobianMatrixGeometry,tf.matmul(tf.matmul(grad,grad,transpose_a=True),jacobianMatrixGeometry),transpose_a=True),axis=1)
                Atb=tf.reduce_sum(tf.matmul(jacobianMatrixGeometry,tf.matmul(grad,diff,transpose_a=True),transpose_a=True),axis=1)

                diag   = tf.matrix_diag_part(AtA)
                AtA    = AtA+tf.matrix_diag(tf.squeeze(tf.matmul(tf.expand_dims(diag,axis=-1)+1e-5,lambda_prediction),axis=-1))
                #motion = tf.matmul(tf.matrix_inverse(AtA),Atb)

                if not qr:
                    motion = tf.matmul(tf.matrix_inverse(AtA),Atb)
                else:
                    q_full,r_full=tf.qr(AtA,full_matrices=True)
                    motion=tf.linalg.solve(r_full,tf.matmul(q_full,Atb,transpose_a=True))

            with tf.name_scope("Update"):

                wx,wy,wz,tx,ty,tz=tf.unstack(motion,num=6,axis=1)
                dr=self.AngleaAxisRotation(wx,wy,wz)
                dv=self.VMatrix(wx,wy,wz)
                dt=tf.stack([tx,ty,tz],axis=1)
                updatedR=tf.matmul(dr,R)
                updatedT=tf.add(tf.matmul(dv,dt),tf.matmul(dr,T))

            with tf.name_scope("CheckUpdate"):
                
                Rp = tf.matmul(updatedR,p)
                RP = tf.multiply(Rp,tf.tile(tf.transpose(D,[0,2,1]),[1,3,1]))
                RPT= tf.add(RP,tf.tile(updatedT,[1,1,npixels]))

                Z  = RPT[:,2,:]
                X  = RPT[:,0,:]
                Y  = RPT[:,1,:]

                x  = tf.div(X,Z)
                y  = tf.div(Y,Z)

                px=fx*x+ox
                py=fy*y+oy

                _conv2,_mask=utils.interpolate2d(conv2,px,py)
                _num_valid   =npixels/tf.reduce_sum(_mask,axis=1,keepdims=True)
                _diff        =_mask*(_conv2[:,:,0:nchannels1]-conv1)
                _avg_residual= _num_valid*tf.reduce_mean(tf.abs(_diff),axis=1,keep_dims=True)
                _avg_residual=tf.reduce_mean(_avg_residual)

                motion=tf.squeeze(motion)
                # assert_op = tf.Assert(False,[motion[3:]],summarize=10000)
                # with tf.control_dependencies([assert_op]):
                #     motion=tf.identity(motion)



                # def return_origin():
                #     return R,T,tf.to_float(0.0),tf.to_float(0.0),tf.squeeze(num_valid)

                # def return_update():
                #     global motion
                #     assert_op = tf.Assert(False,[tf.norm(motion[:3]),tf.norm(motion[3:])],summarize=10000)
                #     with tf.control_dependencies([assert_op]):
                #         motion=tf.identity(motion)
                #     return updatedR,updatedT,tf.norm(motion[:3]),tf.norm(motion[3:]),tf.squeeze(num_valid)

        return tf.cond(tf.less(_avg_residual,residual_ratio*avg_residual),
                               lambda:(updatedR,updatedT,tf.norm(motion[:3]),tf.norm(motion[3:]),tf.squeeze(num_valid)),
                               lambda:(R,T,tf.to_float(0.0),tf.to_float(0.0),tf.squeeze(num_valid)))


    def CameraIteration3(self,conv1,conv2,intrinsic,P,R,T,level):

        conv1_shape=conv1.get_shape()
        nbatch    =int(conv1_shape[0])
        npixels   =int(conv1_shape[1])
        
        nchannels1=int(conv1_shape[2])
        nchannels2=int(2*nchannels1)
        nchannels3=nchannels1+nchannels2

        with tf.name_scope("jacobian_and_projection"):
        	jacobianMatrixGeometry,projection=jacobian_construction(P,intrinsics,R,T)

        with tf.name_scope("weighting"):

            _conv2,_mask=utils.interpolate2d(conv2,projection[:,:,0],projection[:,:,1])
            num_valid=npixels/tf.reduce_sum(_mask,axis=1,keepdims=True)
            mask   = tf.expand_dims(_mask,axis=-1)
            _diff  = tf.expand_dims(_conv2[:,:,0:nchannels1]-conv1,-1)
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
            lambda_prediction=tf.pow(tf.norm(avg_residual,axis=-1,keepdims=True),1.0+lambda_conv5)
            avg_residual     =tf.reduce_mean(avg_residual)

        with tf.name_scope("FirstEstimation"):

            with tf.name_scope("Solve"):

                AtA,Atb= equation_construction(jacobian=jacobianMatrixGeometry,gradient=grad,difference=diff)
                diag   = tf.matrix_diag_part(AtA)
                AtA    = AtA+tf.matrix_diag(tf.squeeze(tf.matmul(tf.expand_dims(diag,axis=-1)+1e-5,lambda_prediction),axis=-1))

                if not qr:
                    motion = tf.matmul(tf.matrix_inverse(AtA),Atb)
                else:
                    q_full,r_full=tf.qr(AtA,full_matrices=True)
                    motion=tf.linalg.solve(r_full,tf.matmul(q_full,Atb,transpose_a=True))

            with tf.name_scope("Update"):

                wx,wy,wz,tx,ty,tz=tf.unstack(motion,num=6,axis=1)
                dr=self.AngleaAxisRotation(wx,wy,wz)
                dv=self.VMatrix(wx,wy,wz)
                dt=tf.stack([tx,ty,tz],axis=1)
                updatedR=tf.matmul(dr,R)
                updatedT=tf.add(tf.matmul(dv,dt),tf.matmul(dr,T))

            with tf.name_scope("CheckUpdate"):
                
                Rp = tf.matmul(updatedR,p)
                RP = tf.multiply(Rp,tf.tile(tf.transpose(D,[0,2,1]),[1,3,1]))
                RPT= tf.add(RP,tf.tile(updatedT,[1,1,npixels]))

                Z  = RPT[:,2,:]
                X  = RPT[:,0,:]
                Y  = RPT[:,1,:]

                x  = tf.div(X,Z)
                y  = tf.div(Y,Z)

                px=fx*x+ox
                py=fy*y+oy

                _conv2,_mask =utils.interpolate2d(conv2,px,py)
                _num_valid   =npixels/tf.reduce_sum(_mask,axis=1,keepdims=True)
                _diff        =_mask*(_conv2[:,:,0:nchannels1]-conv1)
                _avg_residual= _num_valid*tf.reduce_mean(tf.abs(_diff),axis=1,keep_dims=True)
                _avg_residual=tf.reduce_mean(_avg_residual)
                motion=tf.squeeze(motion)

        return tf.cond(tf.less(_avg_residual,residual_ratio*avg_residual),
                               lambda:(updatedR,updatedT,tf.norm(motion[:3]),tf.norm(motion[3:]),tf.squeeze(num_valid)),
                               lambda:(R,T,tf.to_float(0.0),tf.to_float(0.0),tf.squeeze(num_valid)))



    def load_models(self,path):
        saver = tf.train.Saver()
        saver.restore(self.session,path)

    
    def __init__(self,model_path,image_size=(256,320),is_training=False,reuse_variables=False,num_points=1024,iters=[3,5,7]):
        self.reuse_variables=reuse_variables
        self.session = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

        self.placeholder_image1 = tf.placeholder(dtype=tf.float32,shape=(1,image_size[0],image_size[1],3))
        self.placeholder_image2 = tf.placeholder(dtype=tf.float32,shape=(1,image_size[0],image_size[1],3))
        photos=tf.concat([self.placeholder_image1,self.placeholder_image2],axis=0)

        self.placeholder_intrin = tf.placeholder(dtype=tf.float32,shape=(1,4,1))
        self.placeholder_points = tf.placeholder(dtype=tf.float32,shape=(1,num_points,2))
        self.placeholder_depths = tf.placeholder(dtype=tf.float32,shape=(1,num_points,1))
        self.placeholder_initR  = tf.placeholder(dtype=tf.float32,shape=(1,3,3))
        self.placeholder_initT  = tf.placeholder(dtype=tf.float32,shape=(1,3,1))

        drn=Features.DRN(is_training=is_training,reuse_variables=reuse_variables)
        self.backbone_features=drn.drn54_no_dilation(photos)

        pyramid_builder=Features.Pyramid(filters=128,is_training=is_training,reuse_variables=reuse_variables)
        self.pyramid=pyramid_builder.pyramid(self.backbone_features)

        self.predict_rotations,self.predict_translations,self.ratio=self.trackTF(self.placeholder_intrin,
                                                                      self.pyramid,
                                                                      self.placeholder_points,
                                                                      self.placeholder_depths,
                                                                      self.placeholder_initR,
                                                                      self.placeholder_initT,
                                                                      iters)
        self.load_models(model_path)

    def trackPY(self,image1,image2,intrinsics,points,depths,initR,initT):
        fetches  = {'rotations':self.predict_rotations,
                    'translations':self.predict_translations,
                    'keep_ratio':self.ratio}
        feed_dict= {self.placeholder_image1:image1,
                    self.placeholder_image2:image2,
                    self.placeholder_intrin:intrinsics,
                    self.placeholder_points:points,
                    self.placeholder_depths:depths,
                    self.placeholder_initR:initR,
                    self.placeholder_initT:initT}
        results=self.session.run(fetches,feed_dict=feed_dict)
        return results['rotations'],results['translations'],results['keep_ratio']








