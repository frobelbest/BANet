import numpy as np
import cv2
import re
import struct
import os
from shutil import copyfile
import quaternion as _quaternion
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
from deeptam_tracker.tracker import TrackerCore
import minieigen
import deeptam_tracker.utils.rotation_conversion as rotation_converter
import deeptam_tracker.utils.datatypes as datatypes
import sys


num_pt=4096
height=480
width =640
thres =80

x     =np.linspace(0,width-1,width)
y     =np.linspace(0,height-1,height)
x,y   =np.meshgrid(x,y)
x     =x.flatten()
y     =y.flatten()
xy    =np.stack((x,y),axis=-1).astype(np.float32)

def rotation2quaternion3D(R):
    diag=1.0+R[:,0,0]+R[:,1,1]+R[:,2,2]
    q0=np.sqrt(diag)/2.0
    q1=(R[:,2,1]-R[:,1,2])/(4.0*q0)
    q2=(R[:,0,2]-R[:,2,0])/(4.0*q0)
    q3=(R[:,1,0]-R[:,0,1])/(4.0*q0)
    q=np.stack((q0,q1,q2,q3))
    q=q.flatten()
    q=q/np.linalg.norm(q)
    return q

def load_pair():
    sequences={}
    for i in range(200):
        sequences[i]=[]
    with open('/local-scratch3/batest/test.txt') as f:
        g=f.readlines()
        for x in g:
            x=x[:-1]
            end=x.find('/1341')
            index=int(x[2:end])
            image=x[end+1:-4]
            sequences[index].append(image)
    for i in range(200):
        sequences[i].sort()
    return sequences

def load_data():
    data={}
    with open('/local-scratch3/batest/samples2.txt') as f:
        g=f.readlines()
        for x in g:
            x=x.split()
            data[x[0]]={}
            data[x[0]]['t']=np.asarray(x[1:4]).astype(float).flatten()
            data[x[0]]['q']=np.asarray(x[4:8]).astype(float).flatten()
            data[x[0]]['depth']=x[8]
    return data

def valid_point_and_depth(image,depth,num_points):
    dx=cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3)
    dy=cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3)
    dxy=np.sqrt(np.sum(np.square(dx),axis=-1)+np.sum(np.square(dy),axis=-1))
    dxy=dxy.flatten()
    d=depth.flatten()
    index=np.logical_and(np.greater(dxy,thres),np.greater(d,1e-5))
    d=d[index]
    p=xy[index,:]
    index=np.random.randint(0,p.shape[0],num_points)
    return np.reshape(p[index,:],(1,num_points,2)),np.reshape(d[index],(1,num_points,1))


def drawCorrespondences(_image1,_image2,points,depths,rotation,translation,oldKey,newKey):

    image1=_image1.copy()
    image2=_image2.copy()
    _intrisics=intrinsics.flatten()

    depths=depths.flatten()
    for i in range(200):
        px=(points[0,i,0]-_intrisics[2])/_intrisics[0]
        py=(points[0,i,1]-_intrisics[3])/_intrisics[1]

        rotated=np.matmul(rotation,depths[i]*np.reshape(np.asarray([px,py,1.0]),[3,1]))
        rotated=(rotated.flatten()+translation.flatten())

        px=((rotated[0]/rotated[2])*_intrisics[0])+_intrisics[2]
        py=((rotated[1]/rotated[2])*_intrisics[1])+_intrisics[3]

        color=255*np.random.rand(3)
        cv2.circle(image1,(points[0,i,0],points[0,i,1]),5,color)
        cv2.circle(image2,(int(px),int(py)),5,color)

    cv2.imwrite("/home/cta73/match%d_%d_1.png"%(oldKey,newKey),cv2.cvtColor(image1,cv2.COLOR_RGB2BGR))
    cv2.imwrite("/home/cta73/match%d_%d_2.png"%(oldKey,newKey),cv2.cvtColor(image2,cv2.COLOR_RGB2BGR))

def valid_point_and_depth2(image1,image2,depth1,depth2,rotation,translation,intrinsics):
    _intrisics=intrinsics.flatten()
    points=[]
    depths=[]
    count=0
    
    dx=cv2.Sobel(image1,cv2.CV_32F,1,0,ksize=3)
    dy=cv2.Sobel(image1,cv2.CV_32F,1,0,ksize=3)
    dxy=np.sqrt(np.sum(np.square(dx),axis=-1)+np.sum(np.square(dy),axis=-1))

    for i in range(height):
        for j in range(width):

            if depth1[i,j]<1e-5 or dxy[i,j]<80:
                continue

            px=(j-_intrisics[2])/_intrisics[0]
            py=(i-_intrisics[3])/_intrisics[1]

            rotated=np.matmul(rotation,depth1[i,j]*np.reshape(np.asarray([px,py,1.0]),[3,1]))
            rotated=(rotated.flatten()+translation.flatten())

            px=((rotated[0]/rotated[2])*_intrisics[0])+_intrisics[2]
            py=((rotated[1]/rotated[2])*_intrisics[1])+_intrisics[3]


            if int(py)<0 or int(py)>=height or int(px)<0 or int(px)>=width:
                continue

            color1=image1[i,j,:]
            color2=image2[int(py),int(px),:]

            if np.linalg.norm(color1-color2)>64:
                continue

            if (abs(rotated[2]-depth2[int(py),int(px)])/rotated[2])>0.2:
                continue

            points.append([j,i])
            depths.append(depth1[i,j])
            count+=1

    points=np.reshape(np.asarray(points),(1,count,2))
    depths=np.reshape(np.asarray(depths),(1,count,1))
    index =np.random.randint(0,count,num_pt)
    return points[:,index,:],depths[:,index,:]



examples_dir = os.path.dirname(__file__)
checkpoint = os.path.join(examples_dir, '..', 'weights','deeptam_tracker_weights', 'snapshot-300000')
tracking_module_path = os.path.join(examples_dir, '..', 'python/deeptam_tracker/models/networks.py')
        
intrinsics      = np.array([0.89115971, 1.18821287, 0.5, 0.5],dtype=np.float32)#np.eye(3)
tracker_core    = TrackerCore(tracking_module_path,checkpoint,intrinsics)


new_K = np.eye(3)
new_K[0,0] = intrinsics[0]*320
new_K[1,1] = intrinsics[1]*240
new_K[0,2] = intrinsics[2]*320
new_K[1,2] = intrinsics[3]*240


old_K = np.eye(3)
old_K[0,0] = 535.4
old_K[1,1] = 539.2
old_K[0,2] = 320.1
old_K[1,2] = 247.6

warpK=np.matmul(new_K,np.linalg.inv(old_K))
warpK/=warpK[-1,-1]

pairs=load_pair()
data=load_data()
for i in range(199):
    
    if len(pairs[i])==0:
        continue

    invalid=False
    for j in range(len(pairs[i])):
        if pairs[i][j] not in data:
            invalid=True

    if invalid:
        continue

    image1    =cv2.imread('/local-scratch/tgz/rgbd_dataset_freiburg3_long_office_household'+'/rgb/'+pairs[i][0]+'.png')
    image1    =cv2.cvtColor(image1,cv2.COLOR_RGB2BGR)
    
    #image1    =cv2.resize(image1,(320,240))
    image1    =cv2.warpPerspective(image1,warpK,(320,240))
    
    depth     =cv2.imread('/local-scratch/tgz/rgbd_dataset_freiburg3_long_office_household'+'/depth/'+data[pairs[i][0]]['depth']+'.png',cv2.IMREAD_ANYDEPTH)
    depth     =depth.astype(np.float32)/5000.0
    #depth     =cv2.resize(depth,(320,240),cv2.INTER_NEAREST)
    depth     =cv2.warpPerspective(depth,warpK,(320,240),flags=cv2.INTER_NEAREST)
    depth     =1.0/depth
    image1    =np.expand_dims(image1,axis=0).astype(np.float32)
    image1    =np.transpose(image1,(0,3,1,2))
    image1    =image1/255-0.5

    pose       = datatypes.Pose_identity()
    tracker_core.set_keyframe(image1,depth,pose)

    for j in range(len(pairs[i])-1,len(pairs[i])):
        
        image2    =cv2.imread('/local-scratch/tgz/rgbd_dataset_freiburg3_long_office_household'+'/rgb/'+pairs[i][j]+'.png')
        image2    =cv2.cvtColor(image2,cv2.COLOR_RGB2BGR)
    
        #image2    =cv2.resize(image2,(320,240))
        image2    =cv2.warpPerspective(image2,warpK,(320,240))
        
        image2    =np.expand_dims(image2,axis=0).astype(np.float32)
        image2    =np.transpose(image2,(0,3,1,2))

        image2    =image2/255-0.5

        results   = tracker_core.compute_current_pose(image2,pose)
        pose      = results['pose']

        rotation=results['rotation']
        translation=results['translation']


    T1=np.identity(4)
    q1=np.array([_quaternion.quaternion(data[pairs[i][0]]['q'][3],data[pairs[i][0]]['q'][0],data[pairs[i][0]]['q'][1],data[pairs[i][0]]['q'][2])])
    T1[:3,:3]=_quaternion.as_rotation_matrix(q1)
    T1[:3,3] =data[pairs[i][0]]['t']
    T2=np.identity(4)

    q2=np.array([_quaternion.quaternion(data[pairs[i][len(pairs[i])-1]]['q'][3],data[pairs[i][len(pairs[i])-1]]['q'][0],data[pairs[i][len(pairs[i])-1]]['q'][1],data[pairs[i][len(pairs[i])-1]]['q'][2])])
    T2[:3,:3]=_quaternion.as_rotation_matrix(q2)
    T2[:3,3] =data[pairs[i][len(pairs[i])-1]]['t']
    T=np.matmul(np.linalg.inv(T2),T1)

    _pred_quaternion=_quaternion.from_rotation_matrix(np.reshape(rotation,(3,3)))
    pred_quaternion=np.asarray([_pred_quaternion.w,_pred_quaternion.x,_pred_quaternion.y,_pred_quaternion.z]).flatten()
    if _pred_quaternion.w<0:
        pred_quaternion*=-1
    pred_translation=translation.flatten()

    motion=np.zeros((7))
    quaternion=_quaternion.from_rotation_matrix(T[:3,:3])
    motion[0]=quaternion.w
    motion[1]=quaternion.x
    motion[2]=quaternion.y
    motion[3]=quaternion.z
    motion[4:]=T[:3,3]
    if quaternion.w<0:
        motion[:4]*=-1

    # print(rotation,translation,pred_quaternion,pred_translation,motion)
    # sys.exit()
    print(2*180*np.arccos(np.clip(np.dot(motion[:4],pred_quaternion),-1.0,1.0))/3.14,'/',2*180*np.arccos(pred_quaternion[0])/3.14,np.linalg.norm(motion[4:]-pred_translation.flatten()),'/',np.linalg.norm(pred_translation))
    #drawCorrespondences(image1[0,:,:,:],image2[0,:,:,:],points,depths,np.reshape(rotation,(3,3)),translation,i,i+1)

