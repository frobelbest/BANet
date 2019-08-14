import ba
import numpy as np
import cv2
import re
import struct
import os
import sys
import quaternion as _quaternion

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
num_pt=4096
height=480
width =640
thres =120

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

def readPFM(file):
    file = open(file, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data,scale

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

def readlist(path):

    f=open(path+'/samples.txt')
    file=f.readlines()
    f.close()
    lists=[]

    cameras=[]
    for x in file:
        x=x.split()
        lists.append(path+'/'+x[1])
        lists.append(path+'/'+x[2])
        lists.append(x[0])
        camera1=np.asarray(x[3:7]).astype(np.float64)
        camera2=np.asarray(x[11:15]).astype(np.float64)
        #print camera1,camera2
        camera=(camera2[0]/(camera2[0]+camera1[0]))*camera1[1:]+(camera1[0]/(camera2[0]+camera1[0]))*camera2[1:]
        cameras.append(camera.flatten())
    
    lists=np.asarray(lists)
    lists=np.reshape(lists,(lists.shape[0]/3,3))
    return lists,cameras




lists,cameras=readlist('/local-scratch/tgz/rgbd_dataset_freiburg3_long_office_household')
tracker=ba.Tracker('/local-scratch3/bundlenet/scan_pose_2018_09_25_08_16_58/depth_28000.ckpt',image_size=(height,width),num_points=num_pt,iters=[5,8,8])
initR=np.reshape(np.asarray([1,0,0,0,1,0,0,0,1]),[1,3,3]).astype(np.float32)
initT=np.reshape(np.asarray([0,0,0]),[1,3,1]).astype(np.float32)
intrinsics=np.reshape(np.asarray([535.4,539.2,320.1,247.6]),[1,4,1]).astype(np.float32)


def drawCorrespondences(_image1,_image2,points,depths,rotation,translation,oldKey,newKey):
    
    image1=_image1.copy()
    image2=_image2.copy()
    _intrisics=intrinsics.flatten()

    depths=depths.flatten()
    for i in range(200):
        px=(points[0,i,0]-_intrisics[2])/_intrisics[0]
        py=(points[0,i,1]-_intrisics[3])/_intrisics[1]

        rotated=np.matmul(rotation,depths[i]*np.reshape(np.asarray([px,py,1.0]),[3,1]))
        rotated=(rotated+translation).flatten()

        px=((rotated[0]/rotated[2])*_intrisics[0])+_intrisics[2]
        py=((rotated[1]/rotated[2])*_intrisics[1])+_intrisics[3]

        color=255*np.random.rand(3)
        cv2.circle(image1,(points[0,i,0],points[0,i,1]),5,color)
        cv2.circle(image2,(int(px),int(py)),5,color)

    cv2.imwrite("/home/cta73/match%d_%d_1.png"%(oldKey,newKey),cv2.cvtColor(image1,cv2.COLOR_RGB2BGR))
    cv2.imwrite("/home/cta73/match%d_%d_2.png"%(oldKey,newKey),cv2.cvtColor(image2,cv2.COLOR_RGB2BGR))


image1    =cv2.imread(lists[0,0])
image1    =cv2.cvtColor(image1,cv2.COLOR_RGB2BGR)
depth     =cv2.imread(lists[0,1],cv2.IMREAD_ANYDEPTH)
depth     =depth.astype(np.float32)/5000.0
points,depths=valid_point_and_depth(image1,depth,num_pt)
points=points.astype(np.float32)
image1=np.expand_dims(image1,axis=0).astype(np.float32)

globalRotaitons   =[initR]
globalTranslations=[initT]
globalCameras=[initT.flatten()]
keyframeIndex=0

for i in range(1000):

    frameIndex=i+1

    if (float(lists[frameIndex,2])-float(lists[keyframeIndex,2]))>0.9:
        print float(lists[frameIndex,2])-float(lists[keyframeIndex,2])
        sys.exit()

    image2    =cv2.imread(lists[frameIndex,0])
    image2    =cv2.cvtColor(image2,cv2.COLOR_RGB2BGR)
    image2    =np.expand_dims(image2,axis=0).astype(np.float32)
    #print initR,initT
    rotation,translation,keep_ratio=tracker.trackPY(image1,image2,intrinsics,points,depths,initR,initT)
    #print(float(lists[frameIndex,2])-float(lists[keyframeIndex,2]))
    #print rotation[-1],translation[-1]
    globalRotaiton=np.matmul(rotation,globalRotaitons[keyframeIndex])
    globalTranslation=np.matmul(rotation,translation)+globalTranslations[keyframeIndex]
    globalRotaitons.append(globalRotaiton)
    globalTranslations.append(globalTranslation)

    #print globalRotaiton.shape
    globalRotaiton=np.transpose(globalRotaiton,(0,2,1)).astype(np.float64)
    camera=-np.matmul(globalRotaiton,globalTranslation).flatten()
    #print np.matmul(globalRotaiton[0,:,:],np.transpose(globalRotaiton,(0,2,1)))
    quaternion=_quaternion.from_rotation_matrix(globalRotaiton[0,:,:])
    print float(lists[frameIndex,2]),camera[0],camera[1],camera[2],quaternion.x,quaternion.y,quaternion.z,quaternion.w
    #globalCameras.append(camera)
    #dc_pred=globalCameras[-1]-globalCameras[-2]
    #dc_gt  =cameras[frameIndex]-cameras[keyframeIndex]
    #print np.abs(np.linalg.norm(dc_pred)-np.linalg.norm(dc_gt)),'/',np.linalg.norm(dc_gt)
    
    #print lists[keyframeIndex,2],lists[frameIndex,2]
    # print lists[keyframeIndex,2],lists[frameIndex,2],np.linalg.norm(dc_pred-dc_gt),'/',np.linalg.norm(dc_gt)
    # #print lists[keyframeIndex,2],lists[frameIndex,2],camera[0],camera[1],camera[2],cameras[frameIndex][0],cameras[frameIndex][1],cameras[frameIndex][2],float(lists[frameIndex,2]),np.linalg.norm(globalTranslations[-1]-globalTranslations[-2])/(float(lists[frameIndex,2])-float(lists[frameIndex-1,2]))
    #print lists[keyframeIndex,2],lists[frameIndex,2]
    if keep_ratio<0.8 or (float(lists[frameIndex,2])-float(lists[keyframeIndex,2]))>0.1:
        #print keyframeIndex,frameIndex
        #drawCorrespondences(image1[0,:,:,:],image2[0,:,:,:],points,depths,np.reshape(rotation[-1],(3,3)),np.reshape(translation[-1],(3,1)),keyframeIndex,frameIndex)
        
        keyframeIndex=frameIndex
        image1=image2
        depth =cv2.imread(lists[frameIndex,1],cv2.IMREAD_ANYDEPTH)
        depth =depth.astype(np.float32)/5000.0

        points,depths=valid_point_and_depth(image1[0,:,:,:],depth,num_pt)
        points=points.astype(np.float32)

        initR=np.reshape(np.asarray([1,0,0,0,1,0,0,0,1]),[1,3,3]).astype(np.float32)
        initT=np.reshape(np.asarray([0,0,0]),[1,3,1]).astype(np.float32)

    else:

        initR=rotation#[-1]
        initT=translation#[-1]



   