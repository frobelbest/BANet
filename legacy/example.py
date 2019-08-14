import ba2 as ba
import numpy as np
import cv2
import re
import struct
import os
from shutil import copyfile
ba.early_termination=False
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
num_pt=4096
height=384
width =512
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

tracker=ba.Tracker('/local-scratch3/bundlenet/scan_pose_2018_09_25_08_16_58/depth_28000.ckpt',image_size=(height,width),num_points=num_pt,iters=[5,8,12])
for x in range(2):

    initR=np.reshape(np.asarray([1,0,0,0,1,0,0,0,1]),[1,3,3]).astype(np.float32)
    initT=np.reshape(np.asarray([0,0,0]),[1,3,1]).astype(np.float32)
    intrinsics=np.reshape(np.asarray([0.88*512,1.17*384,256.0,192.0]),[1,4,1]).astype(np.float32)

    image1    =cv2.imread("/local-scratch2/cta73/tum_test/image_1/%06d.png"%(x))
    # image1    =cv2.cvtColor(image1,cv2.COLOR_RGB2BGR)
    image1    =cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    image1    =np.expand_dims(image1,axis=-1)
    image1    =np.tile(image1,(1,1,3))
    image2    =cv2.imread("/local-scratch2/cta73/tum_test/image_2/%06d.png"%(x))
    # image2    =cv2.cvtColor(image2,cv2.COLOR_RGB2BGR)
    image2    =cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
    image2    =np.expand_dims(image2,axis=-1)
    image2    =np.tile(image2,(1,1,3))
    depth,_   =readPFM('/local-scratch2/cta73/tum_test/depth/%06d.pfm'%(x))
    depth[np.isnan(depth)]=0.0

    file=open('/local-scratch2/cta73/tum_test/motion/%06d.bin'%(x),'r')
    motion=struct.unpack('f'*7,file.read(28))
    file.close()
    motion=np.asarray(motion).flatten()
    motion[:4]=motion[:4]/np.linalg.norm(motion[:4])

    points,depths=valid_point_and_depth(image1,depth,num_pt)

    image1=np.expand_dims(image1,axis=0).astype(np.float32)
    image2=np.expand_dims(image2,axis=0).astype(np.float32)
    points=points.astype(np.float32)
    rotation,translation,ratio=tracker.trackPY(image1,image2,intrinsics,points,depths,initR,initT)
    if not ba.early_termination:
        pred_quaternion=rotation2quaternion3D(rotation)
        pred_translation=translation
    else:
        pred_quaternion=rotation2quaternion3D(rotation)
        pred_translation=translation
    print 2*180*np.arccos(np.clip(np.dot(motion[:4],pred_quaternion),-1.0,1.0))/3.14,'/',2*180*np.arccos(pred_quaternion[0])/3.14,np.linalg.norm(motion[4:]-pred_translation.flatten()),'/',np.linalg.norm(pred_translation)