import os
from PIL import Image
import numpy as np
import glob
import re
from minieigen import Quaternion

from .rgbd_benchmark.associate import *
from .rgbd_benchmark.evaluate_rpe import transform44
from ..utils.datatypes import *
from ..utils.view_utils import adjust_intrinsics
from ..utils.rotation_conversion import *

class RGBDSequence:

    _all_intrinsics = {
        'ros_default': [545.0, 525.0, 319.5, 239.5],
        'freiburg1': [517.3, 516.5, 318.6, 255.3],
        'freiburg2': [520.9, 521.0,	325.1, 249.7],
        'freiburg3': [535.4, 539.2,	320.1, 247.6],
    }

    def __init__(self,sequence_dir,require_depth=False, require_pose=False):
        """Creates an object for accessing an rgbd benchmark sequence

        sequence_dir: str
            Path to the directory of a sequence
        use_all_images: boolean
            If True use all rgb images even if there is no depth or ground truth pose
        """
        self.sequence_dir = sequence_dir


        self.intrinsics = None


        depth_txt = os.path.join(sequence_dir, 'depth.txt')
        rgb_txt = os.path.join(sequence_dir, 'rgb.txt')
        groundtruth_txt = os.path.join(sequence_dir, 'groundtruth.txt')
        K_txt = os.path.join(sequence_dir, 'K.txt')

        if os.path.exists(K_txt):
            self._K = np.loadtxt(K_txt)
            self.intrinsics = [self._K[0,0], self._K[1,1], self._K[0,2], self._K[1,2]]
            self.rgb_dict = read_file_list(rgb_txt)
            self.depth_dict = read_file_list(depth_txt)
            if os.path.isfile(groundtruth_txt):
                self.groundtruth_dict = read_file_list(groundtruth_txt)
            else:
                self.groundtruth_dict = None

        elif 'freiburg' in sequence_dir: 

            self.rgb_dict = read_file_list(rgb_txt)
            self.depth_dict = read_file_list(depth_txt)

            for k,v in self._all_intrinsics.items():
                if k in sequence_dir:
                    self.intrinsics = v
                    self._K = np.eye(3)
                    self._K[0,0] = v[0]
                    self._K[1,1] = v[1]
                    self._K[0,2] = v[2]
                    self._K[1,2] = v[3]
                    break

            if os.path.isfile(groundtruth_txt):
                self.groundtruth_dict = read_file_list(groundtruth_txt)
            else:
                self.groundtruth_dict = None

        else:
            raise Exception("sequence not detected {0}".format(sequence_dir))



        self.matches_depth = associate(self.rgb_dict, self.depth_dict)    
        self.matches_depth_dict = dict(self.matches_depth)
        if not self.groundtruth_dict is None:
            self.matches_pose = associate(self.rgb_dict, self.groundtruth_dict)    
            self.matches_pose_dict = dict(self.matches_pose)

        self.matches_depth_pose = []
        for trgb in sorted(self.rgb_dict.keys()):
            img_path = os.path.join(self.sequence_dir, *self.rgb_dict[trgb])
            if not os.path.exists(img_path):
                continue
            if trgb in self.matches_depth_dict:
                tdepth = self.matches_depth_dict[trgb]
                depth_path = os.path.join(self.sequence_dir, *self.depth_dict[tdepth])
                if not os.path.exists(depth_path):
                    tdepth = None
            else:
                tdepth = None
            if require_depth and tdepth is None:
                continue
            if trgb in self.matches_pose_dict:
                tpose = self.matches_pose_dict[trgb]
            else:
                tpose = None
            if require_pose and tpose is None:
                continue
            self.matches_depth_pose.append((trgb,tdepth,tpose))

        # make sure the initial frame has a depth map and a pose
        while self.matches_depth_pose[0][1] is None or self.matches_depth_pose[0][2] is None:
            del self.matches_depth_pose[0]

        self.seq_len = len(self.matches_depth_pose)

        if self.intrinsics is None:
            raise Exception("No suitable intrinsics found")

        if not self.groundtruth_dict is None:
            self.groundtruth_txt = groundtruth_txt

        # open first matched image to get the original image size
        self.original_image_size = Image.open(os.path.join(self.sequence_dir, *self.rgb_dict[self.matches_depth_pose[0][0]])).size



    def name(self):
        return os.path.split(self.sequence_dir.strip('/'))[1]

    def get_sequence_length(self):
        return self.seq_len

    def get_timestamp(self, frame):
        """Returns the timestamp which corresponds to the rgb frame"""
        return self.matches_depth_pose[frame][2]

    def get_original_normalized_intrinsics(self):
        """Returns the original intrinsics in normalized form"""
        return np.array([
                self._K[0,0]/self.original_image_size[0], 
                self._K[1,1]/self.original_image_size[1], 
                self._K[0,2]/self.original_image_size[0], 
                self._K[1,2]/self.original_image_size[1]
                ], dtype=np.float32)

    def get_view(self, frame, normalized_intrinsics=None, width=None, height=None, depth=True):
        """Returns a view object for the given rgb frame
        
        frame: int
            The rgb frame number
        
        normalized_intrinsics: np.array or list
            Normalized intrinsics. Default is sun3d

        width: int
            image width. default is 128

        height: int
            image height. default is 96

        image: bool
            If true the returned view object contains the image

        depth: bool
            If true the returned view object contains the depth map

        inverse_depth: bool
            If true the returned depth is the inverse depth
        """
        
        if width is None:
            width = 128

        if height is None:
            height = 96

        if normalized_intrinsics is None:
            normalized_intrinsics = self.get_sun3d_intrinsics()
        new_K = np.eye(3)
        new_K[0,0] = normalized_intrinsics[0]*width
        new_K[1,1] = normalized_intrinsics[1]*height
        new_K[0,2] = normalized_intrinsics[2]*width
        new_K[1,2] = normalized_intrinsics[3]*height

        trgb, tdepth, tpose = self.matches_depth_pose[frame]

        img_path = os.path.join(self.sequence_dir, *self.rgb_dict[trgb])
        #print(img_path)
        img = Image.open(img_path)
        img.load()

        if depth and tdepth:
            depth_path = os.path.join(self.sequence_dir, *self.depth_dict[tdepth])
            #print(depth_path)
            
            dpth = self.read_depth_image(depth_path)
            dpth_metric = 'camera_z'
        else:
            dpth = None
            dpth_metric = None
        
        if tpose:
            print(tpose)
            timestamp_pose = [tpose] + self.groundtruth_dict[tpose]
            T = transform44(timestamp_pose)
            T = np.linalg.inv(T) # convert to world to cam

            R = T[:3,:3]
            t = T[:3,3]
        else:
            R = np.eye(3)
            t = np.array([0,0,0],dtype=np.float)

        view = View(R=R,t=t,K=self._K,image=img, depth=dpth, depth_metric=dpth_metric)

        new_view = adjust_intrinsics(view, new_K, width, height)
        if depth and tdepth:
            d = new_view.depth
            d[d <= 0] = np.nan
            new_view = new_view._replace(depth=d)

        view.image.close()
        del view
        return new_view



    def get_image(self, frame, normalized_intrinsics=None, width=None, height=None):
        """Returns the image for the specified frame as numpy array

        frame: int
            The rgb frame number
        
        normalized_intrinsics: np.array or list
            Normalized intrinsics. Default is sun3d

        width: int
            image width. default is 128

        height: int
            image height. default is 96

        """
        img = self.get_view(frame, normalized_intrinsics, width, height, depth=False).image
        img_arr = np.array(img).transpose([2,0,1]).astype(np.float32)/255 -0.5
        return img_arr


    def get_depth(self, frame, normalized_intrinsics=None, width=None, height=None, inverse=False):
        """Returns the depth for the specified frame

        frame: int
            The rgb frame number
        
        normalized_intrinsics: np.array or list
            Normalized intrinsics. Default is sun3d

        width: int
            image width. default is 128

        height: int
            image height. default is 96

        """
        depth = self.get_view(frame, normalized_intrinsics, width, height, depth=True, ).depth
        if inverse and not depth is None:
            depth = 1/depth
        return depth
        
        
    def get_image_depth(self, frame, normalized_intrinsics=None, width=None, height=None, inverse=False):
        """Returns the depth for the specified frame

        frame: int
            The rgb frame number
        
        normalized_intrinsics: np.array or list
            Normalized intrinsics. Default is sun3d

        width: int
            image width. default is 128

        height: int
            image height. default is 96

        """
        view = self.get_view(frame, normalized_intrinsics, width, height, depth=True)
        depth = view.depth
        if inverse and not depth is None:
            depth = 1/depth
        return (view.image, depth)
        
        

    def get_dict(self, frame, normalized_intrinsics=None, width=None, height=None):
        """Returns image, depth and pose as a dict of numpy arrays
        The depth is the inverse depth.
        
        frame: int
            The rgb frame number

        width: int
            image width. default is 128

        height: int
            image height. default is 96
        """
        view = self.get_view(frame, normalized_intrinsics=normalized_intrinsics, width=width, height=height, depth=True)
        
        img_arr = np.array(view.image).transpose([2,0,1]).astype(np.float32)/255 -0.5
        rotation = Quaternion(view.R).toAngleAxis()
        rotation = rotation[0]*np.array(rotation[1])

        result = { 
            'image': img_arr[np.newaxis,:,:,:], 
            'depth': None,
            'rotation': rotation[np.newaxis,:],
            'translation': view.t[np.newaxis,:],
            'pose':Pose(R=Matrix3(angleaxis_to_rotation_matrix(rotation)), t=Vector3(view.t))
        }
        if not view.depth is None:
            result['depth'] = (1/view.depth)[np.newaxis,np.newaxis,:,:]
        return result

    def get_relative_motion(self, frame1, frame2):
        """Returns the realtive transformation from frame1 to frame2
        """
        if self.groundtruth_dict is None:
            return None

        trgb, tdepth, tpose = self.matches_depth_pose[frame1]
        timestamp_pose = [tpose] + self.groundtruth_dict[tpose]
        inv_T1 = transform44(timestamp_pose)
        
        trgb, tdepth, tpose = self.matches_depth_pose[frame2]
        timestamp_pose = [tpose] + self.groundtruth_dict[tpose]
        T2 = transform44(timestamp_pose)
        T2 = np.linalg.inv(T2) # convert to world to cam

        T = T2.dot(inv_T1)
        R12 = T[:3,:3]
        t12 = T[:3,3]
        rotation = Quaternion(R12).toAngleAxis()
        rotation = rotation[0]*np.array(rotation[1])

        return { 
                'rotation': rotation[np.newaxis,:],
                'translation': t12[np.newaxis,:],
                }



    @staticmethod
    def get_sun3d_intrinsics():
        """Returns the normalized intrinsics of sun3d"""
        return np.array([0.89115971, 1.18821287, 0.5, 0.5], dtype=np.float32)


    @staticmethod
    def read_depth_image(path):
        """Reads a png depth image and returns it as 2d numpy array.
        Invalid values will be represented as NAN

        path: str
            Path to the image
        """
        scalingFactor = 5000.0
        depth = Image.open(path)
        depth.load()
        if depth.mode != "I":
            raise Exception("Depth image is not in intensity format {0}".format(path))
        depth_arr = np.array(depth)/scalingFactor
        #depth_arr[depth_arr == 0] = np.nan
        del depth
        return depth_arr.astype(np.float32)
    
    @staticmethod
    def write_rgbd_pose_format(path, poses, timestamps):
        """writes a pose txt file compatible with the rgbd eval tools
    
        path: str 
            Path to the output file

        poses: list of Pose
        timestamps: list of float
        """
        assert len(poses) == len(timestamps)
        with open(path,'w') as f:
            
            for i in range(len(poses)):
                pose = poses[i]
                timestamp = timestamps[i]

                T = np.eye(4)
                T[:3,:3] = np.array(pose.R)
                T[:3,3] = np.array(pose.t)
                T = np.linalg.inv(T) # convert to cam to world
                R = T[:3,:3]
                t = T[:3,3]
                
                q = Quaternion(R)
                f.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format(timestamp, *t, *q))