import numpy as np
import tensorflow as tf
from .utils.helpers import *
from .utils.datatypes import *
from .utils.rotation_conversion import *

class Tracker:
    """This class implements a sequence tracker using TrackerCore. It allows for new key frame generation.
    
    """
    
    def __init__(self,
                tracking_module_path,
                checkpoint,
                intrinsics,
                key_valid_pixel_ratio_threshold=0.5,
                key_angle_deg_threshold=6.0,
                key_distance_threshold=0.15,
                init_pose=Pose_identity()):
        """
        tracking_module_path: str 
            Path to the networks.py

        checkpoint: str
            Path to a checkpoint

        intrinsics: np.array
            Normalized intrinsics

        key_valid_pixel_ratio_threshold: float
            Threshold for creating a new key frame

        key_angle_deg_threshold: float
            Threshold for creating a new key frame.

        key_distance_threshold: float
            Threshold for creating a new key frame.
            
        init_pose: Pose
            Initialized pose for the starting frame

        """
        self._tracker_core = TrackerCore(tracking_module_path, checkpoint, intrinsics)
        self._image_width = self._tracker_core.image_width
        self._image_height = self._tracker_core.image_height
        self._key_valid_pixel_ratio_threshold = key_valid_pixel_ratio_threshold
        self._key_angle_deg_threshold = key_angle_deg_threshold
        self._key_distance_threshold = key_distance_threshold
        self._init_pose = init_pose

        self.clear()
    
    def __del__(self):
        del self._tracker_core
        
    def clear(self):
        """Clears the pose lists and resets the init_tracker flag as True
        """
        self._init_tracker = True
        self._key_poses = []
        self._poses = []
    
    def set_init_pose(self, init_pose):
        """Sets init_pose
        
        init_pose: Pose
        """
        self._init_pose = init_pose
        
    @property
    def image_height(self):
        return self._image_height

    @property
    def image_width(self):
        return self._image_width

    @property
    def poses(self):
        return self._poses
    
    @property
    def key_poses(self):
        return self._key_poses
    
    @staticmethod
    def position_diff(pose1, pose2):
        """Computes the position difference between two poses

        pose1: Pose
        pose2: Pose
        """
        return (pose1.R.transpose()*pose1.t - pose2.R.transpose()*pose2.t).norm()


    @staticmethod
    def angle_diff(pose1, pose2):
        """Computes the angular difference between two poses

        pose1: Pose
        pose2: Pose
        """
        dot = pose1.R.row(2).dot(pose2.R.row(2))
        return np.rad2deg(np.arccos(np.clip(dot,0,1))) 
    
    def check_new_keyframe(self, new_pose, depth):
        """Checks if a new keyframe has to be set based on the distance, angle, valid_pixel threshold and the availability of the depth
        
        new_pose: Pose
        """
        
        set_new_keyframe = False
        key_pose = self._tracker_core._key_pose
        
        position_diff = self.position_diff(key_pose, new_pose)
        if not set_new_keyframe and position_diff > self._key_distance_threshold:
            set_new_keyframe = True
            print('setting new key frame because of distance threshold {0}'.format(position_diff))

        angle_diff = self.angle_diff(key_pose, new_pose)
        if not set_new_keyframe and angle_diff > self._key_angle_deg_threshold:
            set_new_keyframe = True
            print('setting new key frame because of angle threshold {0}'.format(angle_diff))

#         valid_warped_pixels = np.count_nonzero(output['warped_image'])
        if not set_new_keyframe and self._tracker_core._valid_warped_pixels/self._tracker_core._key_valid_depth_pixels < self._key_valid_pixel_ratio_threshold:
            set_new_keyframe = True
            print('setting new key frame because of valid pixel ratio threshold {0}'.format(valid_warped_pixels/self._key_valid_depth_pixels))
            
        if set_new_keyframe:
            if depth is None:
                set_new_keyframe = False
                print("cannot set new key frame because of missing depth") 
        
        return set_new_keyframe
    
    def set_new_keyframe(self, image, depth, pose):
        """Sets a new keyframe
        
        image: np.array
            Channels first(r,g,b), normalized in range [-0.5, 0.5]
            
        depth: np.array
            [height, width], inverse depth [1/m]
            
        pose: Pose
        """
        self._tracker_core.set_keyframe(key_image=image, key_depth=depth, key_pose=pose)
        
    
    def init_tracker(self, image, depth):
        """Initializes the tracker pose 
        
        image: np.array
        
        depth: np.array
        """
        self.clear()
        self._poses.append(self._init_pose)
        if depth is not None:
            self.set_new_keyframe(image, depth, self._init_pose)
            self._key_poses.append(self._init_pose)
            self._init_tracker = False
            return{
                    'pose': self._init_pose,
                    'keyframe': True,
                    'warped_image': None,
                    }
        else:
            print("cannot set new key frame because of missing depth")
            return {
                    'pose': self._init_pose,
                    'keyframe': False,
                    'warped_image':None
                    }
        
        
    def feed_frame(self, image, depth):
        """Feeds a new frame to the tracking algorithm
        
        image: np.array
            Current image, channels first(r,g,b), normalized in range [-0.5, 0.5]
        """
        if self._init_tracker:
            results = self.init_tracker(image, depth)
            return results

        results = self._tracker_core.compute_current_pose(image,self._poses[-1])
        new_pose = results['pose']
        self._poses.append(new_pose)
        
        keyframe = self.check_new_keyframe(new_pose, depth)
        if keyframe:
            self.set_new_keyframe(image, depth, new_pose)
            self._key_poses.append(new_pose)
            
        return{
                'pose': new_pose,
                'warped_image': results['warped_image'],
                'keyframe': keyframe
                }
    
class TrackerCore:
    """This class includes the basic functions to track w.r.t a key frame using DeepTAM networks.
    """
    
    def __init__(self,
                tracking_module_path,
                checkpoint,
                intrinsics):
        """
        tracking_module_path: str
            Path to the tracking module
            
        checkpoint:str
            Path to a checkpoint
            
        intrinsics: np.array
            Normalized intrinsics
        """
        self._tracking_module = tracking_module_path
        self._checkpoint = checkpoint
        self._intrinsics = np.squeeze(intrinsics)[np.newaxis,:]
        
        gpu_options = tf.GPUOptions()
        gpu_options.per_process_gpu_memory_fraction=0.8
        self._session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
        self._session.run(tf.global_variables_initializer())
        
        self._tracking_mod = load_myNetworks_module_noname(self._tracking_module)
        self._tracking_net = self._tracking_mod.TrackingNetwork()
        self._tracking_net_output = self._tracking_net.build_net(**self._tracking_net.placeholders)
        optimistic_restore(self._session, self._checkpoint, verbose=True)

        [self._image_height, self._image_width] = self._tracking_net.placeholders['image_key'].get_shape().as_list()[-2:]
        
    def __del__(self):
        """
        """
        del self._session
        tf.reset_default_graph()
        print("Tensorflow graph reseted")
     
    @property
    def image_width(self):
        return self._image_width
    
    @property
    def image_height(self):
        return self._image_height
    
    @property
    def intrinsics(self):
        return self._intrinsics
        
    def set_keyframe(self, key_image, key_depth, key_pose):
        """ Sets the keyframe 
        
        key_pose: Pose
        
        key_image: np.array
            Channels first(r,g,b), normalized in range [-0.5, 0.5]
        
        key_depth: np.array
            [height, width], inverse depth [1/m]
        """
        self._key_pose = key_pose
        self._key_image = np.squeeze(key_image)[np.newaxis,:,:,:]
        self._key_depth = np.squeeze(key_depth)[np.newaxis,np.newaxis,:,:]  
        self._key_valid_depth_pixels = np.count_nonzero(self._key_depth[np.isfinite(self._key_depth)]>0)
        
    def compute_current_pose(self, image, pose_guess=None):
        """ Computes the current pose
        
        pose_guess: Pose
        
        image: np.array
            Channels first(r,g,b), normalized in range [-0.5, 0.5]
        """
        
        # if pose guess is not given, use key_pose, this can reduce the performance when the motion is large
        # better use previous pose as pose_guess
        if pose_guess is None:
            pose_guess = self._key_pose
            
        image = np.squeeze(image)[np.newaxis,:,:,:]
        R_relative = pose_guess.R * self._key_pose.R.transpose()
        t_relative = pose_guess.t - R_relative*self._key_pose.t


        # print(self._intrinsics.shape)
        
        feed_dict = {
                    self._tracking_net.placeholders['prev_rotation']: rotation_matrix_to_angleaxis(R_relative)[np.newaxis,:].astype(np.float32),
                    self._tracking_net.placeholders['prev_translation']: np.array(t_relative, dtype=np.float32)[np.newaxis,:],
                    self._tracking_net.placeholders['image_key']: self._key_image,
                    self._tracking_net.placeholders['depth_key']: self._key_depth,
                    self._tracking_net.placeholders['image_current']: image,
                    self._tracking_net.placeholders['intrinsics']: self._intrinsics,
                }

        # for x in feed_dict:
        #     print(x,x.get_shape())

        fetch_dict = {
                    'predict_rotation': self._tracking_net_output['predict_rotation'],
                    'predict_translation': self._tracking_net_output['predict_translation'],
                    'warped_image': self._tracking_net_output['warped_image'],
        }
        output = self._session.run(fetch_dict, feed_dict=feed_dict)
        
        #print(Matrix3.__module__)
        #print(numpy_to_Vector3.__module__)
        # compute new Pose
        R_relative = Matrix3(angleaxis_to_rotation_matrix(output['predict_rotation']))
        t_relative = Vector3(numpy_to_Vector3(output['predict_translation'][0]))
        new_pose = Pose(R=R_relative*self._key_pose.R, t=t_relative+R_relative*self._key_pose.t)
        self._valid_warped_pixels = np.count_nonzero(output['warped_image'])
        return {
            'pose': new_pose,
            'warped_image': output['warped_image'],
            'rotation':angleaxis_to_rotation_matrix(output['predict_rotation']),
            'translation':output['predict_translation'][0]
        }
    
    