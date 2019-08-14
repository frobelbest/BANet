from abc import ABC, abstractmethod
import tensorflow as tf

class TrackingNetworkBase(ABC):

    def __init__(self, batch_size=1):

        self._placeholders = {
            'depth_key': tf.placeholder(tf.float32, shape=(batch_size, 1, 96, 128)),
            'image_key': tf.placeholder(tf.float32, shape=(batch_size, 3, 96, 128)),
            'image_current': tf.placeholder(tf.float32, shape=(batch_size, 3, 96, 128)),
            'intrinsics': tf.placeholder(tf.float32, shape=(batch_size, 4)),
            'prev_rotation': tf.placeholder(tf.float32, shape=(batch_size, 3)),
            'prev_translation': tf.placeholder(tf.float32, shape=(batch_size, 3)),
        }

    @property
    def placeholders(self):
        """All placeholders required for feeding this network"""
        return self._placeholders


    @abstractmethod
    def build_net(self, depth_key, image_key, image_current, intrinsics, prev_rotation, prev_translation):
        """Build the tracking network

        depth_key: the depth map of the key frame
        image_key: the image of the key frame
        image_current: the current image
        intrinsics: the camera intrinsics
        prev_rotation: the current guess for the camera rotation as angle axis representation
        prev_translation: the current guess for the camera translation

        Returns all network outputs as a dict.
        The following must be returned:

            predict_rotation
            predict_translation

        """
        pass



