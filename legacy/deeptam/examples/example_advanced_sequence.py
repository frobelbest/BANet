from deeptam_tracker.tracker import Tracker
from deeptam_tracker.evaluation.rgbd_sequence import RGBDSequence 
from deeptam_tracker.evaluation.metrics import rgbd_rpe
from deeptam_tracker.utils.vis_utils import convert_between_c2w_w2c,convert_array_to_colorimg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from PIL import ImageChops
import argparse

def init_visualization(title='DeepTAM Tracker'):
    """Initializes a simple visualization for tracking
    
    title: str
    """
    fig = plt.figure()
    fig.set_size_inches(10.5, 8.5)
    fig.suptitle(title, fontsize=16)
    
    ax1 = fig.add_subplot(2,2,1,projection='3d',aspect='equal')
    ax1.plot([],[],[],
            'r',
            label='Prediction')
    
    ax1.plot([],[],[],
            'g',
            label='Ground truth')
    ax1.legend()
    ax1.set_zlim(0.5,1.8)
    ax1.set_title('Trajectory')
    
    ax2 = fig.add_subplot(2,2,2)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    
    ax2.set_title('Current image')
    ax3 = fig.add_subplot(2,2,4)
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    
    ax3.set_title('Virtual current image')
    ax4 = fig.add_subplot(2,2,3)
    
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)
    ax4.set_title('Diff image')
    
    return [ax1, ax2, ax3, ax4]

def update_visualization(axes, pr_poses, gt_poses, image_cur, image_cur_virtual):
    """ Updates the visualization for tracking
    
    axes: a list of plt.axes
    
    pr_poses, gt_poses: a list of Pose
    
    image_cur, image_cur_virtual: np.array
    
    """
    pr_poses_c2w = [convert_between_c2w_w2c(x) for x in pr_poses]
    gt_poses_c2w = [convert_between_c2w_w2c(x) for x in gt_poses]
    
    axes[0].plot(np.array([x.t[0] for x in pr_poses_c2w]),
            np.array([x.t[1] for x in pr_poses_c2w]),
            np.array([x.t[2] for x in pr_poses_c2w]),
            'r',
            label='Prediction')
    
    axes[0].plot(np.array([x.t[0] for x in gt_poses_c2w]),
            np.array([x.t[1] for x in gt_poses_c2w]),
            np.array([x.t[2] for x in gt_poses_c2w]),
            'g',
            label='Ground truth')

    if image_cur_virtual is not None:
        image_cur = convert_array_to_colorimg(image_cur.squeeze())
        image_cur_virtual = convert_array_to_colorimg(image_cur_virtual.squeeze()) 
        diff = ImageChops.difference(image_cur, image_cur_virtual)
        axes[1].cla()
        axes[1].set_title('Current image')
        axes[2].cla()
        axes[2].set_title('Virtual current image')
        axes[3].cla()
        axes[3].set_title('Diff image')
        axes[1].imshow(np.array(image_cur))       
        axes[2].imshow(np.array(image_cur_virtual))
        axes[3].imshow(np.array(diff))

    plt.pause(1e-9)
    
def track_rgbd_sequence(checkpoint, datadir, tracking_module_path, visualization):
    """Tracks a rgbd sequence using deeptam tracker
    
    checkpoint: str
        directory to the weights
    
    datadir: str
        directory to the sequence data
    
    tracking_module_path: str
        file which contains the model class
        
    visualization: bool
    """
    
    ## initialization
    sequence = RGBDSequence(datadir)
    intrinsics = sequence.get_sun3d_intrinsics()
    tracker = Tracker(tracking_module_path,checkpoint,intrinsics)

    gt_poses = []
    timestamps = []
    key_pr_poses =[]
    key_gt_poses = []
    key_timestamps = []
    
    axes = init_visualization()

    frame = sequence.get_dict(0, intrinsics, tracker.image_width, tracker.image_height)
    pose0_gt = frame['pose']
    tracker.clear()
    tracker.set_init_pose(pose0_gt) # this step can be left out if gt_poses is aligned such that it starts from identity pose
    
    ## track a sequence
    for frame_idx in range(sequence.get_sequence_length()):
        print('frame {}'.format(frame_idx))
        frame = sequence.get_dict(frame_idx, intrinsics, tracker.image_width, tracker.image_height)
        timestamps.append(sequence.get_timestamp(frame_idx))
        result = tracker.feed_frame(frame['image'], frame['depth'])
        gt_poses.append(frame['pose'])
        pr_poses = tracker.poses
        
        if visualization:
            update_visualization(axes, pr_poses, gt_poses, frame['image'], result['warped_image'])
        
        if result['keyframe']:
            key_pr_poses.append(tracker.poses[-1])
            key_gt_poses.append(frame['pose'])
            key_timestamps.append(sequence.get_timestamp(frame_idx))
            
       
    ## evaluation
    pr_poses = tracker.poses
    errors_rpe = rgbd_rpe(gt_poses, pr_poses, timestamps)
    print('Frame-to-keyframe odometry evaluation [RPE], translational RMSE: {}[m/s]'.format(errors_rpe['translational_error.rmse']))
    
    update_visualization(axes, pr_poses, gt_poses, frame['image'], result['warped_image'])
    plt.show()
    
    del tracker
    
def main():
    parser = argparse.ArgumentParser(description='''
    This script uses deeptam tracker to track a sequence in the TUM RGBD-SLAM dataset.
    ''')
    parser.add_argument('--data_dir', help='set a sequence data directory (should be consistent with TUM RGBD-SLAM datasets), by default use sequence ../data/rgbd_dataset_freiburg1_desk', default=None)
    parser.add_argument('--disable_vis', help='disable the frame-by-frame visualization for speed-up',action='store_true')
    
    args = parser.parse_args()
    
    visualization = not args.disable_vis
    data_dir = args.data_dir

    examples_dir = os.path.dirname(__file__)
    checkpoint = os.path.join(examples_dir, '..', 'weights', 'deeptam_tracker_weights', 'snapshot-300000')
    if data_dir is None:
        data_dir = os.path.join(examples_dir, '..', 'data', 'rgbd_dataset_freiburg1_desk')
    tracking_module_path = os.path.join(examples_dir, '..', 'python/deeptam_tracker/models/networks.py')

    track_rgbd_sequence(checkpoint, data_dir, tracking_module_path, visualization)
    
        
if __name__ == "__main__":
    main()