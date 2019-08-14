from deeptam_tracker.tracker import TrackerCore
from deeptam_tracker.evaluation.rgbd_sequence import RGBDSequence 
from deeptam_tracker.evaluation.metrics import position_diff, angle_diff
from deeptam_tracker.utils.vis_utils import convert_array_to_colorimg
from PIL import ImageChops
import matplotlib.pyplot as plt
import os

def simple_evaluation(pr_pose, gt_pose, key_pose, frame_id):
    """ Evaluates a pose prediction
    
    pr_pose, gt_pose, key_pose: Pose
    
    frame_id: int   
    """
    position_err = position_diff(pr_pose, gt_pose)
    orientation_err = angle_diff(pr_pose, gt_pose)
    position_change = position_diff(key_pose, gt_pose)
    orientation_change = angle_diff(key_pose, gt_pose)
    print('position_err:{:6.5f}[m], out of position_change:{:6.5f}[m]'.format(position_err, position_change) )
    print('orientation_err:{:6.5f}[degree], out of orientation_change:{:6.5f}[degree]'.format(orientation_err, orientation_change) )
    
def simple_visualization(image_key, image_cur, image_cur_virtual, frame_id):
    """Visualizes some image results
    
    image_key, image_cur, image_cur_virtual: np.array
    
    frame_id: int
    """
    image_key = convert_array_to_colorimg(image_key.squeeze())
    image_cur = convert_array_to_colorimg(image_cur.squeeze())
    image_cur_virtual = convert_array_to_colorimg(image_cur_virtual.squeeze())  
    
    diff = ImageChops.difference(image_cur, image_cur_virtual) # difference should be small if the predicted pose is correct

    print('Close window to continue...')
    
    plt.subplot(2,2,1)
    plt.gca().set_title('Key frame image')
    fig = plt.imshow(image_key)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.subplot(2,2,2)
    plt.gca().set_title('Current frame image {}'.format(frame_id))
    fig = plt.imshow(image_cur)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.subplot(2,2,3)
    plt.gca().set_title('Virtual current frame image {}'.format(frame_id))
    fig = plt.imshow(image_cur_virtual)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.subplot(2,2,4)
    plt.gca().set_title('Difference image')
    fig = plt.imshow(diff)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.show()
    

def main():
    ## initialization
    examples_dir = os.path.dirname(__file__)
    checkpoint = os.path.join(examples_dir, '..', 'weights','deeptam_tracker_weights', 'snapshot-300000')
    datadir = os.path.join(examples_dir, '..', 'data', 'rgbd_dataset_freiburg1_desk')
    tracking_module_path = os.path.join(examples_dir, '..', 'python/deeptam_tracker/models/networks.py')

    sequence = RGBDSequence(datadir)
    intrinsics = sequence.get_sun3d_intrinsics()
    
    tracker_core = TrackerCore(tracking_module_path,checkpoint,intrinsics)

    ## use first 3 frames as an example, the fisrt frame is selected as key frame
    frame_key = sequence.get_dict(0, intrinsics, tracker_core.image_width, tracker_core.image_height)
    frame_1 = sequence.get_dict(1, intrinsics, tracker_core.image_width, tracker_core.image_height)
    frame_2 = sequence.get_dict(2, intrinsics, tracker_core.image_width, tracker_core.image_height)

    print(frame_key['pose'])

    ## set the keyframe of tracker
    tracker_core.set_keyframe(frame_key['image'], frame_key['depth'], frame_key['pose'])

    ## track frame_1 w.r.t frame_key
    print('Track frame {} w.r.t key frame:'.format(1))
    results_1 = tracker_core.compute_current_pose(frame_1['image'], frame_key['pose'])
    pose_pr_1 = results_1['pose']

    simple_evaluation(results_1['pose'], frame_1['pose'], frame_key['pose'], 1)
    simple_visualization(frame_key['image'], frame_1['image'], results_1['warped_image'], 1)


    ## track frame_2 w.r.t frame_key incrementally using pose_pr_1 as pose_guess
    print('Track frame {} w.r.t key frame incrementally using previous predicted pose:'.format(2))
    results_2 = tracker_core.compute_current_pose(frame_2['image'], pose_pr_1)
    pose_pr_2 = results_2['pose']

    simple_evaluation(results_2['pose'], frame_2['pose'], frame_key['pose'], 2)
    simple_visualization(frame_key['image'], frame_2['image'], results_2['warped_image'], 2)

    del tracker_core
    
if __name__ == "__main__":
    main()