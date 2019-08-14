import os
import numpy as np
from .rgbd_sequence import RGBDSequence
from ..utils.datatypes import *

def position_diff(pose1, pose2):
    """Computes the position difference between two poses

    pose1: Pose
    pose2: Pose
    """
    return (pose1.R.transpose()*pose1.t - pose2.R.transpose()*pose2.t).norm()



def angle_diff(pose1, pose2):
    """Computes the angular difference [in degrees] between two poses

    pose1: Pose
    pose2: Pose
    """
    dot = pose1.R.row(2).dot(pose2.R.row(2))
    return np.rad2deg(np.arccos(np.clip(dot,0,1))) 

def rgbd_rpe(gt_poses, pr_poses, timestamps, cmdline_options=None):
    """Runs the rgbd command line tool for the RPE error

    gt_poses: list of Pose
    pr_poses: list of Pose
    timestamps: list of float

    cmdline_options: str 
        Options passed to the evaluation tool
        Default is '--fixed_delta'

    """
    import tempfile
    import shlex
    from .rgbd_benchmark.evaluate_rpe import evaluate_rpe
    assert len(pr_poses) == len(gt_poses)
    assert len(pr_poses) == len(timestamps)
    f, gt_txt = tempfile.mkstemp()
    os.close(f)
    RGBDSequence.write_rgbd_pose_format(gt_txt, gt_poses, timestamps)
    f, pr_txt = tempfile.mkstemp()
    os.close(f)
    RGBDSequence.write_rgbd_pose_format(pr_txt, pr_poses, timestamps)

    if cmdline_options is None:
        cmdline_options = '--fixed_delta'

    cmdline = '{0} {1} {2}'.format(cmdline_options, gt_txt, pr_txt)
    result = evaluate_rpe(shlex.split(cmdline))
    os.remove(gt_txt)
    os.remove(pr_txt)
    return result


def rgbd_ate(gt_poses, pr_poses, timestamps, cmdline_options=None):
    """Runs the rgbd command line tool for the ATE error

    gt_poses: list of Pose
    pr_poses: list of Pose
    timestamps: list of float

    cmdline_options: str 
        Options passed to the evaluation tool
        Default is ''

    """
    import tempfile
    import shlex
    from .rgbd_benchmark.evaluate_ate import evaluate_ate
    assert len(pr_poses) == len(gt_poses)
    assert len(pr_poses) == len(timestamps)
    f, gt_txt = tempfile.mkstemp()
    os.close(f)
    RGBDSequence.write_rgbd_pose_format(gt_txt, gt_poses, timestamps)
    f, pr_txt = tempfile.mkstemp()
    os.close(f)
    RGBDSequence.write_rgbd_pose_format(pr_txt, pr_poses, timestamps)

    if cmdline_options is None:
        cmdline_options = ''

    cmdline = '{0} {1} {2}'.format(cmdline_options, gt_txt, pr_txt)
    result = evaluate_ate(shlex.split(cmdline))
    os.remove(gt_txt)
    os.remove(pr_txt)
    return result
    

