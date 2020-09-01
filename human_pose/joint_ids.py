# Joint IDs and Connectivity
#
# Dylan Campbell <dylan.campbell@anu.edu.au>

import numpy as np

def get_joint_names_dict(joint_names):
    return {name: i for i, name in enumerate(joint_names)}

def get_ikea_joint_names():
    return [
        "nose", # 0
        "left eye", # 1
        "right eye", # 2
        "left ear", # 3
        "right ear", # 4
        "left shoulder", # 5
        "right shoulder", # 6
        "left elbow", # 7
        "right elbow", # 8
        "left wrist", # 9
        "right wrist", # 10
        "left hip", # 11
        "right hip", # 12
        "left knee", # 13
        "right knee", # 14
        "left ankle", # 15
        "right ankle", # 16
    ]

def get_ikea_connectivity():
    return [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [0, 5],
        [0, 6],
        [5, 6],
        [5, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [5, 11],
        [6, 12],
        [11, 12],
        [11, 13],
        [12, 14],
        [13, 15],
        [14, 16]
    ]

def get_ikea_joint_groups_names():
    return [
        "head",
        "shoulder",
        "elbow",
        "wrist",
        "hip",
        "knee",
        "ankle",
    ]

def get_ikea_joint_groups():
    return [
        [0, 1, 2, 3, 4], # head
        [5, 6], # shoulder
        [7, 8], # elbow
        [9, 10], # wrist
        [11, 12], # hip
        [13, 14], # knee
        [15, 16] # ankle
    ]

def get_ikea_joint_hflip_names():
    return {
        'left eye': 'right eye',
        'right eye': 'left eye',
        'left ear': 'right ear',
        'right ear': 'left ear',
        'left shoulder': 'right shoulder',
        'right shoulder': 'left shoulder',
        'left elbow': 'right elbow',
        'right elbow': 'left elbow',
        'left wrist': 'right wrist',
        'right wrist': 'left wrist',
        'left hip': 'right hip',
        'right hip': 'left hip',
        'left knee': 'right knee',
        'right knee': 'left knee',
        'left ankle': 'right ankle',
        'right ankle': 'left ankle',
    }

def get_body25_joint_names():
    return [
        "nose", # 0
        "neck", # 1
        "right shoulder", # 2
        "right elbow", # 3
        "right wrist", # 4
        "left shoulder", # 5
        "left elbow", # 6
        "left wrist", # 7
        "mid hip", # 8
        "right hip", # 9
        "right knee", # 10
        "right ankle", # 11
        "left hip", # 12
        "left knee", # 13
        "left ankle", # 14
        "right eye", # 15
        "left eye", # 16
        "right ear", # 17
        "left ear", # 18
        "left big toe", # 19
        "left small toe", # 20
        "left heel", # 21
        "right big toe", # 22
        "right small toe", # 23
        "right heel", # 24
        "background", # 25
    ]

def get_body25_connectivity():
    return [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [1, 5],
        [5, 6],
        [6, 7],
        [1, 8],
        [8, 9],
        [9, 10],
        [10, 11],
        [8, 12],
        [12, 13],
        [13, 14],
        [0, 15],
        [0, 16],
        [15, 17],
        [16, 18],
        [2, 9],
        [5, 12],
        [11, 22],
        [11, 23],
        [11, 24],
        [14, 19],
        [14, 20],
        [14, 21],
    ]


def get_body21_joint_names():
    return [
        "nose", # 0
        "neck", # 1
        "right shoulder", # 2
        "right elbow", # 3
        "right wrist", # 4
        "left shoulder", # 5
        "left elbow", # 6
        "left wrist", # 7
        "mid hip", # 8
        "right hip", # 9
        "right knee", # 10
        "right ankle", # 11
        "left hip", # 12
        "left knee", # 13
        "left ankle", # 14
        "right eye", # 15
        "left eye", # 16
        "right ear", # 17
        "left ear", # 18
        "neck (lsp)", # 19
        "top of head (lsp)", # 20
    ]

def get_hmmr_joint_names():
    return [
        "right ankle", # 0
        "right knee", # 1
        "right hip", # 2
        "left hip", # 3
        "left knee", # 4
        "left ankle", # 5
        "right wrist", # 6
        "right elbow", # 7
        "right shoulder", # 8
        "left shoulder", # 9
        "left elbow", # 10
        "left wrist", # 11
        "neck", # 12
        "top of head", # 13
        "nose", # 14
        "left eye", # 15
        "right eye", # 16
        "left ear", # 17
        "right ear", # 18
        "left big toe", # 19
        "right big toe", # 20
        "left small toe", # 21
        "right small toe", # 22
        "left heel", # 23
        "right heel", # 24
    ]

def get_h36m_joint_names():
    return [
        "mid hip", # 0
        "right hip", # 1
        "right knee", # 2
        "right ankle", # 3
        "left hip", # 4
        "left knee", # 5
        "left ankle", # 6
        "spine", # 7 -- mean(neck, mid hip)
        "neck", # 8
        "nose", # 9
        "head", # 10 -- mean(left ear, right ear)
        "left shoulder", # 11
        "left elbow", # 12
        "left wrist", # 13
        "right shoulder", # 14
        "right elbow", # 15
        "right wrist", # 16
    ]


def get_pose_colors(mode='rgb'):
    """

    Parameters
    ----------
    mode : rgb | bgr color format to return

    Returns
    -------
    list of part colors for skeleton visualization
    """
    # colormap from OpenPose: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/3c9441ae62197b478b15c551e81c748ac6479561/include/openpose/pose/poseParametersRender.hpp
    colors = np.array(
        [
            [255., 0., 85.],
            # [255., 0., 0.],
            [255., 85., 0.],
            [255., 170., 0.],
            [255., 255., 0.],
            [170., 255., 0.],
            [85., 255., 0.],
            [0., 255., 0.],
            [255., 0., 0.],
            [0., 255., 85.],
            [0., 255., 170.],
            [0., 255., 255.],
            [0., 170., 255.],
            [0., 85., 255.],
            [0., 0., 255.],
            [255., 0., 170.],
            [170., 0., 255.],
            [255., 0., 255.],
            [85., 0., 255.],

            [0., 0., 255.],
            [0., 0., 255.],
            [0., 0., 255.],
            [0., 255., 255.],
            [0., 255., 255.],
            [0., 255., 255.]])
    if mode == 'rgb':
        return colors
    elif mode == 'bgr':
        colors[:, [0, 2]] = colors[:, [2, 0]]
        return colors
    else:
        raise ValueError('Invalid color mode, please specify rgb or bgr')