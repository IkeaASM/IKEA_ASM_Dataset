# VideoPose3D Interface
#
# Adapted from VideoPose3D run.py script:
# https://github.com/facebookresearch/VideoPose3D/blob/master/run.py
# See LICENSE file in repository for usage:
# https://github.com/facebookresearch/VideoPose3D
#
# Dylan Campbell <dylan.campbell@anu.edu.au>

import os
import sys
import errno
import pickle
import json
import argparse
import numpy as np
from glob import glob
from time import time
from joint_ids import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def main(input_args):
    vp3d_dir = input_args.vp3d_dir
    sys.path.append(vp3d_dir)

    from common.camera import normalize_screen_coordinates
    from common.model import TemporalModel
    from common.generators import UnchunkedGenerator
    from common.arguments import parse_args

    args = parse_args()
    print(args)

    kps_left = [4, 5, 6, 11, 12, 13]
    kps_right = [1, 2, 3, 14, 15, 16]
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]

    filter_widths = [int(x) for x in args.architecture.split(',')]

    num_joints_in = 17
    in_features = 2
    num_joints_out = 17
        
    model_pos = TemporalModel(num_joints_in, in_features, num_joints_out,
                                filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels,
                                dense=args.dense)

    receptive_field = model_pos.receptive_field()
    print('INFO: Receptive field: {} frames'.format(receptive_field))
    pad = (receptive_field - 1) // 2 # Padding on each side
    if args.causal:
        print('INFO: Using causal convolutions')
        causal_shift = pad
    else:
        causal_shift = 0

    model_params = 0
    for parameter in model_pos.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()
        
    if args.resume or args.evaluate:
        chk_filename = os.path.join(vp3d_dir, args.checkpoint, args.resume if args.resume else args.evaluate)
        print('Loading checkpoint', chk_filename)
        checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
        print('This model was trained for {} epochs'.format(checkpoint['epoch']))
        model_pos.load_state_dict(checkpoint['model_pos'])

    # Evaluate
    def evaluate(test_generator, action=None, return_predictions=False):
        epoch_loss_3d_pos = 0
        epoch_loss_3d_pos_procrustes = 0
        epoch_loss_3d_pos_scale = 0
        epoch_loss_3d_vel = 0
        with torch.no_grad():
            model_pos.eval()
            N = 0
            for _, batch, batch_2d in test_generator.next_epoch():
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()
                # Positional model
                predicted_3d_pos = model_pos(inputs_2d)

                # Test-time augmentation (if enabled)
                if test_generator.augment_enabled():
                    # Undo flipping and take average with non-flipped version
                    predicted_3d_pos[1, :, :, 0] *= -1
                    predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                    predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
                    
                if return_predictions:
                    return predicted_3d_pos.squeeze(0).cpu().numpy()
                    
                inputs_3d = torch.from_numpy(batch.astype('float32'))
                if torch.cuda.is_available():
                    inputs_3d = inputs_3d.cuda()
                inputs_3d[:, :, 0] = 0    
                if test_generator.augment_enabled():
                    inputs_3d = inputs_3d[:1]

                error = mpjpe(predicted_3d_pos, inputs_3d)
                epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()

                epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
                N += inputs_3d.shape[0] * inputs_3d.shape[1]
                
                inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
                predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

                epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

                # Compute velocity error
                epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)
                
        if action is None:
            print('----------')
        else:
            print('----'+action+'----')
        e1 = (epoch_loss_3d_pos / N)*1000
        e2 = (epoch_loss_3d_pos_procrustes / N)*1000
        e3 = (epoch_loss_3d_pos_scale / N)*1000
        ev = (epoch_loss_3d_vel / N)*1000
        print('Test time augmentation:', test_generator.augment_enabled())
        print('Protocol #1 Error (MPJPE):', e1, 'mm')
        print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
        print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
        print('Velocity Error (MPJVE):', ev, 'mm')
        print('----------')

        return e1, e2, e3, ev

    def get_gt_dirs(input_path, camera_id='dev3'):
        """Get all directories with ground-truth 2D human pose annotations
        """
        gt_path_list = []
        category_path_list = get_subdirs(input_path)
        for category in category_path_list:
            if os.path.basename(category) != 'Calibration':
                category_scans = get_subdirs(category)
                for category_scan in category_scans:
                    device_list = get_subdirs(category_scan)
                    for device_path in device_list:
                        if camera_id in device_path:
                            if os.path.exists(os.path.join(device_path, 'pose2d')): # 2D annotations exist
                                gt_path_list.append(device_path) # eg <root>/Lack_TV_Bench/0007_white_floor_08_04_2019_08_28_10_47/dev3
        return gt_path_list

    def get_subdirs(input_path):
        '''
        get a list of subdirectories in input_path directory
        :param input_path: parent directory (in which to get the subdirectories)
        :return:
        subdirs: list of subdirectories in input_path
        '''
        subdirs = [os.path.join(input_path, dir_i) for dir_i in os.listdir(input_path)
                   if os.path.isdir(os.path.join(input_path, dir_i))]
        subdirs.sort()
        return subdirs

    fps = 30
    frame_width = 1920.0
    frame_height = 1080.0

    h36m_joint_names = get_h36m_joint_names()
    h36m_joint_names_dict = {name: i for i, name in enumerate(h36m_joint_names)}
    joint_names = get_body25_joint_names()
    joint_names_dict = {name: i for i, name in enumerate(joint_names)}

    dataset_dir = input_args.dataset_dir
    camera_id = input_args.camera_id

    gt_dirs = get_gt_dirs(dataset_dir, camera_id)
    for i, gt_dir in enumerate(gt_dirs):
        print(f"\nProcessing {i} of {len(gt_dirs)}: {' '.join(gt_dir.split('/')[-3:-1])}")
        
        input_dir = os.path.join(gt_dir, 'predictions', 'pose2d', 'openpose')
        output_dir = os.path.join(gt_dir, 'predictions', 'pose3d', 'vp3d')
        os.makedirs(output_dir, exist_ok=True)

        json_mask = os.path.join(input_dir, 'scan_video_00000000????_keypoints.json')
        json_files = sorted(glob(json_mask))
        input_keypoints = []
        for json_file in json_files:
            with open(json_file, 'r') as f:
                pose2d = json.load(f)
            if len(pose2d["people"]) == 0:
                keypoints_op = np.zeros((19, 3))
            else:
                keypoints_op = np.array(pose2d["people"][0]["pose_keypoints_2d"]).reshape(-1, 3) # Takes first detected person every time...
            keypoints = np.zeros((17, 3))
            for i, joint_name in enumerate(h36m_joint_names):
                if joint_name == 'spine' or joint_name == 'head':
                    continue
                joint_id = joint_names_dict[joint_name]
                keypoints[i, :] = keypoints_op[joint_id, :]
            keypoints[h36m_joint_names_dict['mid hip'], :] = np.mean((keypoints[h36m_joint_names_dict['left hip'], :], keypoints[h36m_joint_names_dict['right hip'], :]), axis=0) # mid hip = mean(left hip, right hip)
            keypoints[h36m_joint_names_dict['spine'], :] = np.mean((keypoints[h36m_joint_names_dict['neck'], :], keypoints[h36m_joint_names_dict['mid hip'], :]), axis=0) # spine = mean(neck, mid hip)
            keypoints[h36m_joint_names_dict['head'], :] = np.mean((keypoints_op[joint_names_dict['left ear'], :], keypoints_op[joint_names_dict['right ear'], :]), axis=0) # head = mean(left ear, right ear)
            input_keypoints.append(keypoints)
        input_keypoints = np.array(input_keypoints)

        input_keypoints = input_keypoints[:, :, :2] # For pretrained_h36m_cpn.bin and cpn_ft_h36m_dbb

        input_keypoints[..., :2] = normalize_screen_coordinates(input_keypoints[..., :2], w=frame_width, h=frame_height)

        args.test_time_augmentation=True
        gen = UnchunkedGenerator(None, None, [input_keypoints],
                                 pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                 kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
        prediction = evaluate(gen, return_predictions=True) # Nx17x3

        pickle.dump(prediction, open(os.path.join(output_dir, 'vp3d_output.pkl'), "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/home/djcam/Documents/HDD/datasets/ikea/ikea_asm/',
                        help='directory of the IKEA assembly dataset')
    parser.add_argument('--camera_id', type=str, default='dev3',
                        help='camera device ID for dataset (GT annotations for dev3 only)')
    parser.add_argument('--vp3d_dir', type=str, default='./VideoPose3D/',
                        help='VideoPose3D directory')
    args, remainder = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remainder

    main(args)