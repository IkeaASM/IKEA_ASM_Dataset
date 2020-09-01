import os
import re
import sys
sys.path.append('.')
import cv2
import math
import time
import json
import scipy
import argparse
import subprocess
import shutil
import matplotlib
import numpy as np
import pylab as plt
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter

def main(args):

    sys.path.append(args.openpose_dir) # In case calling from an external script
    from lib.network.rtpose_vgg import get_model
    from lib.network.rtpose_vgg import use_vgg
    from lib.network import im_transform
    from evaluate.coco_eval import get_outputs, handle_paf_and_heat
    from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans
    from lib.utils.paf_to_pose import paf_to_pose_cpp
    from lib.config import cfg, update_config

    update_config(cfg, args)

    model = get_model('vgg19')
    model = torch.nn.DataParallel(model).cuda()
    use_vgg(model)
    
    # model.load_state_dict(torch.load(args.weight))
    checkpoint = torch.load(args.weight)
    epoch = checkpoint['epoch']
    best_loss = checkpoint['best_loss']
    state_dict = checkpoint['state_dict']
    # state_dict = {key.replace("module.",""):value for key, value in state_dict.items()} # Remove "module." from vgg keys
    model.load_state_dict(state_dict)
    # optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {})".format(args.weight, epoch))

    model.float()
    model.eval()

    image_folders = args.image_folders.split(',')

    for i, image_folder in enumerate(image_folders):
        print(f"\nProcessing {i} of {len(image_folders)}: {' '.join(image_folder.split('/')[-4:-2])}")

        if args.all_frames: # Split video and run inference on all frames
            output_dir = os.path.join(os.path.dirname(image_folder), 'predictions', 'pose2d', 'openpose_pytorch_ft_all')
            os.makedirs(output_dir, exist_ok=True)
            video_path = os.path.join(image_folder, 'scan_video.avi') # break up video and run on all frames
            temp_folder = image_folder.split('/')[-3] + '_openpose'
            image_folder = os.path.join('/tmp', f'{temp_folder}') # Overwrite image_folder
            os.makedirs(image_folder, exist_ok=True)
            split_video(video_path, image_folder)
        else: # Just use GT-annotated frames
            output_dir = os.path.join(os.path.dirname(image_folder), 'predictions', 'pose2d', 'openpose_pytorch_ft')
            os.makedirs(output_dir, exist_ok=True)

        img_mask = os.path.join(image_folder, '??????.png')
        img_names = glob(img_mask)
        for img_name in img_names:
            image_file_path = img_name

            oriImg = cv2.imread(image_file_path) # B,G,R order
            shape_dst = np.min(oriImg.shape[0:2])

            with torch.no_grad():
                paf, heatmap, im_scale = get_outputs(oriImg, model,  'rtpose')

            humans = paf_to_pose_cpp(heatmap, paf, cfg)

            # Save joints in OpenPose format
            image_h, image_w = oriImg.shape[:2]
            people = []
            for i, human in enumerate(humans):
                keypoints = []
                for j in range(18):
                    if j == 8:
                        keypoints.extend([0, 0, 0]) # Add extra joint (midhip) to correspond to body_25
                    if j not in human.body_parts.keys():
                        keypoints.extend([0, 0, 0])
                    else:
                        body_part = human.body_parts[j]
                        keypoints.extend([body_part.x * image_w, body_part.y * image_h, body_part.score])
                person = {
                    "person_id":[i-1],
                    "pose_keypoints_2d":keypoints
                }
                people.append(person)
            people_dict = {"people":people}

            _, filename = os.path.split(image_file_path)
            name, _ = os.path.splitext(filename)
            frame_id = int(name)
            with open(os.path.join(output_dir, f"scan_video_{frame_id:012}_keypoints.json"), 'w') as outfile:
                json.dump(people_dict, outfile)

        if args.all_frames:
            shutil.rmtree(image_folder) # Delete image_folder

def split_video(video_file, image_folder):
    """Split video to frames and keep those with corresponding GT
    """
    command = ['ffmpeg',
               '-i', video_file,
               '-f', 'image2',
               '-start_number', '0',
               '-v', 'error',
               f'{image_folder}/%06d.png']
    subprocess.call(command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='experiment configure file name',
                        default='./experiments/vgg19_368x368_sgd.yaml', type=str)
    parser.add_argument('--weight', type=str,
                        default='pose_model.pth')
    parser.add_argument("--image_filename", type=str, default='')
    parser.add_argument('--output_path', type=str, default='')
    parser.add_argument('--image_folders', type=str, default='')
    parser.add_argument('--openpose_dir', type=str, default='./')
    parser.add_argument('--all_frames', action='store_true')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()

    main(args)