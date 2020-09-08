# Class for IKEA 2D pose annotations
#
# Dylan Campbell <dylan.campbell@anu.edu.au>

import os
import sys
import json
import cv2
import shutil
import argparse
import itertools
import numpy as np
import copy
import logging
import subprocess
import torch
import torchvision.transforms as transforms
from glob import glob
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from joint_ids import *

from pdb import set_trace as st

#########################################################################################################################
# Class for Keypoint R-CNN

class IKEAKeypointRCNNDataset(torch.utils.data.Dataset):
    """ For flattened dataset structure with test and train subfolders with 2D human pose annotations.
    <root>
    --train
    ----Kallax_Shelf_Drawer_0001_black_table_02_01_2019_08_16_14_00_dev3_000000.json
    ----Kallax_Shelf_Drawer_0001_black_table_02_01_2019_08_16_14_00_dev3_000000.png
    ----...
    --test
    ----Kallax_Shelf_Drawer_0001_oak_floor_05_02_2019_08_19_16_54_dev3_000000.json
    ----Kallax_Shelf_Drawer_0001_oak_floor_05_02_2019_08_19_16_54_dev3_000000.png
    ----...
    """
    def __init__(self, root, image_transform=None, preprocess=None, bbox_dilation_factor=1.4):
        
        self.root = root
        image_mask = os.path.join(self.root, '*.png')
        image_paths = sorted(glob(image_mask))
        self.ids = [split_filename(filename)[1] for filename in image_paths]

        self.joint_names = get_ikea_joint_names()
        self.num_joints = len(self.joint_names)

        # Dataset images have uniform sizes:
        image = Image.open(image_paths[0])
        width, height = image.size
        self.image_width = width
        self.image_height = height

        # Load all annotations:
        self.annotations_all = []
        for id in self.ids:
            json_path = os.path.join(self.root, id + '.json')
            with open(json_path, 'r') as json_file:
                pose2d_gt = json.load(json_file)
            keypoints = torch.zeros(self.num_joints, 3) # 17x3
            for i, joint_name in enumerate(self.joint_names):
                position = pose2d_gt[joint_name]["position"]
                confidence = pose2d_gt[joint_name]["confidence"]
                keypoints[i, :] = torch.tensor(position + [confidence])
            self.annotations_all.append({
                "keypoints": keypoints.unsqueeze(0), # 1x17x3 tensor
                "boxes": self.bbox_from_keypoints(keypoints, dilation_factor=bbox_dilation_factor).unsqueeze(0), # 1x4 tensor
                "labels": torch.ones(1, dtype=torch.long), # person category = 1
                "id": id
                })

        self.preprocess = preprocess
        self.image_transform = image_transform

    def __getitem__(self, index):
        image_id = self.ids[index]
        annotations = self.annotations_all[index]
        annotations = copy.deepcopy(annotations)

        image_filename = image_id + ".png"
        image = Image.open(os.path.join(self.root, image_filename)).convert('RGB')

        # Data augmentation (image and annotations):
        if self.preprocess:
            image, annotations = self.preprocess(image, annotations)

        # Data augmentation (image only):
        if not self.image_transform:
            # Basic preprocessing:
            to_tensor = transforms.ToTensor() # converts to (C x H x W) in the range [0.0, 1.0]
            image = to_tensor(image)
        else:
            image = self.image_transform(image)

        return image, annotations # Need ID to retain useful information when evaluating (eg gender)

    def bbox_from_keypoints(self, keypoints, dilation_factor):
        """ Get a bounding box enclosing the keypoints, respecting image boundaries.
        """
        min_p = keypoints[:, :2].min(dim=0).values
        max_p = keypoints[:, :2].max(dim=0).values
        half_widths = 0.5 * (max_p - min_p)
        centre = 0.5 * (max_p + min_p)
        bbox = torch.cat((centre - dilation_factor * half_widths, centre + dilation_factor * half_widths))
        if bbox[0] < 0.0:
            bbox[0] = 0.0
        if bbox[1] < 0.0:
            bbox[1] = 0.0
        if bbox[2] > self.image_width:
            bbox[2] = self.image_width
        if bbox[3] > self.image_height:
            bbox[3] = self.image_height
        return bbox
        
    def __len__(self):
        return len(self.ids)

class IKEAKeypointRCNNTestDataset(torch.utils.data.Dataset):
    """ For standard dataset structure.
    <root>
    --train_cross_env.txt
    --test_cross_env.txt
    --Kallax_Shelf_Drawer
    ----0001_black_table_02_01_2019_08_16_14_00
    ------dev1
    ------dev2
    ------dev3
    --------images
    --------pose2d
    --------pose3d
    --------predictions
    ----...
    --Lack_Coffee_Table
    --Lack_Side_Table
    --Lack_TV_Bench
    """
    def __init__(self, root, split='test', cam='dev3', image_transform=None):
        self.root = root

        # Read split files:
        if split == 'test':
            with open(os.path.join(self.root, 'test_cross_env.txt'), 'r') as f:
                assembly_dirs = sorted(f.read().splitlines())
        elif split == 'train':
            with open(os.path.join(self.root, 'train_cross_env.txt'), 'r') as f:
                assembly_dirs = sorted(f.read().splitlines())

        cam_dirs = [os.path.join(assembly_dir, cam) for assembly_dir in assembly_dirs]
        image_dirs = [os.path.join(cam_dir, 'images') for cam_dir in cam_dirs]
        image_paths = []
        for image_dir in image_dirs:
            image_mask = os.path.join(self.root, image_dir, '*.png')
            image_paths.extend(sorted(glob(image_mask)))
        self.image_paths = image_paths

        self.joint_names = get_ikea_joint_names()
        self.num_joints = len(self.joint_names)

        # Dataset images have uniform sizes:
        image = Image.open(self.image_paths[0])
        width, height = image.size
        self.image_width = width
        self.image_height = height

        self.image_transform = image_transform

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')

        # Data augmentation (image only):
        if not self.image_transform:
            # Basic preprocessing:
            to_tensor = transforms.ToTensor() # converts to (C x H x W) in the range [0.0, 1.0]
            image = to_tensor(image)
        else:
            image = self.image_transform(image)

        return image, image_path
        
    def __len__(self):
        return len(self.image_paths)

class IKEAKeypointRCNNVideoTestDataset(torch.utils.data.Dataset):
    """ For standard dataset structure.
    Returns full video paths rather than images.
    """
    def __init__(self, root, split='test', cam='dev3'):
        self.root = root

        # Read split files:
        if split == 'test':
            with open(os.path.join(self.root, 'test_cross_env.txt'), 'r') as f:
                assembly_dirs = sorted(f.read().splitlines())
        elif split == 'train':
            with open(os.path.join(self.root, 'train_cross_env.txt'), 'r') as f:
                assembly_dirs = sorted(f.read().splitlines())

        cam_dirs = [os.path.join(assembly_dir, cam) for assembly_dir in assembly_dirs]
        image_dirs = [os.path.join(cam_dir, 'images') for cam_dir in cam_dirs]
        
        self.video_paths = [os.path.join(self.root, image_dir, 'scan_video.avi') for image_dir in image_dirs]

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        return video_path
        
    def __len__(self):
        return len(self.video_paths)


###########################################################################################################
# Utility functions:

def split_dataset(args):
    """Utility for splitting dataset into test and train folders
    """

    # Make directory structure:
    os.makedirs(args.dataset_dir, exist_ok=True)
    train_path = os.path.join(args.dataset_dir, 'train')
    os.makedirs(train_path, exist_ok=True)
    test_path = os.path.join(args.dataset_dir, 'test')
    os.makedirs(test_path, exist_ok=True)
    
    # Copy split files:
    train_file = 'train_cross_env.txt'
    test_file = 'test_cross_env.txt'
    shutil.copyfile(os.path.join(args.orig_dataset_dir, train_file), os.path.join(args.dataset_dir, train_file))
    shutil.copyfile(os.path.join(args.orig_dataset_dir, test_file), os.path.join(args.dataset_dir, test_file))

    # Read split files:
    with open(os.path.join(args.dataset_dir, 'test_cross_env.txt'), 'r') as f:
        test_paths = f.read().splitlines()
    with open(os.path.join(args.dataset_dir, 'train_cross_env.txt'), 'r') as f:
        train_paths = f.read().splitlines()

    # Copy image files and flatten structure:
    cam_dirs = get_cam_dirs(args.orig_dataset_dir, args.camera_id)
    with Pool(8) as p:
        for i, cam_dir in enumerate(cam_dirs):
            print(f"\nProcessing {i} of {len(cam_dirs)}: {' '.join(cam_dir.split('/')[-3:-1])}")
            
            furniture, video, camera = cam_dir.split('/')[-3:]
            video_path = '/'.join([furniture, video]) # eg Kallax_Shelf_Drawer/0001_black_table_02_01_2019_08_16_14_00
            cam_path = '/'.join([furniture, video, camera]) # eg Kallax_Shelf_Drawer/0001_black_table_02_01_2019_08_16_14_00/dev3

            pose2d_gt_filenames = os.listdir(os.path.join(cam_dir, 'pose2d'))
            frame_ids = [int(os.path.splitext(pose2d_gt_filename)[0]) for pose2d_gt_filename in pose2d_gt_filenames]

            if video_path in train_paths:
                output_path = os.path.join(args.dataset_dir, 'train')
                p.map(parallel_copy_image_pose2d, zip(frame_ids, itertools.repeat(args.orig_dataset_dir), itertools.repeat(cam_path), itertools.repeat(output_path)))
            elif video_path in test_paths:
                output_path = os.path.join(args.dataset_dir, 'test')
                p.map(parallel_copy_image_pose2d, zip(frame_ids, itertools.repeat(args.orig_dataset_dir), itertools.repeat(cam_path), itertools.repeat(output_path)))

def parallel_copy_image_pose2d(args):
    frame_id, input_path, cam_path, output_path = args
    frame_input_path = os.path.join(input_path, cam_path, "images", f"{frame_id:06}.png")
    frame_output_path = os.path.join(output_path, cam_path.replace("/", "_") + f"_{frame_id:06}.png")
    shutil.copyfile(frame_input_path, frame_output_path)
    pose2d_input_path = os.path.join(input_path, cam_path, "pose2d", f"{frame_id:06}.json")
    pose2d_output_path = os.path.join(output_path, cam_path.replace("/", "_") + f"_{frame_id:06}.json")
    shutil.copyfile(pose2d_input_path, pose2d_output_path)

def get_cam_dirs(input_path, camera_id='dev3', gt_dirs_only=True):
    """Get all directories with ground-truth 2D human pose annotations
    """
    cam_path_list = []
    category_path_list = get_subdirs(input_path)
    for category in category_path_list:
        if os.path.basename(category) != 'Calibration':
            category_scans = get_subdirs(category)
            for category_scan in category_scans:
                device_list = get_subdirs(category_scan)
                for device_path in device_list:
                    if camera_id in device_path:
                        if not gt_dirs_only:
                            cam_path_list.append(device_path)
                        else:
                            if os.path.exists(os.path.join(device_path, 'pose2d')): # 2D annotations exist
                                cam_path_list.append(device_path)
    return cam_path_list

def get_subdirs(input_path):
    subdirs = [os.path.join(input_path, dir_i) for dir_i in os.listdir(input_path)
               if os.path.isdir(os.path.join(input_path, dir_i))]
    subdirs.sort()
    return subdirs

def split_filename(filename):
    path, filename = os.path.split(filename)
    name, ext = os.path.splitext(filename)
    return path, name, ext

def resize_images(args, split='test', new_image_size=(640, 360)):
    image_mask = os.path.join(args.dataset_dir, split, '*.png')
    image_paths = glob(image_mask)
    for i, image_path in enumerate(image_paths):
        if i % 100 == 0:
            print(f"Processed {i} out of {len(image_paths)}")
        with open(image_path, 'rb') as f:
            image = Image.open(f).convert('RGB')
        w, h = image.size
        image = image.resize(new_image_size, Image.BICUBIC)
        image.save(os.path.join(image_path)) # Overwrites file!
        
        # Rescale annotations:
        x_scale = image.size[0] / w
        y_scale = image.size[1] / h
        json_path = os.path.splitext(image_path)[0] + '.json'
        with open(json_path, 'r') as json_file:
            pose2d_gt = json.load(json_file)
        for joint_name in pose2d_gt.keys():
            position = pose2d_gt[joint_name]["position"]
            pose2d_gt[joint_name]["position"][0] = position[0] * x_scale
            pose2d_gt[joint_name]["position"][1] = position[1] * y_scale
        with open(json_path, 'w') as outfile: # Overwrites file!
            json.dump(pose2d_gt, outfile)

def video_to_frames(args):
    """Convert videos to frames
    """
    cam_dirs = get_cam_dirs(args.dataset_dir, camera_id=args.camera_id, gt_dirs_only=False)
    for i, cam_dir in enumerate(cam_dirs):
        print(f"\nProcessing {i} of {len(cam_dirs)}: {' '.join(cam_dir.split('/')[-3:-1])}")
        image_folder = os.path.join(cam_dir, 'images')
        rgb_video_file = os.path.join(image_folder, 'scan_video.avi')
        command = ['ffmpeg',
                   '-i', rgb_video_file,
                   '-start_number', '0',
                   '-f', 'image2',
                   '-v', 'error',
                   f'{image_folder}/%06d.png']
        subprocess.call(command)

def video_to_keyframes(args):
    """Convert videos to frames and keep those with corresponding GT
    """
    cam_dirs = get_cam_dirs(args.dataset_dir, camera_id=args.camera_id, gt_dirs_only=False)
    for i, cam_dir in enumerate(cam_dirs):
        print(f"\nProcessing {i} of {len(cam_dirs)}: {' '.join(cam_dir.split('/')[-3:-1])}")
        image_folder = os.path.join(cam_dir, 'images')
        rgb_video_file = os.path.join(image_folder, 'scan_video.avi')
        pose2d_gt_dir = os.path.join(os.path.dirname(cam_dir), 'dev3', 'pose2d')
        if os.path.exists(pose2d_gt_dir):
            pose2d_gt_filenames = os.listdir(pose2d_gt_dir) # dev3 has annotations
            for pose2d_gt_filename in pose2d_gt_filenames:
                frame_id = int(os.path.splitext(pose2d_gt_filename)[0])
                command = ['ffmpeg',
                           '-i', rgb_video_file,
                           '-f', 'image2',
                           '-v', 'error',
                           '-vf', f"select='eq(n\,{frame_id})'",
                           f'{image_folder}/{frame_id:06d}.png']
                subprocess.call(command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str,
                        help='directory of the IKEA assembly dataset (copy)')
    parser.add_argument('--orig_dataset_dir', type=str,
                        help='directory of the IKEA assembly dataset (original)')
    parser.add_argument('--camera_id', type=str, default='dev3',
                        help='camera device ID for dataset (GT annotations for dev3 only)')
    args = parser.parse_args()

    split_dataset(args)
    resize_images(args, split='test') # Overwrites files
    resize_images(args, split='train') # Overwrites files