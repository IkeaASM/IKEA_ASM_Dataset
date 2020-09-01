# Class for IKEA 2D pose annotations
#
# <openpose root>/lib/datasets/
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
import torch
import torchvision.transforms as transforms
from glob import glob
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool

#from pdb import set_trace as st

from .heatmap import putGaussianMaps
from .paf import putVecMaps
from . import transforms, utils

def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('neck'), keypoints.index('right_hip')],  
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('neck'), keypoints.index('left_hip')],                
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('neck'), keypoints.index('right_shoulder')],          
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],     
        [keypoints.index('right_shoulder'), keypoints.index('right_eye')],        
        [keypoints.index('neck'), keypoints.index('left_shoulder')], 
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_eye')],               
        [keypoints.index('neck'), keypoints.index('nose')],                      
        [keypoints.index('nose'), keypoints.index('right_eye')],
        [keypoints.index('nose'), keypoints.index('left_eye')],        
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')]
    ]
    return kp_lines
    
def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    # Keypoints are not available in the COCO json for the test split, so we
    # provide them here.
    keypoints = [
        'nose',
        'neck',
        'right_shoulder',
        'right_elbow',
        'right_wrist',   
        'left_shoulder',
        'left_elbow',
        'left_wrist',
        'right_hip',
        'right_knee',
        'right_ankle',
        'left_hip',
        'left_knee',
        'left_ankle',
        'right_eye',                                                                    
        'left_eye',
        'right_ear',
        'left_ear']
    return keypoints

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

class IKEAPose2dDataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile=None, image_transform=None, target_transforms=None,
                 n_images=None, preprocess=None, all_images=False, all_persons=False, input_y=368, input_x=368, stride=8):
        
        self.root = root
        image_mask = os.path.join(self.root, '*.png')
        image_paths = sorted(glob(image_mask))
        self.ids = [split_filename(filename)[1] for filename in image_paths]
        print(f'Num images: {len(self.ids)}')

        self.image_size = Image.open(image_paths[0]).size # Assumes all image of equal size

        # Load all annotations:
        joint_names = get_keypoints()
        self.annotations = []
        for id in self.ids:
            json_path = os.path.join(self.root, id + '.json')
            with open(json_path, 'r') as json_file:
                pose2d_gt = json.load(json_file)
            keypoints = [] # 17*3 list
            for joint_name in get_ikea_joint_names():
                position = pose2d_gt[joint_name]["position"]
                confidence = pose2d_gt[joint_name]["confidence"]
                keypoints.extend(position + [confidence])
            self.annotations.append({"keypoints": keypoints, "bbox": [20, 20, 100, 100], "segmentation": None}) # Dummy values for bbox and segmentation

        self.preprocess = preprocess or transforms.Normalize()
        self.image_transform = image_transform or transforms.image_transform
        self.target_transforms = target_transforms
        
        self.HEATMAP_COUNT = len(get_keypoints())
        self.LIMB_IDS = kp_connections(get_keypoints())
        self.input_y = input_y
        self.input_x = input_x        
        self.stride = stride
        self.log = logging.getLogger(self.__class__.__name__)

    def __getitem__(self, index):
        image_id = self.ids[index]
        anns = [self.annotations[index]] # annotations only stores joints for a single person
        anns = copy.deepcopy(anns)

        image_filename = image_id + ".png"
        self.log.debug(image_filename)
        with open(os.path.join(self.root, image_filename), 'rb') as f:
            image = Image.open(f).convert('RGB')

        meta_init = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_filename,
        }

        image, anns, meta = self.preprocess(image, anns, None)
             
        if isinstance(image, list):
            return self.multi_image_processing(image, anns, meta, meta_init)
        return self.single_image_processing(image, anns, meta, meta_init)

    def multi_image_processing(self, image_list, anns_list, meta_list, meta_init):
        return list(zip(*[
            self.single_image_processing(image, anns, meta, meta_init)
            for image, anns, meta in zip(image_list, anns_list, meta_list)
        ]))

    def single_image_processing(self, image, anns, meta, meta_init):
        meta.update(meta_init)

        # transform image
        original_size = image.size
        image = self.image_transform(image) # color_jitter, jpeg_compression_augmentation, RandomGrayscale, Normalize (vgg means/std)
        assert image.size(2) == original_size[0]
        assert image.size(1) == original_size[1]

        # mask valid
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        self.log.debug(meta)

        heatmaps, pafs = self.get_ground_truth(anns)
        heatmaps = torch.from_numpy(heatmaps.transpose((2, 0, 1)).astype(np.float32))
        pafs = torch.from_numpy(pafs.transpose((2, 0, 1)).astype(np.float32))       
        return image, heatmaps, pafs, meta['image_id']

    def remove_illegal_joint(self, keypoints):

        MAGIC_CONSTANT = (-1, -1, 0)
        mask = np.logical_or.reduce((keypoints[:, :, 0] >= self.input_x,
                                     keypoints[:, :, 0] < 0,
                                     keypoints[:, :, 1] >= self.input_y,
                                     keypoints[:, :, 1] < 0))
        keypoints[mask] = MAGIC_CONSTANT

        return keypoints
        
    def add_neck(self, keypoint):
        '''
        MS COCO annotation order:
        0: nose         1: l eye        2: r eye    3: l ear    4: r ear
        5: l shoulder   6: r shoulder   7: l elbow  8: r elbow
        9: l wrist      10: r wrist     11: l hip   12: r hip   13: l knee
        14: r knee      15: l ankle     16: r ankle
        The order in this work:
        (0-'nose'   1-'neck' 2-'right_shoulder' 3-'right_elbow' 4-'right_wrist'
        5-'left_shoulder' 6-'left_elbow'        7-'left_wrist'  8-'right_hip'
        9-'right_knee'   10-'right_ankle'   11-'left_hip'   12-'left_knee'
        13-'left_ankle'  14-'right_eye'     15-'left_eye'   16-'right_ear'
        17-'left_ear' )
        '''
        our_order = [0, 17, 6, 8, 10, 5, 7, 9,
                     12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        # Index 6 is right shoulder and Index 5 is left shoulder
        right_shoulder = keypoint[6, :]
        left_shoulder = keypoint[5, :]
        neck = (right_shoulder + left_shoulder) / 2
        if right_shoulder[2] == 2 and left_shoulder[2] == 2:
            neck[2] = 2
        else:
            neck[2] = right_shoulder[2] * left_shoulder[2]

        neck = neck.reshape(1, len(neck))
        neck = np.round(neck)
        keypoint = np.vstack((keypoint, neck))
        keypoint = keypoint[our_order, :]

        return keypoint
                
    def get_ground_truth(self, anns):
    
        grid_y = int(self.input_y / self.stride)
        grid_x = int(self.input_x / self.stride)
        channels_heat = (self.HEATMAP_COUNT + 1)
        channels_paf = 2 * len(self.LIMB_IDS)
        heatmaps = np.zeros((int(grid_y), int(grid_x), channels_heat))
        pafs = np.zeros((int(grid_y), int(grid_x), channels_paf))

        keypoints = []
        for ann in anns:
            single_keypoints = np.array(ann['keypoints']).reshape(17,3)
            single_keypoints = self.add_neck(single_keypoints)
            keypoints.append(single_keypoints)
        keypoints = np.array(keypoints)
        keypoints = self.remove_illegal_joint(keypoints)

        # confidance maps for body parts
        for i in range(self.HEATMAP_COUNT):
            joints = [jo[i] for jo in keypoints]
            for joint in joints:
                if joint[2] > 0.5: # if confidence > 0.5 (always true for our annotations)
                    center = joint[:2]
                    gaussian_map = heatmaps[:, :, i]
                    heatmaps[:, :, i] = putGaussianMaps(
                        center, gaussian_map,
                        7.0, grid_y, grid_x, self.stride)
        # pafs
        for i, (k1, k2) in enumerate(self.LIMB_IDS):
            # limb
            count = np.zeros((int(grid_y), int(grid_x)), dtype=np.uint32)
            for joint in keypoints:
                if joint[k1, 2] > 0.5 and joint[k2, 2] > 0.5:
                    centerA = joint[k1, :2]
                    centerB = joint[k2, :2]
                    vec_map = pafs[:, :, 2 * i:2 * (i + 1)]

                    pafs[:, :, 2 * i:2 * (i + 1)], count = putVecMaps(
                        centerA=centerA,
                        centerB=centerB,
                        accumulate_vec_map=vec_map,
                        count=count, grid_y=grid_y, grid_x=grid_x, stride=self.stride
                    )

        # background
        heatmaps[:, :, -1] = np.maximum(
            1 - np.max(heatmaps[:, :, :self.HEATMAP_COUNT], axis=2),
            0.
        )
        return heatmaps, pafs
        
    def __len__(self):
        return len(self.ids)
