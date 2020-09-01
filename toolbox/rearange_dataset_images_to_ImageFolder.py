# Author: Yizhak Ben-Shabat (Itzik), 2020
# go over all of the dataset images and rearange them to comply with pytorch ImageFolder structure for the data loader

import os
import numpy as np
import sys
sys.path.append(os.path.abspath('../action/'))
from IKEAActionDataset import IKEAActionDataset
import shutil
from tqdm import tqdm
from multiprocessing import Pool
import itertools
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller',
                    help='path to raw image IKEA ASM dataset dir after images were resized')
parser.add_argument('--output_path', type=str, default='/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller_ImageFolder',
                    help='ImageFolder structure of the dataset')
args = parser.parse_args()

dataset_path = args.dataset_path
output_path = args.output_path

def par_copy(arguments):
    frame_num, input_path, video_path, output_class_path = arguments
    frame_input_path = os.path.join(input_path, video_path, str(frame_num).zfill(6) + ".jpg")
    frame_output_path = os.path.join(output_class_path,
                                     video_path.replace("/", "_") + "_" + str(frame_num).zfill(6) + ".jpg")
    shutil.copyfile(frame_input_path, frame_output_path)


def save_cls_images(frame_list, label_list, video_path, phase, num_classes, dataset_path, output_path, action_dict, p):
    # copy images from original dataset to imagefolder structure

    for cls in range(0, num_classes):
        frames = frame_list[np.where(label_list == cls)]
        cls_str = action_dict[cls]
        if not len(frames) == 0:
            output_class_path = os.path.join(output_path, phase, cls_str)
            os.makedirs(output_class_path, exist_ok=True)

            p.map(par_copy, zip(frames, itertools.repeat(dataset_path), itertools.repeat(video_path),
                                    itertools.repeat(output_class_path)))

os.makedirs(output_path, exist_ok=True)
train_path = os.path.join(output_path, 'train')
os.makedirs(train_path, exist_ok=True)
test_path = os.path.join(output_path, 'test')
os.makedirs(test_path, exist_ok=True)
db_file = 'ikea_annotation_db_full'
train_file = 'train_cross_env.txt'
test_file = 'test_cross_env.txt'
action_list_file = 'atomic_action_list.txt'
action_object_relation_file = 'action_object_relation_list.txt'
dataset = IKEAActionDataset(dataset_path, db_file, action_list_file, action_object_relation_file, train_file, test_file)


dataset_name = 'ANU_ikea_dataset'

trainset_videos = dataset.trainset_video_list
testset_videos = dataset.testset_video_list
action_dict = dataset.action_list

n_name_chars = len(dataset_name) + 1

num_classes = dataset.num_classes
cursor_vid = dataset.get_annotated_videos_table(device='dev3')
rows = cursor_vid.fetchall()
with Pool(8) as p:
    with tqdm(total=len(rows), file=sys.stdout) as pbar:
        for row in rows:
            pbar.update(1)
            n_frames = int(row["nframes"])
            frame_list = np.arange(0, n_frames)
            label_list = np.zeros_like(frame_list)
            video_idx = row['id']
            video_path = row['video_path']
            video_name = os.path.join(video_path.split('/')[0], video_path.split('/')[1])
            if video_name in trainset_videos or video_name in testset_videos:
                cursor_annotations = dataset.get_video_annotations_table(video_idx)

                for ann_row in cursor_annotations:
                    action_id = dataset.get_action_id(ann_row["atomic_action_id"], ann_row["object_id"])   # no need to +1 because table index starts at 1
                    if action_id is not None:
                        label_list[ann_row['starting_frame']:ann_row['ending_frame']] = action_id


                if video_name in trainset_videos:
                    # trainset
                    save_cls_images(frame_list, label_list, video_path, 'train', num_classes,
                                    dataset_path, output_path, action_dict, p)

                elif video_name in testset_videos:
                    # testset
                    save_cls_images(frame_list, label_list, video_path, 'test', num_classes,
                                    dataset_path, output_path, action_dict, p)



