import os
import sys
sys.path.append('../')
from IKEAActionDataset import IKEAActionVideoDataset as Dataset
import torch
import argparse
from visualization import vis_utils

parser = argparse.ArgumentParser()
parser.add_argument('-db_filename', type=str,
                    default='/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller_video/ikea_annotation_db_full',
                    help='database file')
parser.add_argument('-dataset_path', type=str,
                    default='/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller_video/', help='path to dataset')
parser.add_argument('-output_path', type=str,
                    default='./vis_output/', help='path to dataset')
parser.add_argument('-example', type=int, default=0, help='example index to visualize')
args = parser.parse_args()

output_path = args.output_path
os.makedirs(output_path, exist_ok=True)
batch_size = 64
dataset_path = args.dataset_path
db_filename = args.db_filename
train_filename = 'ikea_trainset.txt'
testset_filename = 'ikea_testset.txt'
idx = args.example
dataset = Dataset(dataset_path, db_filename=db_filename, train_filename=train_filename,
                       transform=None, set='test', camera='dev3', frame_skip=1,
                       frames_per_clip=64, resize=None)

video_set = dataset.video_set
vis_utils.vis_segment_vid(vid_path=video_set[idx][0], gt_labels=video_set[idx][1], action_list=dataset.action_list,
                          output_path=output_path, output_filename='action_segmentation.mp4')
# vis_utils.vis_segment_vid_compare(video_set[idx][0], video_set[idx][1], video_set[idx][1], dataset.action_list)

