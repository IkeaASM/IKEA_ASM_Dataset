# Run inference on dataset
# Pre-trained OpenPose detector (default configuration)
# Fine-tuned OpenPose detector (PyTorch implementation)
# Pre-trained OpenPose detector (video - STAF)
# Pre-trained VIBE detector (demo)
# Pre-trained HMMR detector (demo)
#
# Dylan Campbell <dylan.campbell@anu.edu.au>

import os
import argparse
import subprocess
import json
import numpy as np
from glob import glob
from pdb import set_trace as st

def run_inference_openpose(args):
    image_folders = get_image_dirs(args.dataset_dir, args.camera_id)
    for i, image_folder in enumerate(image_folders):
        print(f"\nProcessing {i} of {len(image_folders)}: {' '.join(image_folder.split('/')[-4:-2])}")
        video_path = os.path.join(image_folder, 'scan_video.avi')
        output_dir = os.path.join(os.path.dirname(image_folder), 'predictions', 'pose2d', 'openpose')
        os.makedirs(output_dir, exist_ok=True)
        openpose_binary = os.path.join(args.openpose_dir, 'build/examples/openpose/openpose.bin')
        model_folder = os.path.join(args.openpose_dir, 'models')
        cmd = [
            openpose_binary,
            "--model_pose", "BODY_25",
            "--model_folder", model_folder,
            "--display", "0",
            "--render_pose", "0",
            "--video", video_path,
            "--write_json", output_dir,
            "--write_coco_json", os.path.join(output_dir, 'output_coco_format.json')
            ]
        print(' '.join(cmd))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        subprocess.call(cmd)

def run_inference_openpose_staf(args):
    image_folders = get_image_dirs(args.dataset_dir, args.camera_id)
    for i, image_folder in enumerate(image_folders):
        print(f"\nProcessing {i} of {len(image_folders)}: {' '.join(image_folder.split('/')[-4:-2])}")
        video_path = os.path.join(image_folder, 'scan_video.avi')
        output_dir = os.path.join(os.path.dirname(image_folder), 'predictions', 'pose2d', 'openpose_staf')
        os.makedirs(output_dir, exist_ok=True)
        openpose_staf_binary = os.path.join(args.openpose_staf_dir, 'build/examples/openpose/openpose.bin')
        model_folder = os.path.join(args.openpose_staf_dir, 'models')
        cmd = [
            openpose_staf_binary,
            "--model_pose", "BODY_21A",
            "--tracking", "1",
            "--model_folder", model_folder,
            "--display", "0",
            "--render_pose", "0",
            "--video", video_path,
            "--write_json", output_dir,
            ]
        print(' '.join(cmd))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        subprocess.call(cmd)

def run_inference_openpose_pytorch(args):
    runfile = os.path.join(args.openpose_pytorch_dir, 'demo', 'ikea_demo.py')
    image_folders = get_image_dirs(args.dataset_dir, args.camera_id)

    cmd = [
        "python3", runfile,
        "--image_folders", ",".join(image_folders),
        "--cfg", os.path.join(args.openpose_pytorch_dir, "experiments/vgg19_368x368_sgd.yaml"),
        "--weight", "./weights/openpose_pytorch.pth",
        "--openpose_dir", args.openpose_pytorch_dir,
    ]
    if args.all_frames:
        cmd.append("--all_frames")
    print(' '.join(cmd))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    subprocess.call(cmd)

def run_inference_vibe(args):
    image_folders = get_image_dirs(args.dataset_dir, args.camera_id)
    for i, image_folder in enumerate(image_folders):
        print(f"\nProcessing {i} of {len(image_folders)}: {' '.join(image_folder.split('/')[-4:-2])}")
        video_path = os.path.join(image_folder, 'scan_video.avi')
        output_dir = os.path.join(os.path.dirname(image_folder), 'predictions', 'pose3d', 'vibe')
        os.makedirs(output_dir, exist_ok=True)
        vibe_file = os.path.join(args.vibe_dir, 'demo.py')
        if args.vibe_run_smplify:
            cmd = [
                "python", vibe_file,
                "--no_render",
                "--tracking_method", args.vibe_tracking_method,
                "--staf_dir", args.openpose_staf_dir,
                "--detector", args.vibe_detector,
                "--run_smplify",
                "--vid_file", video_path,
                "--output_folder", output_dir
                ]
        else:
            cmd = [
                "python", vibe_file,
                "--no_render",
                "--tracking_method", args.vibe_tracking_method,
                "--staf_dir", args.openpose_staf_dir,
                "--detector", args.vibe_detector,
                "--vid_file", video_path,
                "--output_folder", output_dir
                ]
        print(' '.join(cmd))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        subprocess.call(cmd)

def run_inference_hmmr(args):
    image_folders = get_image_dirs(args.dataset_dir, args.camera_id)
    for i, image_folder in enumerate(image_folders):
        print(f"\nProcessing {i} of {len(image_folders)}: {' '.join(image_folder.split('/')[-4:-2])}")
        video_path = os.path.join(image_folder, 'scan_video.avi')
        output_dir = os.path.join(os.path.dirname(image_folder), 'predictions', 'pose3d', 'hmmr')
        os.makedirs(output_dir, exist_ok=True)
        hmmr_file = os.path.join(args.hmmr_dir, 'demo_video_ikea.py')
        cmd = [
            "python", hmmr_file,
            "--vid_path", video_path,
            "--out_dir", output_dir,
            "--track_dir", output_dir
            ]
        print(' '.join(cmd))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        cwd = os.getcwd()
        os.chdir(args.hmmr_dir)
        subprocess.call(cmd)
        os.chdir(cwd)

def get_image_dirs(input_path, camera_id='dev3'):
    '''
    get_dirs retrieves all image directories under the dataset main directories:
    :param input_path: path to ANU IKEA Dataset directory

    :return:
    scan_path_list: path to all available scans
    category_path_list: path to all available categories
    '''

    category_path_list = get_subdirs(input_path)
    scan_path_list = []
    for category in category_path_list:
        if os.path.basename(category) != 'Calibration':
            category_scans = get_subdirs(category)
            for category_scan in category_scans:
                scan_path_list.append(category_scan)

    rgb_path_list = []
    for scan in scan_path_list:
        device_list = get_subdirs(scan)
        for device_path in device_list:
            if camera_id in device_path:
                rgb_path = os.path.join(device_path, 'images')
                if os.path.exists(rgb_path):
                    rgb_path_list.append(rgb_path)

    return rgb_path_list

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_openpose', action='store_true',
                        help='run OpenPose inference')
    parser.add_argument('--run_openpose_staf', action='store_true',
                        help='run OpenPose video inference')
    parser.add_argument('--run_openpose_pytorch', action='store_true',
                        help='run OpenPose PyTorch inference')
    parser.add_argument('--run_vibe', action='store_true',
                        help='run VIBE inference')
    parser.add_argument('--run_hmmr', action='store_true',
                        help='run HMMR inference')
    parser.add_argument('--dataset_dir', type=str, default='/home/djcam/Documents/HDD/datasets/ikea/ikea_asm/',
                        help='directory of the IKEA assembly dataset')
    parser.add_argument('--openpose_dir', type=str, default='/home/djcam/Documents/code/human_pose/openpose/',
                        help='directory of the OpenPose detector')
    parser.add_argument('--openpose_staf_dir', type=str, default='/home/djcam/Documents/code/human_pose/openpose_staf/',
                        help='directory of the STAF tracker')
    parser.add_argument('--openpose_pytorch_dir', type=str, default='/home/djcam/Documents/code/human_pose/openpose_pytorch/',
                        help='directory of the OpenPose PyTorch detector')
    parser.add_argument('--vibe_dir', type=str, default='/home/djcam/Documents/code/human_pose/VIBE/',
                        help='directory of the VIBE detector')
    parser.add_argument('--vibe_tracking_method', type=str, default='pose', choices=['bbox', 'pose'],
                        help='tracking method to calculate the tracklet of a subject from the input video')
    parser.add_argument('--vibe_detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='object detector to be used for bbox tracking')
    parser.add_argument('--vibe_run_smplify', action='store_true',
                        help='run smplify for refining the results, you need pose tracking to enable it')
    parser.add_argument('--hmmr_dir', type=str, default='/home/djcam/Documents/code/human_pose/human_dynamics/',
                        help='directory of the HMMR detector')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device ID')
    parser.add_argument('--camera_id', type=str, default='dev3',
                        help='camera device ID for dataset (GT annotations for dev3 only)')
    parser.add_argument('--all_frames', action='store_true',
                        help='run detector on all frames in video')
    parser.add_argument('--split_id', type=int, default=0)
    args = parser.parse_args()

    if args.run_openpose:
        run_inference_openpose(args)
    if args.run_openpose_staf:
        run_inference_openpose_staf(args)
    if args.run_openpose_pytorch:
        run_inference_openpose_pytorch(args)
    if args.run_vibe:
        run_inference_vibe(args)
    if args.run_hmmr:
        run_inference_hmmr(args)