# Triangulate 3D joints
#
# Dylan Campbell <dylan.campbell@anu.edu.au>

import os
import cv2 as cv
import argparse
import json
import pickle
import subprocess
import numpy as np
from glob import glob
import itertools
from multiprocessing import Pool
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from joint_ids import *
from dataset_ids import *

from pdb import set_trace as st

np.seterr(all='raise')

scenes = get_scenes()
cams = get_cams()

connectivity_ikea = get_ikea_connectivity() # == COCO format
connectivity_body25 = get_body25_connectivity()

def triangulate_joints(args):
    camera_parameters = get_camera_parameters(args)

    scan_folders = get_scan_dirs(args.dataset_dir)

    # Select save format:
    if args.save_format == 'ikea':
        joint_names = get_ikea_joint_names()
        connectivity = connectivity_ikea
    elif args.save_format == 'body25':
        joint_names = get_body25_joint_names()[:25]
        connectivity = connectivity_body25

    if 'keypoint_rcnn' in args.input_predictions:
        input_joint_names = get_ikea_joint_names()
    elif 'openpose' in args.input_predictions:
        input_joint_names = get_body25_joint_names()
    input_joint_names_dict = {name: i for i, name in enumerate(input_joint_names)}

    num_joints = len(joint_names)

    with open(os.path.join(args.dataset_dir, 'test_cross_env.txt'), 'r') as f:
        test_paths = f.read().splitlines()
    with open(os.path.join(args.dataset_dir, 'train_cross_env.txt'), 'r') as f:
        train_paths = f.read().splitlines()

    reproj_gt_meter = AverageMeter('Reprojection error')
    reproj_gt_meter_train = AverageMeter('Reprojection error')
    reproj_gt_meter_test = AverageMeter('Reprojection error')
    pck_gt_meter = AverageMeter('PCK')
    pck_gt_meter_train = AverageMeter('PCK')
    pck_gt_meter_test = AverageMeter('PCK')

    for i, scan_folder in enumerate(scan_folders):
        print(f"\nProcessing {i} of {len(scan_folders)}: {' '.join(scan_folder.split('/')[-2:])}")

        # Determine scene ID:
        label = scan_folder.split('/')[-1]
        scene = label.split('_')[3]
        assert scene in scenes

        prediction_path = os.path.join('predictions', 'pose2d', args.input_predictions)
        
        use_all_frames = False
        if use_all_frames:
            # Use all frames:
            # Check all cams, since some have more frames than others...
            json_mask = os.path.join(scan_folder, 'dev1', prediction_path, 'scan_video_????????????_keypoints.json')
            json_files1 = glob(json_mask)
            json_mask = os.path.join(scan_folder, 'dev2', prediction_path, 'scan_video_????????????_keypoints.json')
            json_files2 = glob(json_mask)
            json_mask = os.path.join(scan_folder, 'dev3', prediction_path, 'scan_video_????????????_keypoints.json')
            json_files3 = glob(json_mask)
            json_index = np.argmin([len(json_files1), len(json_files2), len(json_files3)])
            json_files = [json_files1, json_files2, json_files3][json_index]
            keypoint_filenames = sorted([os.path.basename(json_file) for json_file in json_files])
        else:
            # Use frames with GT 2D annotations:
            json_mask = os.path.join(scan_folder, 'dev3', 'pose2d', '??????.json') # GT 2D annotations
            json_files = sorted(glob(json_mask)) # eg <root>/<scan_folder>/dev3/pose2d/000000.json
            frame_strs = [os.path.splitext(os.path.basename(json_file))[0] for json_file in json_files] # eg 000000
            keypoint_filenames = sorted([f'scan_video_000000{frame_str}_keypoints.json' for frame_str in frame_strs]) # eg scan_video_000000000000_keypoints.json

        for file_index, keypoint_filename in enumerate(keypoint_filenames):
            joints2d = np.zeros((num_joints, 3, 3))
            for cam in cams:
                json_file = os.path.join(scan_folder, cam, prediction_path, keypoint_filename)
                if not os.path.exists(json_file):
                    continue # Predictions don't exist (missing video frame)
                with open(json_file) as f:
                    pose2d = json.load(f)
                if len(pose2d["people"]) == 0:
                    continue
                keypoints = None
                # max_length = 0.0
                max_score = -np.Inf
                # Choose highest scoring person in frame:
                for person_id, person in enumerate(pose2d["people"]):
                    kps = np.array(person["pose_keypoints_2d"]).reshape(-1, 3) # [x1, y1, c1], [x2, ... in COCO or body25 format
                    average_score = np.mean(kps[:, 2])
                    if average_score > max_score:
                        max_score = average_score
                        keypoints = kps
                # Convert to ikea joints:
                for j, joint_name in enumerate(joint_names):
                    joint_id = input_joint_names_dict[joint_name]
                    joints2d[j, cams.index(cam), :] = keypoints[joint_id, :]
            # Undistort points:
            do_undistort = False
            if do_undistort:
                for cam_id, cam in enumerate(cams):
                    joints2d_cam = joints2d[:, cam_id, :2] # 17x2
                    K = camera_parameters[scene][cam]["K"]
                    dist_coefs = camera_parameters[scene][cam]["dist_coefs"]
                    joints2d_cam_undistorted = cv.undistortPoints(joints2d_cam.T, K, dist_coefs, None, None, K).squeeze() # input 2xN/Nx2, output 1xN/Nx1 2-channel
                    joints2d[:, cam_id, :2] = joints2d_cam_undistorted

            # Loop over joints:
            joints3d = np.zeros((num_joints, 4)) # 17x4
            for j in range(num_joints):
                joint2d = joints2d[j, :, :] # 3x3
                if np.count_nonzero(joint2d[:, 2] >= args.score_threshold) < 2: # Skip if insufficient good detections for triangulation
                    continue

                Ps = []
                xs = []
                C = 1.0
                for cam_id, cam in enumerate(cams):
                    if joint2d[cam_id, 2] >= args.score_threshold:
                        Ps.append(camera_parameters[scene][cam]["P"].astype(float))
                        xs.append(np.array([joint2d[cam_id, 0], joint2d[cam_id, 1], 1.0]).astype(float)) # homogeneous
                        C *= joint2d[cam_id, 2]

                if len(Ps) == 2:
                    # Triangulate points from 2 views:
                    X = cv.triangulatePoints(Ps[0], Ps[1], xs[0][:2], xs[1][:2]) # dev1+dev3 (preferred pair)
                    X /= X[3]
                    X = X.squeeze()
                    X[3] = C
                else:
                    # Triangulate from all 2-view pairs and average (suboptimal):
                    X1 = cv.triangulatePoints(Ps[0], Ps[1], xs[0][:2], xs[1][:2]) # dev1+dev2
                    X2 = cv.triangulatePoints(Ps[0], Ps[2], xs[0][:2], xs[2][:2]) # dev1+dev3
                    X3 = cv.triangulatePoints(Ps[1], Ps[2], xs[1][:2], xs[2][:2]) # dev2+dev3
                    X1 /= X1[3]
                    X2 /= X2[3]
                    X3 /= X3[3]
                    X1 = X1.squeeze()
                    X2 = X2.squeeze()
                    X3 = X3.squeeze()

                    X = np.mean((X1, X2, X3), axis=0)
                    X[3] = C

                joints3d[j, :] = X

            # Filter any points that are far from the median of the others (dimension-wise):
            non_zero_indices = joints3d[:, 3] > 0.0
            if non_zero_indices.any(): # At least one joint
                joints3d_median = np.median(joints3d[non_zero_indices, :], axis=0) # excluding zeros
                error = np.abs(joints3d[:, :3] - joints3d_median[:3])
                for j in range(num_joints):
                    if joints3d[j, 3] > 0.0 and any(error[j, :] > args.distance_to_median_threshold): # 200 cm
                        joints3d[j, :] = np.zeros(4)

            # Filter any points that are far from any other point:
            for j in range(num_joints):
                if joints3d[j, 3] > 0.0:
                    distances = []
                    for j2 in range(num_joints):
                        if joints3d[j2, 3] > 0.0 and j != j2:
                            distances.append(np.linalg.norm(joints3d[j, :3] - joints3d[j2, :3]))
                    if distances and np.array(distances).min() > args.distance_to_closest_threshold: # 100 cm
                        joints3d[j, :] = np.zeros(4)

            # Compute reprojection errors in each view:
            # Discard joints with large reprojection error in any view
            for cam_id, cam in enumerate(cams):
                P = camera_parameters[scene][cam]["P"]
                for j in range(num_joints):
                    if joints3d[j, 3] > 0.0: # Skip joints that were not triangulated
                        if joints2d[j, cam_id, 2] > args.score_threshold: # Skip 2D joints that were not well detected
                            x2d = joints2d[j, cam_id, :2]
                            x3dproj = P @ np.array([joints3d[j, 0], joints3d[j, 1], joints3d[j, 2], 1.0]) # Project to 2D
                            x3dproj /= x3dproj[2]
                            x3dproj = x3dproj[:2]
                            reprojection_error = np.linalg.norm(x2d - x3dproj)
                            # print(f"{cam} {joint_names[j]} \t\t {reprojection_error:3.1f}")
                            if reprojection_error > args.reprojection_threshold:
                                joints3d[j, :] = np.zeros(4)

            # Compare against GT 2D annotations:
            # Discard joints with large reprojection error in this view
            if not use_all_frames:
                with open(json_files[file_index]) as f:
                    pose2d_gt = json.load(f)
                cam = 'dev3'
                P = camera_parameters[scene][cam]["P"]
                # GET GT PARAMS
                for j in range(num_joints):
                    if joints3d[j, 3] > 0.0: # Skip joints that were not triangulated
                        joint_name = get_ikea_joint_names()[j]
                        position_gt = np.array(pose2d_gt[joint_name]["position"])
                        confidence_gt = pose2d_gt[joint_name]["confidence"]
                        x2d = position_gt
                        x3dproj = P @ np.array([joints3d[j, 0], joints3d[j, 1], joints3d[j, 2], 1.0]) # Project to 2D
                        x3dproj /= x3dproj[2]
                        x3dproj = x3dproj[:2]
                        reprojection_error = np.linalg.norm(x2d - x3dproj)
                        # print(f"{cam} {joint_names[j]} \t\t {reprojection_error:3.1f}")
                        if reprojection_error > args.reprojection_threshold:
                            joints3d[j, :] = np.zeros(4)

                # Quantify pseudoGT error:
                for j in range(num_joints):
                    if joints3d[j, 3] > 0.0: # Skip joints that were not triangulated
                        joint_name = get_ikea_joint_names()[j]
                        position_gt = np.array(pose2d_gt[joint_name]["position"])
                        confidence_gt = pose2d_gt[joint_name]["confidence"]
                        x2d = position_gt
                        x3dproj = P @ np.array([joints3d[j, 0], joints3d[j, 1], joints3d[j, 2], 1.0]) # Project to 2D
                        x3dproj /= x3dproj[2]
                        x3dproj = x3dproj[:2]
                        reprojection_error = np.linalg.norm(x2d - x3dproj)

                        if confidence_gt == 3:
                            reproj_gt_meter.update(reprojection_error, 1)
                            if '/'.join(scan_folder.split('/')[-2:]) in train_paths:
                                reproj_gt_meter_train.update(reprojection_error, 1)
                            if '/'.join(scan_folder.split('/')[-2:]) in test_paths:
                                reproj_gt_meter_test.update(reprojection_error, 1)
                            if reprojection_error < 10.0: # PCK @ 10 pixels
                                pck_gt_meter.update(1.0, 1)
                                if '/'.join(scan_folder.split('/')[-2:]) in train_paths:
                                    pck_gt_meter_train.update(1.0, 1)
                                if '/'.join(scan_folder.split('/')[-2:]) in test_paths:
                                    pck_gt_meter_test.update(1.0, 1)
                            else:
                                pck_gt_meter.update(0.0, 1)
                                if '/'.join(scan_folder.split('/')[-2:]) in train_paths:
                                    pck_gt_meter_train.update(0.0, 1)
                                if '/'.join(scan_folder.split('/')[-2:]) in test_paths:
                                    pck_gt_meter_test.update(0.0, 1)
                    else:
                        if confidence_gt == 3:
                            pck_gt_meter.update(0.0, 1)
                            if '/'.join(scan_folder.split('/')[-2:]) in train_paths:
                                pck_gt_meter_train.update(0.0, 1)
                            if '/'.join(scan_folder.split('/')[-2:]) in test_paths:
                                pck_gt_meter_test.update(0.0, 1)

            # Plot results:
            # do_plot = True
            do_plot = False
            if do_plot:
                if file_index % 1 == 0:
                    print(file_index)
                    print(scan_folder)
                    for cam_id, cam in enumerate(cams):
                        if not use_all_frames:
                            image_path = os.path.join(scan_folder, cam, 'images', frame_strs[file_index] + '.png')
                            img = cv.imread(image_path)
                            plt.imshow(img)
                        P = camera_parameters[scene][cam]["P"]
                        for j in range(num_joints):
                            if joints2d[j, cam_id, 2] > args.score_threshold:
                                plt.scatter(joints2d[j, cam_id, 0], joints2d[j, cam_id, 1], c='w')
                            if joints3d[j, 3] > 0.0:
                                x = P @ np.array([joints3d[j, 0], joints3d[j, 1], joints3d[j, 2], 1.0]) # Project to 2D
                                x /= x[2]
                                plt.scatter(x[0], x[1], c='r', s=16)
                        for limb in connectivity:
                            if joints2d[limb[0], cam_id, 2] > args.score_threshold and joints2d[limb[1], cam_id, 2] > args.score_threshold:
                                plt.plot([joints2d[limb[0], cam_id, 0], joints2d[limb[1], cam_id, 0]], [joints2d[limb[0], cam_id, 1], joints2d[limb[1], cam_id, 1]], c='w')
                            if joints3d[limb[0], 3] > 0.0 and joints3d[limb[1], 3] > 0.0:
                                x1 = P @ np.array([joints3d[limb[0], 0], joints3d[limb[0], 1], joints3d[limb[0], 2], 1.0]) # Project to 2D
                                x1 /= x1[2]
                                x2 = P @ np.array([joints3d[limb[1], 0], joints3d[limb[1], 1], joints3d[limb[1], 2], 1.0]) # Project to 2D
                                x2 /= x2[2]
                                plt.plot([x1[0], x2[0]], [x1[1], x2[1]], c='r')
                        plt.axis('equal')
                        # plt.gca().invert_yaxis()
                        plt.show()

                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.set_aspect('equal')
                    ax.view_init(elev=-89., azim=-89.)
                    for j in range(num_joints):
                        if joints3d[j, 3] > 0.0:
                            ax.scatter(joints3d[j, 0], joints3d[j, 1], joints3d[j, 2], c='k') # dev1
                    for limb in connectivity:
                        if joints3d[limb[0], 3] > 0.0 and joints3d[limb[1], 3] > 0.0:
                            ax.plot([joints3d[limb[0], 0], joints3d[limb[1], 0]], [joints3d[limb[0], 1], joints3d[limb[1], 1]], [joints3d[limb[0], 2], joints3d[limb[1], 2]])
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    set_axes_equal(ax)
                    plt.show()

            # Save 3D joints [X,Y,Z,C]
            frame_index = int(keypoint_filename.split('_')[2])
            for cam in cams:
                # Transform into camera coordinate system
                R = camera_parameters[scene][cam]["R"]
                T = camera_parameters[scene][cam]["T"]
                joints3d_cam = joints3d.copy()
                non_zero_indices = joints3d[:, 3] > 0.0
                joints3d_cam[non_zero_indices, :3] = (R @ joints3d_cam[non_zero_indices, :3].T + T).T

                # Rearrange into dictionary for writing:
                output_dict = {}
                # IKEA ANNOTATION STYLE:
                # for joint_id, joint_name in enumerate(joint_names):
                #     output_dict[joint_name] = {"position": list(joints3d_cam[joint_id, :3]), "score": joints3d_cam[joint_id, 3]}
                # OPENPOSE ANNOTATION STYLE:
                output_dict["format"] = args.save_format
                output_dict["pose_keypoints_3d"] = list(joints3d_cam.reshape(-1)) # [X1,Y1,Z1,C1,...XN,YN,ZN,CN]


                # output_path = os.path.join(scan_folder, cam, 'pose3d', f'format_{args.save_format}')
                output_path = os.path.join(scan_folder, cam, 'pose3d') # Directly store in pose3d
                os.makedirs(output_path, exist_ok=True)
                json_file = os.path.join(output_path, f"{frame_index:06}.json")
                with open(json_file, 'w') as f:
                    json.dump(output_dict, f)

    errors_2d_gt = {"reproj": reproj_gt_meter.avg,
                    "reproj_train": reproj_gt_meter_train.avg,
                    "reproj_test": reproj_gt_meter_test.avg,
                    "pck": pck_gt_meter.avg,
                    "pck_train": pck_gt_meter_train.avg,
                    "pck_test": pck_gt_meter_test.avg}
    with open(os.path.join('./', 'pseudoGT_errors_2d.pkl'), 'wb') as f:
        pickle.dump(errors_2d_gt, f)
    print('Averages:')
    print(f"reproj_gt: {reproj_gt_meter.avg}, reproj_gt_train: {reproj_gt_meter_train.avg}, reproj_gt_test: {reproj_gt_meter_test.avg}, pck_gt: {pck_gt_meter.avg}, pck_gt_train: {pck_gt_meter_train.avg}, pck_gt_test: {pck_gt_meter_test.avg}")
    print('Counts:')
    print(f"reproj_gt: {reproj_gt_meter.count}, reproj_gt_train: {reproj_gt_meter_train.count}, reproj_gt_test: {reproj_gt_meter_test.count}, pck_gt: {pck_gt_meter.count}, pck_gt_train: {pck_gt_meter_train.count}, pck_gt_test: {pck_gt_meter_test.count}")


def get_camera_parameters(args):
    calib_path = os.path.join(args.dataset_dir, 'Calibration')
    camera_parameters_file = os.path.join(calib_path, 'camera_parameters.pkl')
    if os.path.exists(camera_parameters_file):
        with open(camera_parameters_file, 'rb') as f:
            camera_parameters = pickle.load(f)
    else:
        camera_parameters = {}
        for scene in scenes:
            scene_path = os.path.join(calib_path, scene)
            camera_parameters[scene] = {}
            for cam in cams:
                cam_path = os.path.join(scene_path, cam)
                with open(os.path.join(cam_path, 'camera_parameters.pkl'), 'rb') as f:
                    camera_parameters[scene][cam] = pickle.load(f)
                    P = camera_parameters[scene][cam]["K"] @ cv.hconcat((camera_parameters[scene][cam]["R"], camera_parameters[scene][cam]["T"]))
                    camera_parameters[scene][cam]["P"] = P
        with open(camera_parameters_file, 'wb') as f:
            pickle.dump(camera_parameters, f)
    return camera_parameters

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def triangulate_nviews(P, ip):
    """
    Triangulate a point visible in n camera views.
    P is a list of camera projection matrices.
    ip is a list of homogenised image points. eg [ [x, y, 1], [x, y, 1] ], OR,
    ip is a 2d array - shape nx3 - [ [x, y, 1], [x, y, 1] ]
    len of ip must be the same as len of P
    """
    if not len(ip) == len(P):
        raise ValueError('Number of points and number of cameras not equal.')
    n = len(P)
    M = np.zeros([3*n, 4+n])
    for i, (x, p) in enumerate(zip(ip, P)):
        M[3*i:3*i+3, :4] = p
        M[3*i:3*i+3, 4+i] = -x
    V = np.linalg.svd(M)[-1]
    X = V[-1, :4]
    return X / X[3]

def get_scan_dirs(input_path):
    """Get all scan directories (scan = an assembly)
    """
    scan_path_list = []
    category_path_list = get_subdirs(input_path)
    for category in category_path_list:
        if os.path.basename(category) != 'Calibration':
            category_scans = get_subdirs(category)
            for category_scan in category_scans:
                scan_path_list.append(category_scan)
    return scan_path_list

def get_subdirs(input_path):
    subdirs = [os.path.join(input_path, dir_i) for dir_i in os.listdir(input_path)
               if os.path.isdir(os.path.join(input_path, dir_i))]
    subdirs.sort()
    return subdirs

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/home/djcam/Documents/HDD/datasets/ikea/ikea_asm/',
                        help='directory of the IKEA assembly dataset')
    parser.add_argument('--save_format', type=str, default='ikea',
                        help='output body format [ikea, body25]')
    parser.add_argument('--input_predictions', type=str, default='keypoint_rcnn',
                        help='input 2D predictions [keypoint_rcnn, openpose]')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                        help='score threshold for 2D joint detections to be used in triangulation')
    parser.add_argument('--reprojection_threshold', type=float, default=30.0,
                        help='reprojection error threshold (joints with greater reproj error will be removed)')
    parser.add_argument('--distance_to_median_threshold', type=float, default=200.0,
                        help='triangulated points are discarded if they are further from the median position than this distance (cm)')
    parser.add_argument('--distance_to_closest_threshold', type=float, default=100.0,
                        help='triangulated points are discarded if they are further from the closest point than this distance (cm)')
    args = parser.parse_args()

    triangulate_joints(args)