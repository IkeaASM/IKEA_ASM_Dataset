# Qualitative result plots
#
# Dylan Campbell <dylan.campbell@anu.edu.au>

import os
import os.path as osp
import argparse
import json
import pickle
import joblib
import numpy as np
import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

from joint_ids import *
from dataset_ids import *

from pdb import set_trace as st

def generate_qualitative_results(args):

    dataset_dir = args.dataset_dir
    scan_dir = osp.join(dataset_dir, args.scan_dir)
    save_dir = args.save_dir
    camera_id = args.camera_id
    frame_id = args.frame_id
    plot_type = args.plot_type

    method_2d = 'keypoint_rcnn_ft'
    method_3d = 'vibe'

    joint_names = get_ikea_joint_names()
    joint_names_dict = get_joint_names_dict(joint_names)
    joint_names_body25 = get_body25_joint_names()
    joint_names_body25_dict = get_joint_names_dict(joint_names_body25)
    connectivity = get_ikea_connectivity()
    num_joints = len(joint_names)
    prediction_dir = osp.join(scan_dir, camera_id, 'predictions')

    cm2mm = 10.0
    m2mm = 1000.0

    # Load image:
    image_path = osp.join(scan_dir, camera_id, 'images', f'{frame_id:06d}.png')
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_size = image.shape
    
    # Load camera parameters:
    label = osp.normpath(scan_dir).split('/')[-1]
    scene = label.split('_')[3]
    with open(osp.join(dataset_dir, 'Calibration', 'camera_parameters.pkl'), 'rb') as f:
        camera_parameters = pickle.load(f)
    K = camera_parameters[scene][camera_id]['K']
    R = camera_parameters[scene][camera_id]['R']
    T = camera_parameters[scene][camera_id]['T']
    P = camera_parameters[scene][camera_id]['P']

    # Load 2D GT:
    with open(osp.join(scan_dir, camera_id, 'pose2d', f'{frame_id:06d}.json')) as json_file:
        pose2d_gt = json.load(json_file)
    keypoints_2d_gt = np.zeros((num_joints, 3))
    for i, joint_name in enumerate(joint_names):
        keypoints_2d_gt[i, :2] = np.array(pose2d_gt[joint_name]["position"])
        keypoints_2d_gt[i, 2] = pose2d_gt[joint_name]["confidence"]

    # Load 2D prediction:
    pose2d_filename = f"scan_video_{frame_id:012}_keypoints.json"
    with open(os.path.join(prediction_dir, 'pose2d', method_2d, pose2d_filename)) as json_file:
        pose2d = json.load(json_file)
    person = pose2d["people"][0]
    keypoints_2d_pred = person["pose_keypoints_2d"]
    keypoints_2d_pred = np.array(keypoints_2d_pred).reshape(-1, 3)
    reorder = [joint_names_dict[joint_name] for joint_name in joint_names]
    keypoints_2d_pred = keypoints_2d_pred[reorder, :]

    # Load 3D pGT:
    with open(osp.join(scan_dir, camera_id, 'pose3d', f'{frame_id:06d}.json')) as json_file:
        pose3d_gt = json.load(json_file)
    keypoints_3d_gt = pose3d_gt["pose_keypoints_3d"]
    keypoints_3d_gt = np.array(keypoints_3d_gt).reshape(-1, 4) * cm2mm
    keypoints_3d_gt[:, :3] = keypoints_3d_gt[:, :3] @ R # Into side view

    # Load 3D prediction:
    if method_3d == 'vibe':
        vibe_filename = os.path.join(prediction_dir, 'pose3d', method_3d, 'vibe_output.pkl')
        if os.path.exists(vibe_filename):
            pose3d = joblib.load(vibe_filename)
            num_frames = max([pose3d[track_id]["frame_ids"].shape[0] for track_id in pose3d.keys()])
        # track_id = min(list(pose3d.keys()))
        # frame_ids = list(pose3d[track_id]["frame_ids"])
        track_id = [track_id for track_id in pose3d.keys() if frame_id in pose3d[track_id]["frame_ids"]][0]
        frame_ids = list(pose3d[track_id]["frame_ids"])
        # st()
        frame_index = frame_ids.index(frame_id)
        keypoints_3d_pred = pose3d[track_id]["joints3d"][frame_index, :, :] # 49x3, first 25 are in body25 format
        reorder = [joint_names_body25_dict[joint_name] for joint_name in joint_names]
        keypoints_3d_pred = keypoints_3d_pred[reorder, :] * m2mm
        keypoints_3d_pred[:, :3] = keypoints_3d_pred[:, :3] @ R # Into side view
        keypoints_3d_pred = np.c_[keypoints_3d_pred, np.ones(keypoints_3d_pred.shape[0])]

    for plot_type in ['image', '2d_gt', '2d_pred', '3d_gt', '3d_pred']:
        if plot_type == 'image':
            scale = 100.0
            fig, ax = setup_fig(size=(image_size[1] / scale, image_size[0] / scale))
            ax.imshow(image, aspect='equal')
            # rect = patches.Rectangle((976,208), 50, 15, facecolor='k') # Mask eyes out
            # ax.add_patch(rect)
            # rect = patches.Rectangle((986,302), 90, 42, facecolor='k')
            # ax.add_patch(rect)
            # rect = patches.Rectangle((860,544), 30, 10, facecolor='k')
            # ax.add_patch(rect)
            filename = osp.join(save_dir, f'{frame_id:06d}_2d.png')
            plt.savefig(filename, dpi=scale)
        elif plot_type == '2d_gt':
            scale = 100.0
            fig, ax = setup_fig(size=(image_size[1] / scale, image_size[0] / scale))
            ax.imshow(image, aspect='equal')
            # rect = patches.Rectangle((986,302), 90, 42, facecolor='k')
            # ax.add_patch(rect)
            plot_skeleton(image, keypoints_2d_gt, connectivity)
            filename = osp.join(save_dir, f'{frame_id:06d}_2d_gt.png')
            plt.savefig(filename, dpi=scale)
        elif plot_type == '2d_pred':
            scale = 100.0
            fig, ax = setup_fig(size=(image_size[1] / scale, image_size[0] / scale))
            ax.imshow(image, aspect='equal')
            # rect = patches.Rectangle((986,302), 90, 42, facecolor='k')
            # ax.add_patch(rect)
            plot_skeleton(image, keypoints_2d_pred, connectivity)
            filename = osp.join(save_dir, f'{frame_id:06d}_2d_pred.png')
            plt.savefig(filename, dpi=scale)
        elif plot_type == '3d_gt':
            filename = osp.join(save_dir, f'{frame_id:06d}_3d_gt.png')
            plot_skeleton_3d(keypoints_3d_gt, connectivity)
            plt.savefig(filename, bbox_inches='tight')
        elif plot_type == '3d_pred':
            filename = osp.join(save_dir, f'{frame_id:06d}_3d_pred.png')
            plot_skeleton_3d(keypoints_3d_pred, connectivity)
            plt.savefig(filename, bbox_inches='tight')

    # K = camera_parameters[scene][cam]['K']
    # image_path = os.path.join(gt_dir, 'images', f'{frame_id:06d}.png')
    # img = cv.imread(image_path)
    # plt.imshow(img)
    # for joint_id_gt, joint_name in enumerate(ikea_joint_names):
    #     if confidence_gt[joint_id_gt] > 0.0:
    #         x = K @ np.array([joints3d_gt[joint_id_gt, 0], joints3d_gt[joint_id_gt, 1], joints3d_gt[joint_id_gt, 2]]) # Project to 2D
    #         x /= x[2]
    #         plt.scatter(x[0], x[1], c='r', s=16)
    # height = img.shape[0]
    # width = img.shape[1]
    # fx = width / 2.
    # fy = height / 2.
    # cx = width / 2.
    # cy = height / 2.
    # j3dx = fx * sx * (joints3d[:, 0] / m2mm + tx) + cx
    # j3dy = fy * sy * (joints3d[:, 1] / m2mm + ty) + cy
    # plt.scatter(j3dx, j3dy, c='w', s=10)
    # plt.show()

def plot_skeleton(image, keypoints, connectivity):
    positions = keypoints[:, :2]
    visibility = keypoints[:, 2]
    for joint_index, position in enumerate(positions):
        if visibility[joint_index] > 0.0:
            plt.scatter(position[0].item(), position[1].item(), c='w', s=64)
    for limb in connectivity:
        if visibility[limb[0]] > 0.0 and visibility[limb[1]] > 0.0:
            plt.plot([positions[limb[0], 0].item(), positions[limb[1], 0].item()], [positions[limb[0], 1].item(), positions[limb[1], 1].item()], c='w', lw=3)

def plot_skeleton_3d(keypoints, connectivity):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    # ax.view_init(elev=-89., azim=-89.)
    # ax.view_init(elev=160., azim=0.)
    ax.view_init(elev=150., azim=-5.)
    for i in range(keypoints.shape[0]):
        if keypoints[i, 3] > 0.0:
            ax.scatter(keypoints[i, 0], keypoints[i, 1], keypoints[i, 2], c='k')
    for limb in connectivity:
        if keypoints[limb[0], 3] > 0.0 and keypoints[limb[1], 3] > 0.0:
            ax.plot([keypoints[limb[0], 0], keypoints[limb[1], 0]], [keypoints[limb[0], 1], keypoints[limb[1], 1]], [keypoints[limb[0], 2], keypoints[limb[1], 2]], c='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.axis('off')
    set_axes_equal(ax)
    # plt.show()

def make_image(data, outputname, size=(1, 1), dpi=80):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    # plt.set_cmap('hot')
    ax.imshow(data, aspect='equal')
    plt.savefig(outputname, dpi=dpi)

def setup_fig(size=(1, 1)):
    fig = plt.figure()
    fig.set_size_inches(size)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    return fig, ax

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/home/djcam/Documents/HDD/datasets/ikea/ikea_asm/',
                        help='directory of the IKEA assembly dataset')
    parser.add_argument('--save_dir', type=str, default='./qualitative_results/',
                        help='directory to save images')
    parser.add_argument('--camera_id', type=str, default='dev3',
                        help='camera device ID for dataset (GT annotations for dev3 only)')
    parser.add_argument('--scan_dir', type=str, default='Kallax_Shelf_Drawer/0001_black_table_02_01_2019_08_16_14_00/',
                        help='scan directory to plot')
    parser.add_argument('--frame_id', type=int, default=0,
                        help='frame ID to plot')
    parser.add_argument('--plot_type', type=str, default='2d_gt',
                        help='plot type')
    args = parser.parse_args()

    generate_qualitative_results(args)

    