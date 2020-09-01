# Evaluate human pose results
#
# For every capture directory:
# 1. Read pose2d and pose3d JSON files
# 2. Read output JSON files
# 3. Read test/train split files
# 4. Compute MPJPE and PCK (2D) and MPJPE and PA-MPJPE (3D)
#
# Dylan Campbell <dylan.campbell@anu.edu.au>

import os
import sys
import numpy as np
import json
import joblib
import pickle
import argparse
import time
import torch
import logging
import subprocess
import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.seterr(all='raise')

from dataset_ids import *
from joint_ids import *

from pdb import set_trace as st

def eval2d(args, eval_type=None):
    """Evaluate 2D human pose predictions
    """

    ikea_joint_names = get_ikea_joint_names()
    ikea_joint_names_dict = {name: i for i, name in enumerate(ikea_joint_names)}
    if "openpose" in eval_type:
        if "staf" in eval_type:
            joint_names = get_body21_joint_names()
            joint_names_dict = {name: i for i, name in enumerate(joint_names)}
        else:
            joint_names = get_body25_joint_names()
            joint_names_dict = {name: i for i, name in enumerate(joint_names)}
    elif "rcnn" in eval_type:
        joint_names = get_ikea_joint_names()
        joint_names_dict = {name: i for i, name in enumerate(joint_names)}

    results = []
    gt_dirs = get_gt_dirs(args.dataset_dir, args.camera_id) # Only those folders with pose2d annotations
    for i, gt_dir in enumerate(gt_dirs):
        print(f"\nProcessing {i} of {len(gt_dirs)}: {' '.join(gt_dir.split('/')[-3:-1])}")
        prediction_dir = os.path.join(gt_dir, 'predictions', 'pose2d', f"{eval_type}")
        pose2d_gt_filenames = os.listdir(os.path.join(gt_dir, 'pose2d'))
        path = '/'.join(gt_dir.split('/')[-3:-1]) # Save this as well
        video_results = {'path': path}
        errors = []
        scores = []
        confidences_gt = []
        for pose2d_gt_filename in pose2d_gt_filenames:
            frame_id = int(os.path.splitext(pose2d_gt_filename)[0])
            with open(os.path.join(gt_dir, 'pose2d', pose2d_gt_filename)) as json_file:
                pose2d_gt = json.load(json_file)
            pose2d_filename = f"scan_video_{frame_id:012}_keypoints.json"
            with open(os.path.join(prediction_dir, pose2d_filename)) as json_file:
                pose2d = json.load(json_file)

            # Iterate over people (choose best):
            people_data = []
            mean_position_error = []
            for person in pose2d["people"]:
                keypoints = person["pose_keypoints_2d"] # x1, y1, c1, x2, ...

                position_error = []
                score = []
                confidence_gt = []
                for joint_name in ikea_joint_names:
                    position_gt = np.array(pose2d_gt[joint_name]["position"])
                    confidence_gt.append(pose2d_gt[joint_name]["confidence"])
                    joint_id = joint_names_dict[joint_name]
                    position = np.array(keypoints[3*joint_id:(3*joint_id + 2)])
                    score.append(keypoints[3*joint_id + 2])
                    position_error.append(np.linalg.norm(position - position_gt))

                if "keypoint_visibility" in person: # MaskRCNN has a separate visibility score
                    keypoint_visibility = person["keypoint_visibility"]
                    mean_position_error.append(np.mean([e for e, v in zip(position_error, keypoint_visibility) if v > 0]))
                    score = [s if v > 0 else -np.Inf for s, v in zip(score, keypoint_visibility)] # Set score to -Inf if invisible
                else: # OpenPose uses score = 0 as not visible
                    mean_position_error.append(np.mean([e for e, s in zip(position_error, score) if s > 0]))
                    score = [s if s > 0.0 else -np.Inf for s in score] # Set score to -Inf if invisible
                people_data.append([position_error, score, confidence_gt])

            if not mean_position_error: # If empty, there were no detections
                errors.append([np.Inf]*len(ikea_joint_names))
                scores.append([-np.Inf]*len(ikea_joint_names))
                confidence_gt = []
                for joint_name in ikea_joint_names:
                    confidence_gt.append(pose2d_gt[joint_name]["confidence"])
                confidences_gt.append(confidence_gt)
            else:
                person_id = np.argmin(mean_position_error) # Choose person with least mean position error
                errors.append(people_data[person_id][0])
                scores.append(people_data[person_id][1])
                confidences_gt.append(people_data[person_id][2])
        
        video_results["errors"] = errors
        video_results["scores"] = scores
        video_results["confidences_gt"] = confidences_gt
        results.append(video_results)

    with open(f"./results/results_{eval_type}.pkl", 'wb') as f:
        pickle.dump(results, f)

def parse_results_2d(args, results_file, score_threshold=0.0):
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    with open(os.path.join(args.dataset_dir, 'test_cross_env.txt'), 'r') as f:
        test_paths = f.read().splitlines()
    with open(os.path.join(args.dataset_dir, 'train_cross_env.txt'), 'r') as f:
        train_paths = f.read().splitlines()
    ikea_joint_names = get_ikea_joint_names()
    num_joints = len(ikea_joint_names)

    test_results = np.zeros((0, num_joints, 3))
    train_results = np.zeros((0, num_joints, 3))
    for video_results in results:
        path = video_results["path"]
        furniture_type, person_id, colour, surface, calibration_id, room_id, datetime = path_to_types(path)
        errors = np.array(video_results["errors"])
        scores = np.array(video_results["scores"])
        confidences_gt = np.array(video_results["confidences_gt"])
        category = args.category
        if not category or (
            category == 'female' and person_id in get_female_ids()) or (
            category == 'male' and person_id in get_male_ids()) or (
            category == 'floor' and surface == category) or (
            category == 'table' and surface == category):
            if path in test_paths:
                test_results = np.vstack((test_results, np.stack((errors, scores, confidences_gt), axis=-1)))
            elif path in train_paths:
                train_results = np.vstack((train_results, np.stack((errors, scores, confidences_gt), axis=-1)))

    test_mpjpe_meters, test_pck_meters, test_errors, test_detection_meter = compute_error_measures_2d(args, test_results, score_threshold)
    train_mpjpe_meters, train_pck_meters, train_errors, train_detection_meter = compute_error_measures_2d(args, train_results, score_threshold)
    test_mpjpe_group = compute_group_errors(test_mpjpe_meters)
    test_pck_group = compute_group_errors(test_pck_meters)
    train_mpjpe_group = compute_group_errors(train_mpjpe_meters)
    train_pck_group = compute_group_errors(train_pck_meters)
    test_auc = compute_auc(test_errors, max_error=100.)
    train_auc = compute_auc(train_errors, max_error=100.)

    print(f"MPJPE (train) & PCK @ {args.pck_threshold_2d} (train) & AUC (train) & MPJPE (test) & PCK @ {args.pck_threshold_2d} (test) & AUC (test)")
    print(f"{train_mpjpe_meters[-1].avg:.1f} & {100*train_pck_meters[-1].avg:.1f} & {train_auc:.1f} & {test_mpjpe_meters[-1].avg:.1f} & {100*test_pck_meters[-1].avg:.1f} & {test_auc:.1f}")

    ikea_joint_groups_names = get_ikea_joint_groups_names()
    print(f"PCK @ {args.pck_threshold_2d} (test): {', '.join(ikea_joint_groups_names)}")
    print(f"{' & '.join([f'{100*test_pck:.1f}' for test_pck in test_pck_group])}")

    print(f"Detection rate: {100*train_detection_meter.avg:.2f} (train), {100*test_detection_meter.avg:.2f} (test)")
    print(f"Missed detections: {train_detection_meter.count - train_detection_meter.avg * train_detection_meter.count:.0f} / {train_detection_meter.count:.0f} (train), {test_detection_meter.count - test_detection_meter.avg * test_detection_meter.count:.0f} / {test_detection_meter.count:.0f} (test)")

def compute_error_measures_2d(args, results, score_threshold=0.0):
    num_frames = results.shape[0]
    num_joints = len(get_ikea_joint_names())
    mpjpe_meters = [AverageMeter('MPJPE', ':.1f') for _ in range(num_joints + 1)] # joints + all
    pck_meters = [AverageMeter('PCK', ':.1f') for _ in range(num_joints + 1)] # joints + all
    detection_meter = AverageMeter('PCK', ':.1f')
    errors = [[] for _ in range(num_joints + 1)]
    for i in range(num_frames):
        if (results[i, :, 1] < score_threshold).all(): # If no detected joints in entire frame
            detection_meter.update(0.0, 1)
        else:
            detection_meter.update(1.0, 1)
        for j in range(num_joints):
            error = results[i, j, 0]
            score = results[i, j, 1]
            confidence_gt = results[i, j, 2]
            if score >= score_threshold and confidence_gt == 3: # Confident annotations, detected joints only
                mpjpe_meters[j].update(error, 1)
                mpjpe_meters[-1].update(error, 1)
            if confidence_gt == 3: # Confident annotations
                if score < score_threshold: # Convert missed detections (-Inf) into infinite error
                    error = np.Inf
                if error <= args.pck_threshold_2d: # In pixels
                    pck_meters[j].update(1, 1)
                    pck_meters[-1].update(1, 1)
                else:
                    pck_meters[j].update(0, 1)
                    pck_meters[-1].update(0, 1)
                # Store errors
                errors[j].append(error)
                errors[-1].append(error)

    return mpjpe_meters, pck_meters, errors, detection_meter

def compute_auc(errors, max_error=100.):
    """ Compute area under the PCK curve, to a maximum error of 100 pixels (5% of image width)
    """
    errors_all = errors[-1]
    num_samples = len(errors_all)
    errors_all = np.array(sorted(errors_all))
    # Remove missed detections:
    errors_all = errors_all[np.isfinite(errors_all)]
    # AUC for thresholds < 100 only:
    # Add sentinel values, errors are now [0, 100)
    err = np.concatenate(([0.], errors_all[errors_all < max_error], [max_error]))
    prec = np.cumsum(np.concatenate(([0.], np.ones(err.size - 2), [0.]))) / num_samples
    # Look for points were x axis (error) changes:
    i = np.where(err[1:] != err[:-1])[0]
    # and sum (Delta error) * frac
    auc = np.sum((err[i + 1] - err[i]) * prec[i + 1])
    return auc

def compute_group_errors(meters):
    group_errors = []
    for i, joint_ids in enumerate(get_ikea_joint_groups()):
        sum = 0.0
        count = 0
        for joint_id in joint_ids:
            sum += meters[joint_id].sum
            count += meters[joint_id].count
        group_errors.append(sum / count)
    return group_errors

def eval3d(args, eval_type=None):
    """ Compare 3D predictions with 3D pseudo-ground-truth.
    Only evaluates at the frame ids of the 2D GT annotated frames,
    since these pseudoGT annotations have been validated.
    """
    cm2mm = 10.0
    m2mm = 1000.0

    ikea_joint_names = get_ikea_joint_names()
    ikea_joint_names_dict = {name: i for i, name in enumerate(ikea_joint_names)}
    connectivity_ikea = get_ikea_connectivity()
    if "vibe" in eval_type:
        joint_names = get_body25_joint_names() # Vibe uses SPIN format, for which the first 25 joints are in body25 format
    elif "hmmr" in eval_type:
        joint_names = get_hmmr_joint_names()
    elif "vp3d" in eval_type:
        joint_names = get_h36m_joint_names()
    joint_names_dict = {name: i for i, name in enumerate(joint_names)}

    num_joints = len(ikea_joint_names)

    # For plotting:
    with open(os.path.join(args.dataset_dir, 'Calibration/camera_parameters.pkl'), 'rb') as f:
        camera_parameters = pickle.load(f)

    results = []
    gt_dirs = get_gt_dirs(args.dataset_dir, args.camera_id) #  # Only those folders with pose2d annotations, eg <root>/Lack_TV_Bench/0007_white_floor_08_04_2019_08_28_10_47/dev3
    for i, gt_dir in enumerate(gt_dirs):
        print(f"\nProcessing {i} of {len(gt_dirs)}: {' '.join(gt_dir.split('/')[-3:-1])}")
        prediction_dir = os.path.join(gt_dir, 'predictions', 'pose3d', f"{eval_type}")
        pose3d_pgt_dir = os.path.join(gt_dir, 'pose3d') # May contain more pGT annotations than the 2D GT
        pose2d_gt_dir = os.path.join(gt_dir, 'pose2d')
        json_filenames = os.listdir(pose2d_gt_dir) # 2D GT annotations eg 000000.json
        path = '/'.join(gt_dir.split('/')[-3:-1]) # Save this as well
        video_results = {'path': path}

        cam = gt_dir.split('/')[-1]
        label = gt_dir.split('/')[-2]
        scene = label.split('_')[3]

        if "vibe" in eval_type:
            vibe_filename = os.path.join(prediction_dir, 'vibe_output.pkl')
            if os.path.exists(vibe_filename):
                pose3d = joblib.load(vibe_filename)
                num_frames = max([pose3d[track_id]["frame_ids"].shape[0] for track_id in pose3d.keys()])
            else:
                print(f'MISSING VIBE FILE: {vibe_filename}')
        elif "hmmr" in eval_type:
            pkl_mask = os.path.join(prediction_dir, '*.pkl')
            hmmr_filenames = sorted(glob(pkl_mask))
            num_frames = 0
            pose3d = {0: {"cam": [], "joints3d": []}}
            for hmmr_filename in hmmr_filenames:
                if os.path.exists(hmmr_filename):
                    pose3d_partial = joblib.load(hmmr_filename)
                    pose3d[0]["cam"].extend(pose3d_partial['cams']) # N x (3,)
                    pose3d[0]["joints3d"].extend(pose3d_partial['joints']) # N x (25,3)
                    num_frames += len(pose3d_partial['joints'])
                else:
                    print(f'MISSING HMMR FILE: {hmmr_filename}')
            pose3d[0]["frame_ids"] = list(range(num_frames))
            pose3d[0]["joints3d"] = np.array(pose3d[0]["joints3d"]) # Nx25x3
            pose3d[0]["cam"] = np.array(pose3d[0]["cam"]) # Nx3
        elif "vp3d" in eval_type:
            vp3d_filename = os.path.join(prediction_dir, 'vp3d_output.pkl')
            pose3d = {0: {}}
            if os.path.exists(vp3d_filename):
                pose3d_person = joblib.load(vp3d_filename) # Nx17x3 - single person
                num_frames = pose3d_person.shape[0]
                pose3d[0]["frame_ids"] = list(range(num_frames))
                pose3d[0]["orig_cam"] = np.array([1.0, 1.0, 0.0, 0.0]) # Does not return camera
                pose3d[0]["joints3d"] = pose3d_person
            else:
                print(f'MISSING VideoPose3D FILE: {vp3d_filename}')

        errors = []
        errors_pa = []
        errors_reprojection = []
        confidences_gt = []
        confidences_2d_gt = []
        joints3d_all = []
        joints3d_gt_all = []
        joints3d_aligned_all = []
        joints3d_gt_aligned_all = []
        for json_filename in json_filenames: # Only evaluate for frames with GT annotations
            frame_id = int(os.path.splitext(json_filename)[0])
            # print(frame_id)
            with open(os.path.join(pose3d_pgt_dir, json_filename)) as json_file:
                pose3d_gt = np.array(json.load(json_file)["pose_keypoints_3d"]).reshape(-1, 4) # 17x4 [X,Y,Z,C]
            
            # Get 2D GT joints and confidences
            joints2d_gt = np.zeros((num_joints, 2))
            confidence_2d_gt = []
            with open(os.path.join(pose2d_gt_dir, json_filename)) as json_file:
                pose2d_gt = json.load(json_file)
            for joint_id_gt, joint_name in enumerate(ikea_joint_names):
                joints2d_gt[joint_id_gt, :] = np.array(pose2d_gt[joint_name]["position"])
                confidence_2d_gt.append(pose2d_gt[joint_name]["confidence"])
            confidence_2d_gt = np.array(confidence_2d_gt)

            # Iterate over people (choose best):
            people_data = []
            mean_position_error = []
            for track_id in pose3d.keys():
                frame_ids = list(pose3d[track_id]["frame_ids"])
                if frame_id in frame_ids:
                    frame_index = frame_ids.index(frame_id)
                    keypoints = pose3d[track_id]["joints3d"][frame_index, :, :]
                else:
                    continue # person not detected in this frame
                if eval_type == 'vibe':
                    sx, sy, tx, ty = pose3d[track_id]["orig_cam"][frame_index, :]
                elif eval_type == 'hmmr':
                    s, tx, ty = pose3d[track_id]["cam"][frame_index, :]
                elif eval_type == "vp3d":
                    sx, sy, tx, ty = pose3d[track_id]["orig_cam"]

                # Convert joints into common coordinates:
                joints3d = np.zeros((num_joints, 3))
                joints3d_gt = np.zeros((num_joints, 3))
                confidence_gt = np.zeros(num_joints)
                for joint_id_gt, joint_name in enumerate(ikea_joint_names):
                    joints3d_gt[joint_id_gt, :] = pose3d_gt[joint_id_gt, :3] * cm2mm
                    confidence_gt[joint_id_gt] = pose3d_gt[joint_id_gt, 3]
                    if eval_type == 'vibe' or eval_type == 'hmmr':
                        joint_id = joint_names_dict[joint_name]
                        joints3d[joint_id_gt, :] = np.array(keypoints[joint_id, :]) * m2mm
                    elif eval_type == "vp3d": # H36M format does not have r/l ear, e/r eye
                        if joint_name == "right ear" or joint_name == "left ear" or joint_name == "right eye" or joint_name == "left eye":
                            pass
                        else:
                            joint_id = joint_names_dict[joint_name]
                            joints3d[joint_id_gt, :] = np.array(keypoints[joint_id, :]) * m2mm
                if eval_type == "vp3d":
                    # Fill in missing joints with estimates:
                    lrshoulder_direction = joints3d[ikea_joint_names_dict["left shoulder"], :] - joints3d[ikea_joint_names_dict["right shoulder"], :]
                    lrshoulder_direction_norm = np.linalg.norm(lrshoulder_direction)
                    if lrshoulder_direction_norm > 1e-6:
                        lrshoulder_direction /= lrshoulder_direction_norm
                    headnose_direction = (np.array(keypoints[joint_names_dict['head'], :]) - np.array(keypoints[joint_names_dict['nose'], :])) * m2mm
                    headnose_direction_norm = np.linalg.norm(headnose_direction)
                    if headnose_direction_norm > 1e-6:
                        headnose_direction /= headnose_direction_norm
                    joints3d[ikea_joint_names_dict["right ear"], :] = np.array(keypoints[joint_names_dict['head'], :]) * m2mm - lrshoulder_direction * 70.
                    joints3d[ikea_joint_names_dict["left ear"], :] = np.array(keypoints[joint_names_dict['head'], :]) * m2mm + lrshoulder_direction * 70.
                    joints3d[ikea_joint_names_dict["right eye"], :] = np.array(keypoints[joint_names_dict['nose'], :]) * m2mm + headnose_direction * 30. - lrshoulder_direction * 40.
                    joints3d[ikea_joint_names_dict["left eye"], :] = np.array(keypoints[joint_names_dict['nose'], :]) * m2mm + headnose_direction * 30. + lrshoulder_direction * 40.
                # print(joints3d_gt)

                # Align by centroid of predicted joints in common:
                valid_joint_indices = (confidence_gt > 0.0) & (confidence_2d_gt == 3)
                if valid_joint_indices.any():
                    centroid = np.mean(joints3d[valid_joint_indices, :], axis=0)
                    centroid_gt = np.mean(joints3d_gt[valid_joint_indices, :], axis=0)
                else:
                    centroid = np.zeros(3)
                    centroid_gt = np.zeros(3)
                joints3d_aligned = joints3d - centroid
                joints3d_gt_aligned = joints3d_gt - centroid_gt

                # Procrustes alignment:
                joints3d_pa = joints3d_aligned.copy()
                if valid_joint_indices.sum() > 1:
                    joints3d_pa[valid_joint_indices, :] = compute_similarity_transform(joints3d_aligned[valid_joint_indices, :], joints3d_gt_aligned[valid_joint_indices, :])

                # Compute errors:
                position_errors = np.sqrt(((joints3d_aligned - joints3d_gt_aligned) ** 2).sum(axis=-1))
                position_pa_errors = np.sqrt(((joints3d_pa - joints3d_gt_aligned) ** 2).sum(axis=-1))
                position_errors_valid = position_errors[valid_joint_indices]
                if position_errors_valid.size > 0:
                    mean_position_error.append(np.mean(position_errors_valid))
                else:
                    mean_position_error.append(np.Inf)
                
                # Compute reprojection errors:
                K = camera_parameters[scene][cam]['K']
                joints3d_proj_gt = joints3d_gt
                nonzero_joint_indices = joints3d_proj_gt[:, 2] > 0.0
                joints3d_proj_gt[nonzero_joint_indices, :] = (K @ joints3d_gt[nonzero_joint_indices, :].T).T
                joints3d_proj_gt[nonzero_joint_indices, :] /= joints3d_proj_gt[nonzero_joint_indices, 2:]
                joints3d_proj_gt = joints3d_proj_gt[:, :2]
                image_path = os.path.join(gt_dir, 'images', f'{frame_id:06d}.png')
                img = cv.imread(image_path)
                fx = cx = img.shape[1] / 2. # width/2
                fy = cy = img.shape[0] / 2. # height/2
                if eval_type == 'hmmr': # Need to estimate bbox centre -- very approximate
                    centre_x = (np.min(joints2d_gt[:, 0]) + np.max(joints2d_gt[:, 0])) / 2.0
                    centre_y = (np.min(joints2d_gt[:, 1]) + np.max(joints2d_gt[:, 1])) / 2.0
                    sx, sy, tx, ty = crop_to_orig(s, tx, ty, centre_x, centre_y, bbox_side=672., image_width=img.shape[1], image_height=img.shape[0])
                joints3d_proj = joints3d[:, 0:2].copy()
                joints3d_proj[:, 0] = fx * sx * (joints3d[:, 0] / m2mm + tx) + cx
                joints3d_proj[:, 1] = fy * sy * (joints3d[:, 1] / m2mm + ty) + cy

                if eval_type == "hmmr": # align to GT nose
                    cx = joints2d_gt[0, 0] - joints3d_proj[0, 0] # Align to GT nose
                    cy = joints2d_gt[0, 1] - joints3d_proj[0, 1]
                    joints3d_proj[:, 0] += cx
                    joints3d_proj[:, 1] += cy
                if eval_type == "vp3d":
                    sx = sy = img.shape[1] / 5.0
                    joints3d_proj = joints3d[:, 0:2].copy()
                    joints3d_proj[:, 0] = sx * (joints3d[:, 0] / m2mm)
                    joints3d_proj[:, 1] = sy * (joints3d[:, 1] / m2mm)
                    cx = joints2d_gt[0, 0] - joints3d_proj[0, 0] # Align to GT nose
                    cy = joints2d_gt[0, 1] - joints3d_proj[0, 1]
                    joints3d_proj[:, 0] += cx
                    joints3d_proj[:, 1] += cy

                # reprojection_errors = np.sqrt(((joints3d_proj - joints3d_proj_gt) ** 2).sum(axis=-1)) # wrt 3D GT proj
                reprojection_errors = np.sqrt(((joints3d_proj - joints2d_gt) ** 2).sum(axis=-1)) # wrt 2D GT

                people_data.append([position_errors, position_pa_errors, reprojection_errors, confidence_gt, confidence_2d_gt, joints3d, joints3d_gt, joints3d_aligned, joints3d_gt_aligned])

                # Plot
                # do_plot = True
                do_plot = False
                if do_plot:
                    plt.imshow(img)
                    # plt.scatter(joints3d_proj_gt[confidence_gt > 0.0, 0], joints3d_proj_gt[confidence_gt > 0.0, 1], c='r', s=16)
                    plt.scatter(joints2d_gt[:, 0], joints2d_gt[:, 1], c='r', s=16)
                    plt.scatter(joints3d_proj[:, 0], joints3d_proj[:, 1], c='w', s=10)
                    plt.show()

                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.set_aspect('equal')
                    ax.view_init(elev=-89., azim=-89.)
                    for joint_id_gt, joint_name in enumerate(ikea_joint_names):
                        if confidence_gt[joint_id_gt] > 0.0:
                            ax.scatter(joints3d_gt_aligned[joint_id_gt, 0], joints3d_gt_aligned[joint_id_gt, 1], joints3d_gt_aligned[joint_id_gt, 2], c='k')
                            ax.scatter(joints3d_aligned[joint_id_gt, 0], joints3d_aligned[joint_id_gt, 1], joints3d_aligned[joint_id_gt, 2], c='r')
                    for limb in connectivity_ikea:
                        if confidence_gt[limb[0]] > 0.0 and confidence_gt[limb[1]] > 0.0:
                            ax.plot([joints3d_gt_aligned[limb[0], 0], joints3d_gt_aligned[limb[1], 0]], [joints3d_gt_aligned[limb[0], 1], joints3d_gt_aligned[limb[1], 1]], [joints3d_gt_aligned[limb[0], 2], joints3d_gt_aligned[limb[1], 2]], c='k')
                            ax.plot([joints3d_aligned[limb[0], 0], joints3d_aligned[limb[1], 0]], [joints3d_aligned[limb[0], 1], joints3d_aligned[limb[1], 1]], [joints3d_aligned[limb[0], 2], joints3d_aligned[limb[1], 2]], c='r')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    set_axes_equal(ax)
                    plt.show()

                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.set_aspect('equal')
                    ax.view_init(elev=-89., azim=-89.)
                    for joint_id_gt, joint_name in enumerate(ikea_joint_names):
                        if confidence_gt[joint_id_gt] > 0.0:
                            ax.scatter(joints3d_gt_aligned[joint_id_gt, 0], joints3d_gt_aligned[joint_id_gt, 1], joints3d_gt_aligned[joint_id_gt, 2], c='k')
                            ax.scatter(joints3d_pa[joint_id_gt, 0], joints3d_pa[joint_id_gt, 1], joints3d_pa[joint_id_gt, 2], c='r')
                    for limb in connectivity_ikea:
                        if confidence_gt[limb[0]] > 0.0 and confidence_gt[limb[1]] > 0.0:
                            ax.plot([joints3d_gt_aligned[limb[0], 0], joints3d_gt_aligned[limb[1], 0]], [joints3d_gt_aligned[limb[0], 1], joints3d_gt_aligned[limb[1], 1]], [joints3d_gt_aligned[limb[0], 2], joints3d_gt_aligned[limb[1], 2]], c='k')
                            ax.plot([joints3d_pa[limb[0], 0], joints3d_pa[limb[1], 0]], [joints3d_pa[limb[0], 1], joints3d_pa[limb[1], 1]], [joints3d_pa[limb[0], 2], joints3d_pa[limb[1], 2]], c='r')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    set_axes_equal(ax)
                    plt.show()

            if not mean_position_error: # If empty, there were no detections
                errors.append([np.Inf]*len(ikea_joint_names))
                errors_pa.append([np.Inf]*len(ikea_joint_names))
                errors_reprojection.append([np.Inf]*len(ikea_joint_names))
                confidence_gt = []
                for joint_id_gt, joint_name in enumerate(ikea_joint_names):
                    confidence_gt.append(pose3d_gt[joint_id_gt, 3])
                confidences_gt.append(confidence_gt)
                confidences_2d_gt.append(confidence_2d_gt)
                joints3d_all.append(None)
                joints3d_gt_all.append(None)
                joints3d_aligned_all.append(None)
                joints3d_gt_aligned_all.append(None)
            else:
                person_id = np.argmin(mean_position_error) # Choose person with least mean position error
                errors.append(people_data[person_id][0])
                errors_pa.append(people_data[person_id][1])
                errors_reprojection.append(people_data[person_id][2])
                confidences_gt.append(people_data[person_id][3])
                confidences_2d_gt.append(people_data[person_id][4])
                joints3d_all.append(people_data[person_id][5])
                joints3d_gt_all.append(people_data[person_id][6])
                joints3d_aligned_all.append(people_data[person_id][7])
                joints3d_gt_aligned_all.append(people_data[person_id][8])

        
        video_results["errors"] = errors
        video_results["errors_pa"] = errors_pa
        video_results["errors_reprojection"] = errors_reprojection
        video_results["confidences_gt"] = confidences_gt
        video_results["confidences_2d_gt"] = confidences_2d_gt
        video_results["joints3d_all"] = joints3d_all
        video_results["joints3d_gt_all"] = joints3d_gt_all
        video_results["joints3d_aligned_all"] = joints3d_aligned_all
        video_results["joints3d_gt_aligned_all"] = joints3d_gt_aligned_all
        results.append(video_results)

    with open(f"./results/results_{eval_type}.pkl", 'wb') as f:
        pickle.dump(results, f)

def parse_results_3d(args, results_file):
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    with open(os.path.join(args.dataset_dir, 'test_cross_env.txt'), 'r') as f:
        test_paths = f.read().splitlines()
    with open(os.path.join(args.dataset_dir, 'train_cross_env.txt'), 'r') as f:
        train_paths = f.read().splitlines()
    ikea_joint_names = get_ikea_joint_names()
    num_joints = len(ikea_joint_names)

    test_results = np.zeros((0, num_joints, 5))
    train_results = np.zeros((0, num_joints, 5))
    for video_results in results:
        path = video_results["path"]
        furniture_type, person_id, colour, surface, calibration_id, room_id, datetime = path_to_types(path)
        errors = np.array(video_results["errors"])
        errors_pa = np.array(video_results["errors_pa"])
        errors_reprojection = np.array(video_results["errors_reprojection"])
        confidences_gt = np.array(video_results["confidences_gt"])
        confidences_2d_gt = np.array(video_results["confidences_2d_gt"])
        category = args.category
        if not category or (
            category == 'female' and person_id in get_female_ids()) or (
            category == 'male' and person_id in get_male_ids()) or (
            category == 'floor' and surface == category) or (
            category == 'table' and surface == category):
            if path in test_paths:
                test_results = np.vstack((test_results, np.stack((errors, errors_pa, errors_reprojection, confidences_gt, confidences_2d_gt), axis=-1)))
            elif path in train_paths:
                train_results = np.vstack((train_results, np.stack((errors, errors_pa, errors_reprojection, confidences_gt, confidences_2d_gt), axis=-1)))

    test_mpjpe_meters, test_pck_meters, test_errors, test_mpjpe_pa_meters, test_pck_pa_meters, test_errors_pa, test_mpjpe_reproj_meters, test_pck_reproj_meters, test_errors_reproj = compute_error_measures_3d(args, test_results)
    train_mpjpe_meters, train_pck_meters, train_errors, train_mpjpe_pa_meters, train_pck_pa_meters, train_errors_pa, train_mpjpe_reproj_meters, train_pck_reproj_meters, train_errors_reproj = compute_error_measures_3d(args, train_results)
    test_mpjpe_group = compute_group_errors(test_mpjpe_meters)
    test_pck_group = compute_group_errors(test_pck_meters)
    train_mpjpe_group = compute_group_errors(train_mpjpe_meters)
    train_pck_group = compute_group_errors(train_pck_meters)
    test_mpjpe_pa_group = compute_group_errors(test_mpjpe_pa_meters)
    test_pck_pa_group = compute_group_errors(test_pck_pa_meters)
    train_mpjpe_pa_group = compute_group_errors(train_mpjpe_pa_meters)
    train_pck_pa_group = compute_group_errors(train_pck_pa_meters)

    print(f"MPJPE (train) & MPJPE-PA (train) & PCK @ {args.pck_threshold_3d} (train) & PCK-PA @ {args.pck_threshold_3d} (train) & MPJPE (test) & MPJPE-PA (test) & PCK @ {args.pck_threshold_3d} (test) & PCK-PA @ {args.pck_threshold_3d} (test)")
    print(f"{train_mpjpe_meters[-1].avg:.1f} & {train_mpjpe_pa_meters[-1].avg:.1f} & {100*train_pck_meters[-1].avg:.1f} & {100*train_pck_pa_meters[-1].avg:.1f} & {test_mpjpe_meters[-1].avg:.1f} & {test_mpjpe_pa_meters[-1].avg:.1f} & {100*test_pck_meters[-1].avg:.1f} & {100*test_pck_pa_meters[-1].avg:.1f}")

    ikea_joint_groups_names = get_ikea_joint_groups_names()
    print(f"PCK @ {args.pck_threshold_3d} (test): {', '.join(ikea_joint_groups_names)}")
    print(f"{' & '.join([f'{100*test_pck:.1f}' for test_pck in test_pck_group])}")

    print(f"PCK-PA @ {args.pck_threshold_3d} (test): {', '.join(ikea_joint_groups_names)}")
    print(f"{' & '.join([f'{100*test_pck_pa:.1f}' for test_pck_pa in test_pck_pa_group])}")

    test_errors = np.array(test_errors[-1])
    test_errors_pa = np.array(test_errors_pa[-1])
    train_errors = np.array(train_errors[-1])
    train_errors_pa = np.array(train_errors_pa[-1])
    test_medpjpe = np.median(test_errors[np.isfinite(test_errors)])
    test_medpjpe_pa = np.median(test_errors_pa[np.isfinite(test_errors_pa)])
    train_medpjpe = np.median(train_errors[np.isfinite(train_errors)])
    train_medpjpe_pa = np.median(train_errors_pa[np.isfinite(train_errors_pa)])

    print(f"MedPJPE (train) & MedPJPE-PA (train) & MedPJPE (test) & MedPJPE-PA (test)")
    print(f"{train_medpjpe:.1f} & {train_medpjpe_pa:.1f} & {test_medpjpe:.1f} & {test_medpjpe_pa:.1f}")

    print(f"MPJPE-Reproj (train) & PCK-Reproj @ {args.pck_threshold_2d} (train) & MPJPE-Reproj (test) & PCK-Reproj @ {args.pck_threshold_2d} (test)")
    print(f"{train_mpjpe_reproj_meters[-1].avg:.1f} & {100*train_pck_reproj_meters[-1].avg:.1f} & {test_mpjpe_reproj_meters[-1].avg:.1f} & {100*test_pck_reproj_meters[-1].avg:.1f}")

def compute_error_measures_3d(args, results):
    num_frames = results.shape[0]
    num_joints = len(get_ikea_joint_names())
    mpjpe_meters = [AverageMeter('MPJPE', ':.1f') for _ in range(num_joints + 1)] # joints + all
    mpjpe_pa_meters = [AverageMeter('MPJPE-PA', ':.1f') for _ in range(num_joints + 1)] # joints + all
    mpjpe_reproj_meters = [AverageMeter('MPJPE Reprojection', ':.1f') for _ in range(num_joints + 1)] # joints + all
    pck_meters = [AverageMeter('PCK', ':.1f') for _ in range(num_joints + 1)] # joints + all
    pck_pa_meters = [AverageMeter('PCK-PA', ':.1f') for _ in range(num_joints + 1)] # joints + all
    pck_reproj_meters = [AverageMeter('PCK Reprojection', ':.1f') for _ in range(num_joints + 1)] # joints + all
    errors = [[] for _ in range(num_joints + 1)]
    errors_pa = [[] for _ in range(num_joints + 1)]
    errors_reproj = [[] for _ in range(num_joints + 1)]
    for i in range(num_frames):
        for j in range(num_joints):
            error = results[i, j, 0]
            error_pa = results[i, j, 1]
            error_reprojection = results[i, j, 2]
            confidence_gt = results[i, j, 3]
            confidence_2d_gt = results[i, j, 4]
            if confidence_gt >= 0.0 and confidence_2d_gt == 3: # Detected 3D pGT annotations that correspond to confident 2D GT annotations
                if np.isfinite(error):
                    mpjpe_meters[j].update(error, 1) # joint
                    mpjpe_meters[-1].update(error, 1) # all
                if np.isfinite(error_pa):
                    mpjpe_pa_meters[j].update(error_pa, 1) # joint
                    mpjpe_pa_meters[-1].update(error_pa, 1) # all
                if np.isfinite(error_reprojection):
                    mpjpe_reproj_meters[j].update(error_reprojection, 1) # joint
                    mpjpe_reproj_meters[-1].update(error_reprojection, 1) # all

                if error <= args.pck_threshold_3d: # In millimetres
                    pck_meters[j].update(1, 1)
                    pck_meters[-1].update(1, 1)
                else:
                    pck_meters[j].update(0, 1)
                    pck_meters[-1].update(0, 1)
                if error_pa <= args.pck_threshold_3d: # In millimetres
                    pck_pa_meters[j].update(1, 1)
                    pck_pa_meters[-1].update(1, 1)
                else:
                    pck_pa_meters[j].update(0, 1)
                    pck_pa_meters[-1].update(0, 1)

                if error_reprojection <= args.pck_threshold_2d: # In pixels
                    pck_reproj_meters[j].update(1, 1)
                    pck_reproj_meters[-1].update(1, 1)
                else:
                    pck_reproj_meters[j].update(0, 1)
                    pck_reproj_meters[-1].update(0, 1)

                # Store errors
                errors[j].append(error)
                errors[-1].append(error)
                errors_pa[j].append(error_pa)
                errors_pa[-1].append(error_pa)
                errors_reproj[j].append(error_reprojection)
                errors_reproj[-1].append(error_reprojection)
    return mpjpe_meters, pck_meters, errors, mpjpe_pa_meters, pck_pa_meters, errors_pa, mpjpe_reproj_meters, pck_reproj_meters, errors_reproj

def compute_similarity_transform(S1, S2):
    # Borrowed from
    # Author: Angjoo Kanazawa
    # URL: https://github.com/akanazawa/human_dynamics/blob/master/src/evaluation/eval_util.py
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def crop_to_orig(s, tx, ty, cx, cy, bbox_side=672, image_width=1920., image_height=1080.):
    h = bbox_side
    hw, hh = image_width / 2., image_height / 2.
    sx = s * (1. / (image_width / h))
    sy = s * (1. / (image_height / h))
    tx = ((cx - hw) / hw / sx) + tx
    ty = ((cy - hh) / hh / sy) + ty
    return sx, sy, tx, ty

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

def path_to_types(path):
    furniture_type, tag = path.split('/')
    tags = tag.split('_')
    person_id = int(tags[0])
    colour = tags[1]
    surface = tags[2]
    calibration_id = int(tags[3])
    room_id = int(tags[4])
    datetime = tags[5:10]
    return furniture_type, person_id, colour, surface, calibration_id, room_id, datetime

def video_to_keyframes(args):
    """Convert videos to frames and keep those with corresponding GT
    """
    gt_dirs = get_gt_dirs(args.dataset_dir, args.camera_id)  # Only those folders with pose2d annotations
    for i, gt_dir in enumerate(gt_dirs):
        print(f"\nProcessing {i} of {len(gt_dirs)}: {' '.join(gt_dir.split('/')[-3:-1])}")
        pose2d_gt_filenames = os.listdir(os.path.join(gt_dir, 'pose2d'))
        image_folder = os.path.join(gt_dir, 'images')
        rgb_video_file = os.path.join(image_folder, 'scan_video.avi')
        depth_folder = os.path.join(gt_dir, 'depth')
        depth_video_file = os.path.join(depth_folder, 'scan_video.avi')

        for pose2d_gt_filename in pose2d_gt_filenames:
            frame_id = int(os.path.splitext(pose2d_gt_filename)[0])
            command = ['ffmpeg',
                       '-i', rgb_video_file,
                       '-f', 'image2',
                       '-v', 'error',
                       '-vf', f"select='eq(n\,{frame_id})'",
                       f'{image_folder}/{frame_id:06d}.png']
            subprocess.call(command)
            command = ['ffmpeg',
                       '-i', depth_video_file,
                       '-f', 'image2',
                       '-v', 'error',
                       '-vf', f"select='eq(n\,{frame_id})'",
                       f'{depth_folder}/{frame_id:06d}.png']
            subprocess.call(command)

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
    subdirs = [os.path.join(input_path, dir_i) for dir_i in os.listdir(input_path)
               if os.path.isdir(os.path.join(input_path, dir_i))]
    subdirs.sort()
    return subdirs

def visualise_json_ikea(json_path, image_path):
    with open(json_path) as json_file:
        joints = json.load(json_file)
    img = cv.imread(image_path)

    plt.imshow(img)
    for i, joint_name in enumerate(get_ikea_joint_names()):
        j2d = joints[joint_name]["position"]
        plt.scatter(j2d[0], j2d[1], c='w')
    plt.show()

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_openpose_pt', action='store_true',
                        help='evaluate pretrained OpenPose results')
    parser.add_argument('--eval_openpose_ft', action='store_true',
                        help='evaluate finetuned OpenPose results')
    parser.add_argument('--eval_openpose_staf', action='store_true',
                        help='evaluate OpenPose video results')
    parser.add_argument('--eval_keypoint_rcnn_pt', action='store_true',
                        help='evaluate pretrained Keypoint R-CNN results')
    parser.add_argument('--eval_keypoint_rcnn_ft', action='store_true',
                        help='evaluate finetuned Keypoint R-CNN results')
    parser.add_argument('--eval_vibe', action='store_true',
                        help='evaluate VIBE results')
    parser.add_argument('--eval_hmmr', action='store_true',
                        help='evaluate HMMR results')
    parser.add_argument('--eval_vp3d', action='store_true',
                        help='evaluate VideoPose3D results')
    parser.add_argument('--dataset_dir', type=str, default='/home/djcam/Documents/HDD/datasets/ikea/ikea_asm/',
                        help='directory of the IKEA assembly dataset')
    parser.add_argument('--camera_id', type=str, default='dev3',
                        help='camera device ID for dataset (GT annotations for dev3 only)')
    parser.add_argument('--pck_threshold_2d', type=float, default=10.0,
                        help='threshold for PCK measure in pixels')
    parser.add_argument('--pck_threshold_3d', type=float, default=150.0,
                        help='threshold for PCK measure in millimetres')
    parser.add_argument('--keypoint_score_threshold', type=float, default=0.0,
                        help='threshold for keypoint score')
    parser.add_argument('--category', type=str, default='', choices=['', 'female', 'male', 'floor', 'table'],
                        help='category over which to report results')
    args = parser.parse_args()

    if args.eval_openpose_pt:
        eval_type = "openpose"
        eval2d(args, eval_type=eval_type)
        parse_results_2d(args, results_file=f'./results/results_{eval_type}.pkl', score_threshold=args.keypoint_score_threshold)
    if args.eval_openpose_ft:
        eval_type = "openpose_ft"
        eval2d(args, eval_type=eval_type)
        parse_results_2d(args, results_file=f'./results/results_{eval_type}.pkl', score_threshold=args.keypoint_score_threshold)
    if args.eval_openpose_staf:
        eval_type = "openpose_staf"
        eval2d(args, eval_type=eval_type)
        parse_results_2d(args, results_file=f'./results/results_{eval_type}.pkl', score_threshold=args.keypoint_score_threshold)
    if args.eval_keypoint_rcnn_pt:
        eval_type = "keypoint_rcnn_pt"
        eval2d(args, eval_type=eval_type)
        parse_results_2d(args, results_file=f'./results/results_{eval_type}.pkl', score_threshold=args.keypoint_score_threshold)
    if args.eval_keypoint_rcnn_ft:
        eval_type = "keypoint_rcnn_ft"
        eval2d(args, eval_type=eval_type)
        parse_results_2d(args, results_file=f'./results/results_{eval_type}.pkl', score_threshold=args.keypoint_score_threshold)
    if args.eval_vibe:
        eval_type = "vibe"
        eval3d(args, eval_type=eval_type)
        parse_results_3d(args, results_file=f'./results/results_{eval_type}.pkl')
    if args.eval_hmmr:
        eval_type = "hmmr"
        eval3d(args, eval_type=eval_type)
        parse_results_3d(args, results_file=f'./results/results_{eval_type}.pkl')
    if args.eval_vp3d:
        eval_type = "vp3d"
        eval3d(args, eval_type=eval_type)
        parse_results_3d(args, results_file=f'./results/results_{eval_type}.pkl')

