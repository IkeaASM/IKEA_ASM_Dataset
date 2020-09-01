# Estimate extrinsic camera parameters
#
# camera calibration for distorted images with chess board samples
#
# based on: https://github.com/opencv/opencv/blob/master/samples/python/calibrate.py
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
from pdb import set_trace as st

scenes = ['01', '02', '04', '05', '06', '07', '08', '09', '10', '11']
cams = ['dev1', 'dev2', 'dev3']

def get_extrinsic_parameters(args):
    calib_path = os.path.join(args.dataset_dir, 'Calibration')
    for scene in scenes:
        scene_path = os.path.join(calib_path, scene)
        
        cam = 'dev2' # Compute cameras w.r.t dev2
        cam_path = os.path.join(scene_path, cam)
        img_mask = os.path.join(cam_path, 'images', '????.png')
        img_names = glob(img_mask)
        color_param_filename = os.path.join(cam_path, 'ColorIns.txt')
        rgb_ins_params = get_rgb_ins_params(color_param_filename)

        cam1 = 'dev1' # Compute cameras w.r.t dev2
        cam_path1 = os.path.join(scene_path, cam1)
        img_mask1 = os.path.join(cam_path1, 'images', '????.png')
        img_names1 = glob(img_mask1)
        cam3 = 'dev3' # Compute cameras w.r.t dev2
        cam_path3 = os.path.join(scene_path, cam3)
        img_mask3 = os.path.join(cam_path3, 'images', '????.png')
        img_names3 = glob(img_mask3)

        pattern_size = (4, 3) # Number of inner corners per a chessboard row and column
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= args.square_size

        obj_points = []
        img_points = []
        img_points1 = []
        img_points3 = []
        h, w = cv.imread(img_names[0], cv.IMREAD_GRAYSCALE).shape[:2]

        def processImage(fn):
            # print('processing %s... ' % fn)
            img = cv.imread(fn, 0)
            if img is None:
                print("Failed to load", fn)
                return None

            # img = cv.flip(img, 1) # Flip LR
            # cv.imwrite(fn, img)

            assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
            found, corners = cv.findChessboardCorners(img, pattern_size)
            if found:
                term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
                cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
            if args.debug_dir:
                vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
                cv.drawChessboardCorners(vis, pattern_size, corners, found)
                _, name, _ = splitfn(fn)
                outfile = os.path.join(args.debug_dir, name + '_chess.png')
                cv.imwrite(outfile, vis)
            if not found:
                print('chessboard not found')
                return None
            # print('           %s... OK' % fn)
            return (corners.reshape(-1, 2), pattern_points)

        threads_num = args.num_threads
        if threads_num <= 1:
            chessboards1 = [processImage(fn) for fn in img_names1]
            chessboards3 = [processImage(fn) for fn in img_names3]
            chessboards = [processImage(fn) for fn in img_names]
        else:
            # print("Run with %d threads..." % threads_num)
            from multiprocessing.dummy import Pool as ThreadPool
            pool = ThreadPool(threads_num)
            chessboards1 = pool.map(processImage, img_names1)
            chessboards3 = pool.map(processImage, img_names3)
            chessboards = pool.map(processImage, img_names)

        chessboards = [x for x in chessboards if x is not None]
        chessboards1 = [x for x in chessboards1 if x is not None]
        chessboards3 = [x for x in chessboards3 if x is not None]
        for (corners, pattern_points) in chessboards:
            img_points.append(corners)
            obj_points.append(pattern_points)
        for (corners, pattern_points) in chessboards1:
            img_points1.append(corners)
        for (corners, pattern_points) in chessboards3:
            img_points3.append(corners)

        # Calibrate cameras:
        camera_matrix_gt = np.float32(np.array([[rgb_ins_params["fx"], 0.0, rgb_ins_params["cx"]], [0.0, rgb_ins_params["fy"], rgb_ins_params["cy"]], [0.0, 0.0, 1.0]])) # fx and fy 
        dist_coefs_gt = np.float32(np.array([0.0, 0.0, 0.0, 0.0]))
        flags=cv.CALIB_USE_INTRINSIC_GUESS + cv.CALIB_FIX_PRINCIPAL_POINT + cv.CALIB_FIX_ASPECT_RATIO + cv.CALIB_ZERO_TANGENT_DIST + cv.CALIB_FIX_K1 + cv.CALIB_FIX_K2 + cv.CALIB_FIX_K3 + cv.CALIB_FIX_K4 + cv.CALIB_FIX_K5 + cv.CALIB_FIX_K6
        rms, camera_matrix, dist_coefs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), camera_matrix_gt, dist_coefs_gt, flags=flags)
        rms1, camera_matrix1, dist_coefs1, rvecs1, tvecs1 = cv.calibrateCamera(obj_points, img_points1, (w, h), camera_matrix_gt, dist_coefs_gt, flags=flags)
        rms3, camera_matrix3, dist_coefs3, rvecs3, tvecs3 = cv.calibrateCamera(obj_points, img_points3, (w, h), camera_matrix_gt, dist_coefs_gt, flags=flags)

        # if debug: undistort the image with the calibration
        for fn in img_names if args.debug_dir else []:
            _path, name, _ext = splitfn(fn)
            img_found = os.path.join(args.debug_dir, name + '_chess.png')
            outfile = os.path.join(args.debug_dir, name + '_undistorted.png')
            img = cv.imread(img_found)
            if img is None:
                continue
            h, w = img.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
            dst = cv.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
            # crop and save the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            print('Undistorted image written to: %s' % outfile)
            cv.imwrite(outfile, dst)

        flags=cv.CALIB_FIX_INTRINSIC
        rms, camera_matrix, dist_coefs, camera_matrix1, dist_coefs1, R21, T21, _, _ = cv.stereoCalibrate(obj_points, img_points, img_points1, camera_matrix, dist_coefs, camera_matrix1, dist_coefs1, (w, h), flags=flags)
        rms, camera_matrix, dist_coefs, camera_matrix3, dist_coefs3, R23, T23, _, _ = cv.stereoCalibrate(obj_points, img_points, img_points3, camera_matrix, dist_coefs, camera_matrix3, dist_coefs3, (w, h), flags=flags)

        camera_parameters = {
            "K": camera_matrix,
            "dist_coefs": dist_coefs,
            "R": np.eye(3),
            "T": np.zeros((3,1))
        }
        with open(os.path.join(cam_path, 'camera_parameters.pkl'), 'wb') as f:
            pickle.dump(camera_parameters, f)
        with open(os.path.join(cam_path, 'camera_parameters.json'), 'w') as outfile:
            camera_parameters_serialized = {key: value.tolist() for key, value in camera_parameters.items()}
            json.dump(camera_parameters_serialized, outfile)

        camera_parameters = {
            "K": camera_matrix1,
            "dist_coefs": dist_coefs1,
            "R": R21,
            "T": T21
        }
        with open(os.path.join(cam_path1, 'camera_parameters.pkl'), 'wb') as f:
            pickle.dump(camera_parameters, f)
        with open(os.path.join(cam_path1, 'camera_parameters.json'), 'w') as outfile:
            camera_parameters_serialized = {key: value.tolist() for key, value in camera_parameters.items()}
            json.dump(camera_parameters_serialized, outfile)

        camera_parameters = {
            "K": camera_matrix3,
            "dist_coefs": dist_coefs3,
            "R": R23,
            "T": T23
        }
        with open(os.path.join(cam_path3, 'camera_parameters.pkl'), 'wb') as f:
            pickle.dump(camera_parameters, f)
        with open(os.path.join(cam_path3, 'camera_parameters.json'), 'w') as outfile:
            camera_parameters_serialized = {key: value.tolist() for key, value in camera_parameters.items()}
            json.dump(camera_parameters_serialized, outfile)

def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext

def get_rgb_ins_params(param_file):
    '''
    read the rgb intrinsic parameters file
    :param param_file: path to intrinsic parameters file
    :return:
    rgb_ins_params: a libfreenect2 ColorCameraParams object
    '''
    with open(param_file, 'r') as f:
        rgb_ins_params = [float(line.strip()) for line in f if line]

    rgb_camera_params_obj = {
        "fx" : rgb_ins_params[0],
        "fy" : rgb_ins_params[1],
        "cx" : rgb_ins_params[2],
        "cy" : rgb_ins_params[3],

        "shift_d" : rgb_ins_params[4],
        "shift_m" : rgb_ins_params[5],
        "mx_x3y0" : rgb_ins_params[6],
        "mx_x0y3" : rgb_ins_params[7],
        "mx_x2y1" : rgb_ins_params[8],
        "mx_x1y2" : rgb_ins_params[9],
        "mx_x2y0" : rgb_ins_params[10],
        "mx_x0y2" : rgb_ins_params[11],
        "mx_x1y1" : rgb_ins_params[12],
        "mx_x1y0" : rgb_ins_params[13],
        "mx_x0y1" : rgb_ins_params[14],
        "mx_x0y0" : rgb_ins_params[15],

        "my_x3y0" : rgb_ins_params[16],
        "my_x0y3" : rgb_ins_params[17],
        "my_x2y1" : rgb_ins_params[18],
        "my_x1y2" : rgb_ins_params[19],
        "my_x2y0" : rgb_ins_params[20],
        "my_x0y2" : rgb_ins_params[21],
        "my_x1y1" : rgb_ins_params[22],
        "my_x1y0" : rgb_ins_params[23],
        "my_x0y1" : rgb_ins_params[24],
        "my_x0y0" : rgb_ins_params[25]
    }
    return rgb_camera_params_obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='/home/djcam/Documents/HDD/datasets/ikea/ikea_asm/',
                        help='directory of the IKEA assembly dataset')
    parser.add_argument('--square_size', type=float, default=4.0,
                        help='calibration chessboard square size (in centimetres)')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='number of threads for chessboard function')
    parser.add_argument('--debug_dir', type=str, default='',
                        help='path for debug chessboard images')
    args = parser.parse_args()

    get_extrinsic_parameters(args)