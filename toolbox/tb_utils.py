import os
import numpy as np
import cv2
import pathlib
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement

from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Kernel import Vector_3
from CGAL.CGAL_Point_set_processing_3 import *
#import pypcd
from joblib import Parallel, delayed
import multiprocessing
import json
import random
import sys

sys.path.append('../action/')
sys.path.append('../human_pose/')
from IKEAActionDataset import IKEAActionDataset as Dataset
import joint_ids

import math
import trimesh
import pyrender
from pyrender.constants import RenderFlags
from smpl import get_smpl_faces
# import smplx


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


def get_files(input_path, file_type='.png'):
    '''
    get a list of files in input_path directory
    :param input_path: parent directory (in which to get the file list)
    :return:
    files: list of files in input_path
    '''
    files = [os.path.join(input_path, file_i) for file_i in os.listdir(input_path)
               if os.path.isfile(os.path.join(input_path, file_i)) and file_i.endswith(file_type)]
    files.sort()
    return files


def get_list_of_all_files(dir_path, file_type='.jpg'):
    '''
    get a list of all files of a given type in input_path directory
    :param dir_path: parent directory (in which to get the file list)
    :return:
    allFiles: list of files in input_path
    '''
    listOfFile = os.listdir(dir_path)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dir_path, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + get_list_of_all_files(fullPath, file_type=file_type)
        else:
            if fullPath.endswith(file_type):
                allFiles.append(fullPath)

    return allFiles

def get_depth_ins_params(param_file):
    '''
    read the depth intrinsic parameters file
    :param param_file: path to depth intrinsic parameters file DepthIns.txt
    :return:
    ir_camera_params_obj: a libfreenect2 IrCameraParams object
    '''
    with open(param_file, 'r') as f:
        depth_ins_params = [float(line.strip()) for line in f if line]
    ir_camera_params_obj = {
        "fx" : depth_ins_params[0],
        "fy" : depth_ins_params[1],
        "cx" : depth_ins_params[2],
        "cy" : depth_ins_params[3],
        "k1" : depth_ins_params[4],
        "k2" : depth_ins_params[5],
        "k3" : depth_ins_params[6],
        "p1" : depth_ins_params[7],
        "p2" : depth_ins_params[8]
    }
    return ir_camera_params_obj


def get_rgb_ins_params(param_file):
    '''
    read the rgb intrinsic parameters file
    :param param_file: path to depth intrinsic parameters file DepthIns.txt
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


def get_prop_path_list(input_path, property_name='normals'):
    category_path_list = get_subdirs(input_path)
    scan_path_list = []
    for category in category_path_list:
        if os.path.basename(category) != 'Calibration':
            category_scans = get_subdirs(category)
            for category_scan in category_scans:
                scan_path_list.append(category_scan)
    pose_path_list = []
    for scan in scan_path_list:
        device_list = get_subdirs(scan)
        for device in device_list:
            pose_path = os.path.join(device, property_name)  #  property names: 'normals','point_clouds'
            if os.path.exists(pose_path):
                pose_path_list.append(pose_path)
    return pose_path_list


def get_scan_list(input_path, devices='all'):
    '''
    get_scan_list retreieves all of the subdirectories under the dataset main directories:
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
    depth_path_list = []
    normals_path_list = []
    rgb_params_files = []
    depth_params_files = []
    for scan in scan_path_list:
        device_list = get_subdirs(scan)
        for device in device_list:
            if os.path.basename(device) in devices:
                rgb_path = os.path.join(device, 'images')
                depth_path = os.path.join(device, 'depth')
                normals_path = os.path.join(device, 'normals')
                if os.path.exists(rgb_path):
                    rgb_path_list.append(rgb_path)
                    rgb_params_files.append(os.path.join(device, 'ColorIns.txt'))
                if os.path.exists(depth_path):
                    if 'dev3' in device:  # remove redundant depths - remove this line for full 3 views
                        depth_path_list.append(depth_path)
                        depth_params_files.append(os.path.join(device, 'DepthIns.txt'))
                if os.path.exists(normals_path):
                    normals_path_list.append(normals_path)

    return category_path_list, scan_path_list, rgb_path_list, depth_path_list, depth_params_files, rgb_params_files, normals_path_list


def distort(mx, my, depth_ins):
    # see http://en.wikipedia.org/wiki/Distortion_(optics) for description
    # based on the C++ implementation in libfreenect2
    dx = (float(mx) - depth_ins['cx']) / depth_ins['fx']
    dy = (float(my) - depth_ins['cy']) / depth_ins['fy']
    dx2 = np.square(dx)
    dy2 = np.square(dy)
    r2 = dx2 + dy2
    dxdy2 = 2 * dx * dy
    kr = 1 + ((depth_ins['k3'] * r2 + depth_ins['k2']) * r2 + depth_ins['k1']) * r2
    x = depth_ins['fx'] * (dx * kr + depth_ins['p2'] * (r2 + 2 * dx2) + depth_ins['p1'] * dxdy2) + depth_ins['cx']
    y = depth_ins['fy'] * (dy * kr + depth_ins['p1'] * (r2 + 2 * dy2) + depth_ins['p2'] * dxdy2) + depth_ins['cy']
    return x, y


def depth_to_color(mx, my, dz, depth_ins, color_ins):
    # based on the C++ implementation in libfreenect2, constants are hardcoded into sdk
    depth_q = 0.01
    color_q = 0.002199

    mx = (mx - depth_ins['cx']) * depth_q
    my = (my - depth_ins['cy']) * depth_q

    wx = (mx * mx * mx * color_ins['mx_x3y0']) + (my * my * my * color_ins['mx_x0y3']) + \
         (mx * mx * my * color_ins['mx_x2y1']) + (my * my * mx * color_ins['mx_x1y2']) + \
         (mx * mx * color_ins['mx_x2y0']) + (my * my * color_ins['mx_x0y2']) + \
         (mx * my * color_ins['mx_x1y1']) +(mx * color_ins['mx_x1y0']) + \
         (my * color_ins['mx_x0y1']) + (color_ins['mx_x0y0'])

    wy = (mx * mx * mx * color_ins['my_x3y0']) + (my * my * my * color_ins['my_x0y3']) +\
         (mx * mx * my * color_ins['my_x2y1']) + (my * my * mx * color_ins['my_x1y2']) +\
         (mx * mx * color_ins['my_x2y0']) + (my * my * color_ins['my_x0y2']) + (mx * my * color_ins['my_x1y1']) +\
         (mx * color_ins['my_x1y0']) + (my * color_ins['my_x0y1']) + color_ins['my_x0y0']

    rx = (wx / (color_ins['fx'] * color_q)) - (color_ins['shift_m'] / color_ins['shift_d'])
    ry = int((wy / color_q) + color_ins['cy'])

    rx = rx + (color_ins['shift_m'] / dz)
    rx = int(rx * color_ins['fx'] + color_ins['cx'])
    return rx, ry


def getPointXYZ(undistorted, r, c, depth_ins):
    depth_val = undistorted[r, c] #/ 1000.0  # map from mm to meters
    if np.isnan(depth_val) or depth_val <= 0.001:
        x = 0
        y = 0
        z = 0
    else:
        x = (c + 0.5 - depth_ins['cx']) * depth_val / depth_ins['fx']
        y = (r + 0.5 - depth_ins['cy']) * depth_val / depth_ins['fy']
        z = depth_val
    point = [x, y, z]
    return point


def get_rc_from_xyz(x, y, z, depth_ins):
    '''
    project a 3d point back to the image row and column indices
    :param point: xyz 3D point
    :param depth_ins: depth camera intrinsic parameters
    :return:
    '''
    c = int((x * depth_ins['fx'] / z) + depth_ins['cx'])
    r = int((y * depth_ins['fy'] / z) + depth_ins['cy'])
    return c, r


def apply(dz, rx, ry, rgb_ins_params):
    cy = int(ry)
    rx = rx + (rgb_ins_params['shift_m'] / dz)
    cx = int(rx * rgb_ins_params['fx'] + rgb_ins_params['cx'])
    return cx, cy


def compute_point_clouds(depth_ins_params, rgb_ins_params, depth_frames, rgb_frames, j, point_cloud_dir_name):
    frame_filename, depth_frame_file_extension = os.path.splitext(os.path.basename(depth_frames[j]))
    depth_img = cv2.imread(depth_frames[j], cv2.IMREAD_ANYDEPTH).astype(np.float32)
    depth_img = cv2.flip(depth_img, 1)  # images were saved flipped
    # relative_depth_frame = get_relative_depth(depth_img)
    color_img = cv2.imread(rgb_frames[j])
    color_img = cv2.flip(color_img, 1)  # images were saved flipped

    # # Compute point cloud from images
    point_cloud = []
    new_vertices = []
    registered = np.zeros([424, 512, 3], dtype=np.uint8)
    undistorted = np.zeros([424, 512, 3], dtype=np.float32)
    for y in range(424):
        for x in range(512):
            # point = registration.getPointXYZ(depth_frame, y, x)
            mx, my = distort(x, y, depth_ins_params)
            ix = int(mx + 0.5)
            iy = int(my + 0.5)
            point = getPointXYZ(depth_img, y, x, depth_ins_params)
            if (ix >= 0 and ix < 512 and iy >= 0 and iy < 424):  # check if pixel is within the image
                undistorted[iy, ix] = depth_img[y, x]
                z = depth_img[y, x]
                if z > 0 and not np.isnan(z):
                    # point_cloud.append(point)
                    cx, cy = depth_to_color(x, y, z, depth_ins_params, rgb_ins_params)
                    if (cx >= 0 and cx < 1920 and cy >= 0 and cy < 1080):
                        registered[y, x, :] = color_img[cy, cx].flatten()
                        registered[y, x, :] = cv2.cvtColor(registered[y, x].reshape([1, 1, 3]),
                                                           cv2.COLOR_BGR2RGB)

                        point_cloud.append([(point), registered[y, x, :]])
                        new_vertices.append((point[0], point[1], point[2], registered[y, x, 0],
                                             registered[y, x, 1], registered[y, x, 2]))

    # cv2.imshow('relative depth', relative_depth_frame)
    # cv2.imshow('registered image',  cv2.cvtColor(registered,cv2.COLOR_RGB2BGR))
    # cv2.waitKey(0)
    # if point_cloud_type == '.ply':
    pc_file_name = os.path.join(point_cloud_dir_name, frame_filename + '.ply')

    new_vertices = np.array(new_vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(new_vertices, 'vertex')
    PlyData([el]).write(pc_file_name)
    print('Saved Point cloud ' + frame_filename + '.ply to ' + point_cloud_dir_name)
    # else:
    # Add support for other file types
    # pc_file_name = os.path.join(point_cloud_dir_name, frame_filename+'.pcd')
    # pcl.save(new_vertices, pc_file_name, format='.pcd', binary=False)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # point_cloud = np.array(point_cloud)
    # ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], marker='.', s=1, c=point_cloud[:, 3:])
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()

def export_point_clouds(input_path, pc_per_scan=1.0, include_rgb=True, point_cloud_type='.ply', output_path='',
                        devices='dev3'):
    '''
    Export point clouds per frame. Assumes sync and same number of depth and rgb frames.
    :param input_path: path to dataset
    :param pc_per_scan: fration of point clouds to generate, default generate all
    :return:
    '''
    if output_path == '':
        output_path = input_path

    category_path_list, scan_path_list, rgb_path_list, depth_path_list, depth_params_files,\
    rgb_params_files, normals_path_list = get_scan_list(input_path, devices=devices)

    for i, scan in enumerate(depth_path_list):
        depth_param_filename = depth_params_files[i]
        color_param_filename = rgb_params_files[i]
        depth_frames = get_files(scan, file_type='.png')
        rgb_frames = get_files(rgb_path_list[i], file_type='.jpg')
        if depth_frames:  # check if empty
            depth_ins_params = get_depth_ins_params(depth_param_filename)
        if rgb_frames:
            rgb_ins_params = get_rgb_ins_params(color_param_filename)

        out_dir = scan.replace(scan[:scan.index("ANU_ikea_dataset") + 17], output_path)
        point_cloud_dir_name = os.path.join(os.path.abspath(os.path.join(out_dir, os.pardir)), 'point_clouds')
        if not os.path.exists(point_cloud_dir_name):
            os.makedirs(point_cloud_dir_name)

        # for j, _ in enumerate(depth_frames):
        #  parallel point cloud computation and export
        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(compute_point_clouds)(depth_ins_params, rgb_ins_params, depth_frames, rgb_frames, j, point_cloud_dir_name) for j, _ in enumerate(depth_frames))


######################################### Video export utils #######################################
def get_relative_depth(img, min_depth_val=0.0, max_depth_val = 4500, colormap='jet'):
    '''
    Convert the depth image to relative depth for better visualization. uses fixed minimum and maximum distances
    to avoid flickering
    :param img: depth image
           min_depth_val: minimum depth in mm (default 50cm)
           max_depth_val: maximum depth in mm ( default 10m )
    :return:
    relative_depth_frame: relative depth converted into cv2 GBR
    '''

    relative_depth_frame = cv2.convertScaleAbs(img, alpha=(255.0/max_depth_val),
                                               beta=-255.0*min_depth_val/max_depth_val)
    relative_depth_frame = cv2.cvtColor(relative_depth_frame, cv2.COLOR_GRAY2BGR)
    # relative_depth_frame = cv2.applyColorMap(relative_depth_frame, cv2.COLORMAP_JET)
    return relative_depth_frame


def get_absolute_depth(img, min_depth_val=0.0, max_depth_val = 4500, colormap='jet'):
    '''
    Convert the relative depth image to absolute depth. uses fixed minimum and maximum distances
    to avoid flickering
    :param img: depth image
           min_depth_val: minimum depth in mm (default 50cm)
           max_depth_val: maximum depth in mm ( default 10m )
    :return:
    absolute_depth_frame: absolute depth converted into cv2 gray
    '''

    absolute_depth_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # absolute_depth_frame = cv2.convertScaleAbs(absolute_depth_frame, alpha=(max_depth_val/255.0),
    #                                            beta=min_depth_val).astype(np.uint16) #converts to 8 bit
    absolute_depth_frame = absolute_depth_frame * float(max_depth_val/255.0)
    return absolute_depth_frame.astype(np.uint16)


def get_fps(frames, native=False):
    if native:
        start = os.path.getmtime(frames[0])
        end = os.path.getmtime(frames[-1])
        duration = end - start
        n_frames = len(frames)
        fps = n_frames / duration
    else:
        fps = 30
    return fps


def export_video_par(scan, depth_flag, normals_flag, output_path, fps, overide, vid_scale, vid_format):
    """
    function that can run in parallel to export videos from individual frames. supports avi and webm.
    Parameters
    ----------
    scan : scan name [string]
    depth_flag :  true | false - use depth stream
    normals_flag :  true | false - use normal vectors stream
    output_path : outpue path
    fps : frames per seconds
    overide : true | false if to override existing files in the output dir[bool]
    vid_scale :   indicting the scale. to unscale use 1. [float]
    vid_format : 'avi' | 'webm' video format [string]

    Returns
    -------

    """

    if 'processed' not in scan:  # if output path changes this will break
        out_dir = scan.replace(scan[:scan.index("ANU_ikea_dataset") + 17], output_path)
    else:
        out_dir = scan

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    video_file_name = os.path.join(out_dir, 'scan_video.' + vid_format)
    if not os.path.exists(video_file_name) or overide:
        frames = get_files(scan, file_type='.png') if depth_flag or normals_flag else get_files(scan, file_type='.jpg')
        if frames:

            if depth_flag:
                img = cv2.imread(frames[0], cv2.IMREAD_ANYDEPTH)
                img = get_relative_depth(img.astype(np.float32)) # better for visualization
            else:
                img = cv2.imread(frames[0])
            if not vid_scale == 1:
                img = cv2.resize(img, dsize=(int(img.shape[1]/vid_scale), int(img.shape[0]/vid_scale)))
            img_shape = img.shape
            size = (img_shape[1], img_shape[0])
            print('Saving  video file to ' + video_file_name)
            fourcc = cv2.VideoWriter_fourcc(*'VP09') if vid_format == 'webm' else cv2.VideoWriter_fourcc(*'DIVX')
            out = cv2.VideoWriter(video_file_name, fourcc, fps, size)

            for j, _ in enumerate(frames):
                if depth_flag:
                    img = cv2.imread(frames[j], cv2.IMREAD_ANYDEPTH).astype(np.float32)
                    img = get_relative_depth(img) # use default min depth 0 and max depth 4500 for reconstruction from the video
                else:
                    img = cv2.imread(frames[j])
                if not vid_scale == 1:
                    img = cv2.resize(img, dsize=(int(img.shape[1] / vid_scale), int(img.shape[0] / vid_scale)))
                out.write(img)

        out.release()
        print('Done\n')


def video_export_helper(path_list, depth_flag=False, normals_flag=False, output_path='', fps=25, overide=False,
                        vid_scale=1, vid_format='avi'):
    '''
    Helper function to handle different cases of AVI export (for rgb and depth)
    :param path_list: list of paths to image folders to convert to AVI
    :param depth_flag: True / False is for depth
    :return:
    TO DO: Remove this function, after the parallelization it is redundant
    '''
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(export_video_par)(scan, depth_flag, normals_flag, output_path, fps, overide,
                                                         vid_scale, vid_format) for i, scan in enumerate(path_list))

    # for i, scan in enumerate(path_list):





def export_videos(input_path, rgb_flag=False, depth_flag=False, normals_flag=False, pose_flag=False, seg_flag=False,
                  perception_demo_flag=False, output_path='', vid_scale=1, vid_format='avi', overide=False,
                  devices='all'):
    '''
    Saves video files of the scanned rgb and/or depth images
    note: on Ubuntu install SMplayer to view video file
    :param input_path: path to ikea dataset
    :param rgb_flag: True / False if to export the rgb video file
    :param depth_flag: True / False if to export the depth video file
    :param depth_flag: True / False if to export the normals vodeo file
    :param fps: frames per second for videos
    :return:
    '''

    if output_path == '':
        output_path = input_path
    category_path_list, scan_path_list, rgb_path_list, depth_path_list, depth_params_files,\
    rgb_params_files, _ = get_scan_list(input_path, devices=devices)

    if rgb_flag:
        video_export_helper(rgb_path_list, depth_flag=False, output_path=output_path, vid_scale=vid_scale,
                            vid_format=vid_format, overide=overide)
    if depth_flag:
        video_export_helper(depth_path_list, depth_flag=True, output_path=output_path, vid_scale=vid_scale,
                            vid_format=vid_format, overide=overide)

    if normals_flag:
        normals_path_list = get_prop_path_list(output_path, property_name='normals')
        video_export_helper(normals_path_list, depth_flag=False, normals_flag=True, output_path=output_path,
                            vid_scale=vid_scale, vid_format=vid_format, overide=overide)
    if pose_flag:
        # pose_path_list = get_pose_path_list(input_path)
        pose_path_list = get_prop_path_list(output_path, property_name='2DposeImages')
        video_export_helper(pose_path_list, depth_flag=False, output_path=output_path, vid_scale=vid_scale,
                            vid_format=vid_format, overide=overide)
    if seg_flag:
        seg_path_list = get_prop_path_list(output_path, property_name='seg')
        video_export_helper(seg_path_list, depth_flag=False, output_path=output_path, vid_scale=vid_scale,
                            vid_format=vid_format, overide=overide)
    if perception_demo_flag:
        seg_path_list = get_prop_path_list(output_path, property_name='perception_demo')
        video_export_helper(seg_path_list, depth_flag=False, output_path=output_path, vid_scale=vid_scale,
                            vid_format=vid_format, overide=overide)

######################################### Normal Estimation utils #######################################
def convert_normals_to_rgb(normal):
    '''
    Maps the normal direction to the rgb cube and returns the corresponsing rgb color
    :param normal: normal vector
    :return: rgb: corresponsing rgb values
    '''

    r = int(127.5*normal.x()+127.5)
    g = int(127.5*normal.y()+127.5)
    b = int(127.5*normal.z()+127.5)
    rgb = [r, g, b]
    return rgb


def estimate_normals(point_cloud_filename, depth_param_filename, normals_dir_name, n_neighbors=100):
    '''

    :param point_cloud_filename: full path and name of point cloud file to load
    :param depth_param_filename: full path and name of depth parameter file
    :param normals_dir_name: full path to normal estimation output directory
    :param n_neighbors: number of neighbors for computing the normal vector
    :return: None. Saves the normal estimation as  .png file.
    '''
    frame_filename, point_cloud_file_extension = os.path.splitext(os.path.basename(point_cloud_filename))
    depth_ins_params = get_depth_ins_params(depth_param_filename)
    ply = PlyData.read(point_cloud_filename)
    vertex = ply['vertex']
    (x, y, z) = (vertex[t].tolist() for t in ('x', 'y', 'z'))
    points = []
    for j in range(len(x)):
        points.append(Point_3(x[j], y[j], z[j]))
    normals = []
    jet_estimate_normals(points, normals, n_neighbors)  # compute the normal vectors
    new_vertices = []
    normal_image = np.zeros([424, 512, 3], dtype=np.uint8)
    for k, point in enumerate(points):
        if -(point.x() * normals[k].x() + point.y() * normals[k].y() + point.z() * normals[k].z()) < 0:
            normals[k] = -normals[k]  # determine normal ambiguous direction using the camera position
        normal_rgb = convert_normals_to_rgb(normals[k])
        new_vertices.append((point.x(), point.y(), point.z(), normal_rgb[0], normal_rgb[1], normal_rgb[2]))
        r, c = get_rc_from_xyz(point.x(), point.y(), point.z(), depth_ins_params)
        normal_image[c, r] = normal_rgb

    # export normals to .png
    normals_img_file_name = os.path.join(normals_dir_name, frame_filename + '_normals.png')
    cv2.imwrite(normals_img_file_name, cv2.flip(normal_image, 1))
    print('Saved normals image ' + frame_filename + '_normals.png to ' + normals_dir_name)

    # # export the normals to color of ply, normal direction can be stored directly to ply
    # normals_file_name = os.path.join(normals_dir_name, frame_filename+'_normals.ply')
    # new_vertices = np.array(new_vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    #                                  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    # el = PlyElement.describe(new_vertices, 'vertex')
    # PlyData([el]).write(normals_file_name)
    # print('Saved Point cloud ' + frame_filename+'_normals.ply to' + normals_dir_name)


def export_normal_vectors(input_path, n_neighbors = 100, output_path='', devices='dev3'):
    '''
    Saves normal vector images of the scanned point clouds
    :param input_path: path to ikea dataset
    :return: None, exports all png files to output directry
    '''
    if output_path == '':
        output_path = input_path
    category_path_list, scan_path_list, rgb_path_list, depth_path_list, depth_params_files,\
    rgb_params_files, normals_path_list = get_scan_list(input_path, devices=devices)

    for i, scan in enumerate(depth_path_list):
        out_dir = scan.replace(scan[:scan.index("ANU_ikea_dataset") + 17], output_path)
        out_dir = out_dir[:-5]  # remove the depth directory
        point_cloud_path = os.path.join(out_dir, 'point_clouds')
        if os.path.exists(point_cloud_path):
            point_cloud_path_list = get_files(point_cloud_path, file_type='.ply')

            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            normals_dir_name = os.path.join(out_dir, 'normals')
            depth_param_filename = depth_params_files[i]

            if not os.path.exists(normals_dir_name):
                os.makedirs(normals_dir_name)

            # for point_cloud_filename in point_cloud_path_list:
            # parallelize the normal estimation for speed
            num_cores = multiprocessing.cpu_count()

            Parallel(n_jobs=num_cores)(delayed(estimate_normals)(point_cloud_filename, depth_param_filename, normals_dir_name, n_neighbors) for point_cloud_filename in point_cloud_path_list)


def export_video_montage(input_path, output_path, devices='dev3'):
    """
    export a single video containing videos from different captured modalities including depth, rgb,
    normal vectors, point clouds,  2D poses.
    Note: Assumes all modality videos already exist in the input directory
    Parameters
    ----------
    input_path : dataset directory (where all modality videos are available)
    output_path : path to save the  montage videos

    Returns
    -------
        Saves the montage video
    """
    _, scan_path_list, _, _, _, _, _ = get_scan_list(input_path, devices)
    fps = 30
    size = (512*3, 424*2 + 2*50)
    for i, scan in enumerate(scan_path_list):
        device_list = get_subdirs(scan)
        for device in device_list:
            scan_device_path = os.path.join(scan, device)
            processed_scan_path = scan_device_path.replace(scan_device_path[:scan_device_path.index("ANU_ikea_dataset") + 17], output_path)
            rgb_video_file = os.path.join(processed_scan_path, 'images/scan_video.avi')
            depth_video_file = os.path.join(processed_scan_path, 'depth/scan_video.avi')

            normals_video_file = os.path.join(processed_scan_path, 'normals/scan_video.avi')
            point_clouds_video_file = os.path.join(processed_scan_path, 'point_clouds/point_cloud_video.avi')
            pose2D_video_file = os.path.join(processed_scan_path, '2DposeImages/scan_video.avi')
            video_file_name = os.path.join(processed_scan_path, 'montage.avi')
            if os.path.exists(rgb_video_file) and os.path.exists(depth_video_file) and \
               os.path.exists(normals_video_file) and os.path.exists(point_clouds_video_file) and  \
                os.path.exists(pose2D_video_file):
                rgb_vid = cv2.VideoCapture(rgb_video_file)
                depth_vid = cv2.VideoCapture(depth_video_file)
                normals_vid = cv2.VideoCapture(normals_video_file)
                pc_vid = cv2.VideoCapture(point_clouds_video_file)
                pose2D_vid = cv2.VideoCapture(pose2D_video_file)
                montage_video_writer = cv2.VideoWriter(video_file_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
                while (rgb_vid.isOpened()):
                    rgb_ret, rgb_frame = rgb_vid.read()
                    depth_ret, depth_frame = depth_vid.read()
                    normals_ret, normals_frame = normals_vid.read()
                    normals_frame = cv2.flip(normals_frame, 1)
                    pc_ret, pc_frame = pc_vid.read()
                    pc_frame = cv2.flip(pc_frame, 1)
                    pose2D_ret, pose2D_frame = pose2D_vid.read()

                    if rgb_ret and depth_ret and normals_ret and pc_ret and pose2D_ret:
                        rgb_frame = cv2.resize(rgb_frame, (512, 424))
                        pose2D_frame = cv2.resize(pose2D_frame, (512, 424))
                        pc_frame = cv2.resize(pc_frame, (512, 424))
                        rgb_frame = insert_text_to_image(rgb_frame, 'RGB')
                        depth_frame = insert_text_to_image(depth_frame, 'Depth')
                        normals_frame = insert_text_to_image(normals_frame, 'Normal Vectors')
                        pose2D_frame = insert_text_to_image(pose2D_frame, '2D Pose')
                        pc_frame =insert_text_to_image(pc_frame, '3D Point Cloud')

                        montage_row1 = cv2.hconcat([rgb_frame, depth_frame])
                        montage_row2 = cv2.hconcat([normals_frame, pc_frame, pose2D_frame])
                        side_margin = int((montage_row2.shape[1] - montage_row1.shape[1]) / 2)
                        montage_row1 = cv2.copyMakeBorder(montage_row1, 0, 0, side_margin, side_margin, cv2.BORDER_CONSTANT,
                                                    value=[0, 0, 0])
                        montage_frame = cv2.vconcat([montage_row1, montage_row2])
                        montage_video_writer.write(montage_frame)
                    else:
                        break
                rgb_vid.release()
                depth_vid.release()
                normals_vid.release()
                pc_vid.release()
                pose2D_vid.release()
                montage_video_writer.release()
                print('Saved ' + video_file_name)
            else:
                print('One or more videos required for the montage is not available, please preprocess the dataset first')


def insert_text_to_image(img, txt, font=cv2.FONT_HERSHEY_SIMPLEX, font_size=1):
    # get boundary of this text
    textsize = cv2.getTextSize(txt, font, 1, 2)[0]

    # get coords based on boundary
    textX = int((img.shape[1] - textsize[0]) / 2)

    img = cv2.copyMakeBorder(img, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    cv2.putText(img, txt, (textX, 40), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def export_pose_images(input_path, output_path='', device='dev3', scan_name=None, mode='skeleton', skeleton_type='openpose'):
    """
    Saves images with human pose
    Parameters
    ----------
    input_path : path to ikea dataset
    output_path : path to save the output images
    scan_name : None | scane name to export. if None traverses the entire dataset

    Returns
    -------
        exports all jpg files to output directry
    """

    if output_path == '':
        output_path = input_path

    if not scan_name is None:
        scan_path = os.path.join(input_path, scan_name, device)
        output_path = os.path.join(output_path,  scan_name, device, '2DposeImages')
        os.makedirs(output_path, exist_ok=True)
        rgb_frames = get_files(os.path.join(scan_path, 'images'), file_type='.jpg')
        n_frames = len(rgb_frames)

        num_cores = multiprocessing.cpu_count()
        Parallel(n_jobs=num_cores)(delayed(export_pose_helper)(scan_path, rgb_frames, j, output_path, mode, skeleton_type)
                                   for j in range(n_frames))

        # # debug
        # for j in range(n_frames):
        #     export_pose_helper(scan_path, rgb_frames, j, output_path, mode, skeleton_type=skeleton_type)
    else:
        #TODO implement dataset traversal pose export
        pass


def get_seg_data(input_path, output_path='', device='dev3', scan_name=None):

    color_cat = {1: (255, 0, 0), 2: (0, 0, 255), 3: (0, 255, 0), 4: (127, 0, 127), 5: (127, 64, 0), 6: (64, 0, 127),
                 7: (64, 0, 64)}
    cat_dict = {1: 'table_top', 2: 'leg', 3: 'shelf', 4: 'side_panel', 5: 'front_panel', 6: 'bottom_panel',
                7: 'rear_panel'}


    if output_path == '':
        output_path = input_path

    scan_path = os.path.join(input_path, scan_name, device)
    # output_path = os.path.join(output_path,  scan_name, device, 'seg')
    # os.makedirs(output_path, exist_ok=True)
    rgb_frames = get_files(os.path.join(scan_path, 'images'), file_type='.jpg')
    n_frames = len(rgb_frames)

    scan_name = scan_path.split('/')[-2]
    seg_json_filename = os.path.join(scan_path, 'seg', scan_name + '.json')
    tracking_path = os.path.join(scan_path, 'seg', 'tracklets_interp_' + scan_name + '.txt')

    all_segments, dict_tracks, track_id = get_all_segments_and_tracks(seg_json_filename, tracking_path)
    # Obtain Unique colors for each part
    dict_colors = get_object_segment_color_dict(track_id)

    all_segments_dict = {}

    for item in all_segments['annotations']:
        if item['image_id'] not in all_segments_dict:
            all_segments_dict[item['image_id']] = []
        all_segments_dict[item['image_id']].append(item)

    return rgb_frames, dict_tracks, all_segments, all_segments_dict, dict_colors, color_cat, cat_dict, n_frames


def get_seg_data_v2(input_path, output_path='', device='dev3', scan_name=None):
    # This function works specifically for the training data where psudo ground truth is available and tracking data isnt.

    color_cat = {1: (129, 0, 70), 2: (220, 120, 0), 3: (255, 100, 220), 4: (6, 231, 255), 5: (89, 0, 251), 6: (251, 121, 64),
                 7: (171, 128, 126)}

    cat_dict = {1: 'table_top', 2: 'leg', 3: 'shelf', 4: 'side_panel', 5: 'front_panel', 6: 'bottom_panel',
                7: 'rear_panel'}


    if output_path == '':
        output_path = input_path

    scan_path = os.path.join(input_path, scan_name, device)
    rgb_frames = get_files(os.path.join(scan_path, 'images'), file_type='.jpg')
    n_frames = len(rgb_frames)

    seg_pgt_json_filename = os.path.join(scan_path,  'pseudo_gt_coco_format.json')
    seg_gt_json_filename = os.path.join(scan_path, 'manual_coco_format.json')

    gt_segments = json.load(open(seg_gt_json_filename))
    pgt_segments = json.load(open(seg_pgt_json_filename))
    all_segments = {'images': np.concatenate([gt_segments['images'], pgt_segments['images']]),
                    'annotations': np.concatenate([gt_segments['annotations'], pgt_segments['annotations']]),
                    'categories': gt_segments['categories']}

    all_segments_dict = {}
    for item in gt_segments['annotations']:
        file_name = gt_segments['images'][item['image_id']]['file_name']
        if file_name not in all_segments_dict:
            all_segments_dict[file_name] = []
        all_segments_dict[file_name].append(item)

    for item in pgt_segments['annotations']:
        file_name = pgt_segments['images'][item['image_id']-1]['file_name']  # TODO: remove -1 after indexing is fixed
        if file_name not in all_segments_dict:
            all_segments_dict[file_name] = []
        all_segments_dict[file_name].append(item)

    return rgb_frames, all_segments, all_segments_dict, color_cat, cat_dict, n_frames


def export_seg_images(input_path, output_path='', device='dev3', scan_name=None):
    """
    Saves images with object segmentation
    Parameters
    ----------
    input_path : path to ikea dataset
    output_path : path to save the output images
    scan_name : None | scane name to export. if None traverses the entire dataset

    Returns
    -------
        exports all jpg files to output directry
    """
    if not scan_name is None:
        rgb_frames, dict_tracks, all_segments, all_segments_dict, dict_colors, color_cat, cat_dict, n_frames =\
            get_seg_data(input_path=input_path, output_path=output_path, device=device, scan_name=scan_name)
    else:
        #TODO implement dataset traversal pose export
        pass

    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(export_seg_helper)(rgb_frames, j, output_path, dict_tracks,
                                                          all_segments, all_segments_dict, dict_colors, color_cat,
                                                          cat_dict) for j in range(n_frames))


def get_all_segments_and_tracks(seg_json_filename, tracking_path):
    """
    Load the part segments and tracks from json and txt files
    Parameters
    ----------
    seg_json_filename : path to .json segments file
    tracking_path : path to tracking txt file

    Returns
    -------
    """
    all_segments = json.load(open(seg_json_filename))
    fid_track = open(tracking_path)
    tracking_results = str.split(fid_track.read(), '\n')
    track_id = []
    dict_tracks = {}
    for track in tracking_results:
        if track != "":
            track_id.append(int(str.split(track, ' ')[-3]))
            items = str.split(track, ' ')
            if items[-2] not in dict_tracks:
                dict_tracks[items[-2]] = []
            dict_tracks[items[-2]].append([items[0:5], items[-1]])
    return all_segments, dict_tracks, track_id


def get_object_segment_color_dict(track_id):
    """

    Parameters
    ----------
    track_id :

    Returns
    -------

    """
    dict_colors = {}
    max_part = np.max(np.unique(track_id))
    r = random.sample(range(0, 255), max_part)
    g = random.sample(range(0, 255), max_part)
    b = random.sample(range(0, 255), max_part)
    for part_id in np.unique(track_id):
        dict_colors[str(part_id)] = (int(r[part_id - 1]), int(g[part_id - 1]), int(b[part_id - 1]))
    return dict_colors


def export_pose_helper(scan_path, rgb_frames, file_idx, output_path, mode, skeleton_type='openpose'):
    """
    export pose for a single image - allows parallelization
    Parameters
    ----------
    scan_path :
    rgb_frames :
    file_idx :
    output_path :

    Returns
    -------

    """

    frame_filename = str(file_idx).zfill(6)

    pose_json_filename = os.path.join(scan_path, 'predictions', 'pose2d', skeleton_type,
                                      'scan_video_' + str(file_idx).zfill(12) + '_keypoints.json')
    output_filename = os.path.join(output_path, frame_filename + '.jpg')
    img = cv2.imread(rgb_frames[file_idx])
    if mode == 'skeleton':
        img = img_pose_skeleton_overlay(img, pose_json_filename, skeleton_type=skeleton_type )
    else:
        img = img_pose_mesh_overlay(img, pose_json_filename)
    cv2.imwrite(output_filename, img)
    print('Saved pose for ' + frame_filename + '.jpeg to ' + output_path)


def export_seg_helper(rgb_frames, file_idx, output_path, dict_tracks, all_segments, all_segments_dict,
                      dict_colors, color_cat, cat_dict):
    """
    export object segmentation for a single image - allows parallelization
    Parameters
    ----------
    scan_path :
    rgb_frames :
    file_idx :
    output_path :

    Returns
    -------

    """
    frame_filename = str(file_idx).zfill(6)
    image_id = find_seg_id(frame_filename, all_segments)
    fname_id = int(str.split(frame_filename, '.')[0])
    segment = all_segments_dict[image_id]
    track = dict_tracks[str(fname_id)]

    output_filename = os.path.join(output_path, frame_filename + '.jpg')
    img = cv2.imread(rgb_frames[file_idx])
    img = img_seg_overlay(img, segment, track, dict_colors, color_cat, cat_dict)
    cv2.imwrite(output_filename, img)
    print('Saved object segmentation for ' + frame_filename + '.jpeg to ' + output_path)

def find_seg_id(image_name, test_data):
    for item in test_data['images']:
        if item['file_name'].find(image_name) != -1:
            return item['id']
    return -1


def find_seg_id_v2(image_name, test_data):
    for img in test_data.keys():
        if image_name in img:
            return img
    return -1


def img_seg_overlay(image, predictions, part_tracks, dict_colors, color_cat, cat_dict):
    """
    overlays object segmentation from json file on the given image
    Parameters
    ----------
    img : rgb image
    json_path : path to .json file

    Returns
    -------
    img : rgb img with object segments overlay
    """
    for part in part_tracks:
        assigned = 0
        for item in predictions:
            box = item['bbox']
            label = item['category_id']
            segment = item['segmentation']
            segment_id = item['id']
            contours = []
            length = len(segment)
            if segment_id == int(part[1]):
                for i in range(length):
                    id = 0
                    contour = segment[i]
                    cnt = len(contour)
                    c = np.zeros((int(cnt / 2), 1, 2), dtype=np.int32)
                    for j in range(0, cnt, 2):
                        c[id, 0, 0] = contour[j]
                        c[id, 0, 1] = contour[j + 1]
                        id = id + 1
                    contours.append(c)
                cv2.drawContours(image, contours, -1, color_cat[label], -1)
                x1, y1 = box[:2]
                cv2.putText(image, cat_dict[label], (int(x1) - 10, int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1)
                rgb = dict_colors[part[0][-1]]
                assigned = 1
                image = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), rgb, 3)

        if assigned == 0:
            rgb = dict_colors[part[0][-1]]
            image = cv2.rectangle(image, (int(float(part[0][0])), int(float(part[0][1]))), (int(float(part[0][0]) + float(part[0][2])), int(float(part[0][1]) + float(part[0][3]))), rgb, 3)

    return image


def img_seg_overlay_v2(image, predictions, color_cat, cat_dict, show_text=False):
    """
    overlays object segmentation from json file on the given image
    Parameters
    ----------
    img : rgb image
    json_path : path to .json file

    Returns
    -------
    img : rgb img with object segments overlay
    """

    for part in predictions:
        contours = []
        length = len(part['segmentation'])
        bbox = part['bbox']
        for i in range(length):
            id = 0
            contour = part['segmentation'][i]
            cnt = len(contour)
            c = np.zeros((int(cnt / 2), 1, 2), dtype=np.int32)
            for j in range(0, cnt, 2):
                c[id, 0, 0] = contour[j]
                c[id, 0, 1] = contour[j + 1]
                id = id + 1
            if c.shape[0] != 0:
                contours.append(c)
        color = color_cat[part['category_id']]
        cv2.drawContours(image, contours, -1, (color[0], color[1], color[2]), -1)

        # if 'part_id' in part:
        if show_text:
            cv2.putText(image, cat_dict[part['category_id']],
                        (int(bbox[0] + bbox[2] // 2), int(bbox[1] + bbox[3] // 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),
                              (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 0, 0),2)

    return image


def img_pose_skeleton_overlay(img, pose_json_filename, show_numbers=False, anonimyze=False, skeleton_type='openpose'):
    """
    overlays pose from json file on the given image
    Parameters
    ----------
    img : rgb image
    json_path : path to .json file

    Returns
    -------
    img : rgb img with pose overlay
    """

    j2d = read_pose_json(pose_json_filename)

    if skeleton_type == 'openpose':
        skeleton_pairs = joint_ids.get_body25_connectivity()
        skeleton_pairs = skeleton_pairs[0:19]
    else:
        skeleton_pairs = joint_ids.get_ikea_connectivity()
    part_colors = joint_ids.get_pose_colors(mode='bgr')

    if anonimyze:
        # anonimize the img by plotting a black circle centered on the nose
        nose = tuple(int(element) for element in j2d[0])
        radius = 45
        img = cv2.circle(img, nose, radius, (0, 0, 0), -1)

    # plot the joints
    bad_points_idx = []
    for i, point in enumerate(j2d):
        if not point[0] == 0 and not point[1] == 0:
            cv2.circle(img, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        else:
            bad_points_idx.append(i)

    # plot the skeleton
    for i, pair in enumerate(skeleton_pairs):
        partA = pair[0]
        partB = pair[1]
        if partA not in bad_points_idx and partB not in bad_points_idx:
            # if j2d[partA] and j2d[partB]:
            line_color = part_colors[i]
            img = cv2.line(img, tuple([int(el) for el in j2d[partA]]), tuple([int(el) for el in j2d[partB]]),
                           line_color, 3)
    if show_numbers:
        # add numbers to the joints
        for i, point in enumerate(j2d):
            if i not in bad_points_idx:
                cv2.putText(img, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2,
                            lineType=cv2.LINE_AA)


    return img


def img_pose_mesh_overlay(img, pose_json_filename):
    """
    overlays pose mesh (human SMPL model) from json file on the given image
    Parameters
    ----------
    img : rgb image
    json_path : path to .json file

    Returns
    -------
    img : rgb img with pose overlay
    """
    data = read_pose_json(pose_json_filename)
    vertices = data[0]['verts']
    betas = data[0]['betas']
    cam = data[0]['orig_cam']
    mesh_color = [1.0, 1.0, 0.9]
    renderer = Renderer(resolution=(img.shape[1], img.shape[0]), orig_img=True, wireframe=False)
    img = renderer.render(img, vertices, cam=cam, color=mesh_color, mesh_filename=None)
    return img


def read_pose_json(json_path):
    """
    Parameters
    ----------
    json_path : path to json file

    Returns
    -------
    data: a list of dictionaries containing the pose information per video frame
    """
    with open(json_path) as json_file:
        json_data = json.load(json_file)
    data = json_data['people']
    if len(data) > 1:
        data = get_active_person(data)
    else:
        data = np.array(data[0]['pose_keypoints_2d'])  # x,y,confidence

    data = np.reshape(data, (-1, 3))
    data = data[:, :2]
    return data


def get_active_person(people, center=(960, 540), min_bbox_area=20000):
    """
    Select the active skeleton in the scene by applying a heuristic of findng the closest one to the center of the frame
    then take it only if its bounding box is large enough - eliminates small bbox like kids
    Assumes 100 * 200 minimum size of bounding box to consider
    Parameters
    ----------
    data : pose data extracted from json file
    center: center of image (x, y)
    min_bbox_area: minimal bounding box area threshold

    Returns
    -------
    pose: skeleton of the active person in the scene (flattened)
    """

    pose = None
    min_dtc = float('inf')  # dtc = distance to center
    for person in people:
        current_pose = person['pose_keypoints_2d']
        joints_2d = np.reshape(current_pose, (-1, 3))[:, :2]
        if 'boxes' in person.keys():
            #maskrcnn
            bbox = person['boxes']
        else:
            # openpose
            idx = np.where(joints_2d.any(axis=1))[0]
            bbox = [np.min(joints_2d[idx, 0]),
                    np.min(joints_2d[idx, 1]),
                    np.max(joints_2d[idx, 0]),
                    np.max(joints_2d[idx, 1])]


        A = (bbox[2] - bbox[0] ) * (bbox[3] - bbox[1]) #bbox area
        bbox_center = (bbox[0] + (bbox[2] - bbox[0])/2, bbox[1] + (bbox[3] - bbox[1])/2) #bbox center
        # joints_2d = np.reshape(current_pose, (-1, 3))
        # dtc = compute_skeleton_distance_to_center(joints_2d[:, :2], center=center)
        dtc = np.sqrt(np.sum((np.array(bbox_center) - np.array(center))**2))
        if dtc < min_dtc :
            closest_pose = current_pose
            if A > min_bbox_area:
                pose = closest_pose
                min_dtc = dtc
    # if all bboxes are smaller than threshold, take the closest
    if pose is None:
        pose = closest_pose
    return pose


def compute_skeleton_distance_to_center(skeleton, center=(960, 540)):
    """
    Compute the average distance between a given skeleton and the cetner of the image
    Parameters
    ----------
    skeleton : 2d skeleton joint poistiions
    center : image center point

    Returns
    -------
        distance: the average distance of all non-zero joints to the center
    """
    idx = np.where(skeleton.any(axis=1))[0]
    diff = skeleton - np.tile(center, [len(skeleton[idx]), 1])
    distances = np.sqrt(np.sum(diff ** 2, 1))
    mean_distance = np.mean(distances)

    return mean_distance


def img_action_overlay(img, action_label):
    """
    overlay the label text over the image

    Parameters
    ----------
    img : image
    action_label : label string

    Returns
    -------
    img: updated image
    """
    display_text = "Action: " + action_label
    img_h, img_w, _ = img.shape
    font_scale = 3
    font_thickness = 6
    (text_width, text_height) = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
                                                thickness=font_thickness)[0]
    (text_width, text_height) = (int(text_width), int(1.4*text_height))
    text_offset_x, text_offset_y = 0, img_h - int(0.4*text_height)
    # box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width, text_offset_y - text_height))
    box_coords = ((text_offset_x, img_h), (text_offset_x + text_width, text_offset_y - text_height))
    # img = cv2.resize(color_frame, (int(img_w / 2), int(img_h / 2)))
    cv2.rectangle(img, box_coords[0], box_coords[1], (0, 0, 0, 0.2), cv2.FILLED)
    img = cv2.putText(img, display_text, org=(text_offset_x, text_offset_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
                            color=(255, 255, 255), thickness=font_thickness, bottomLeftOrigin=False)
    return img


def export_perception_demo_images(input_path, output_path='', device='dev3', scan_name=None,
                                  show_numbers=False, mode='skeleton', anonimyze=False, version=2,
                                  skeleton_type='openpose'):
    """
    exports images in scan with actions, pose and object segmentation overlay
    this function specifically works for the training set where the pseudo ground truth segments are
    avialable and no tracking data
    Parameters
    ----------
    input_path : path to ikea dataset
    output_path : path to output directory
    device : camera name to use (dev3)
    scan_name : name of scan to export
    set: train/test to manage available data
    Returns
    -------

    """


    output_path = os.path.join(output_path,  scan_name, device, 'perception_demo')
    os.makedirs(output_path, exist_ok=True)
    rgb_frames, all_segments, all_segments_dict, dict_colors, cat_dict, n_frames = \
        get_seg_data_v2(input_path=input_path, output_path=output_path, device=device, scan_name=scan_name)
    scan_path = os.path.join(input_path, scan_name, device)

    # load action info
    gt_json_path = os.path.join(input_path, 'gt_segments.json')

    dataset = Dataset(input_path, action_segments_filename=gt_json_path)
    all_gt_labels = dataset.action_labels
    scan_index = dataset.all_video_list.index(scan_name)
    gt_labels = np.argmax(all_gt_labels[scan_index], axis=1)
    action_list = dataset.action_list

    for file_idx in range(n_frames):
        img = cv2.imread(rgb_frames[file_idx])
        frame_filename = str(file_idx).zfill(6)
        output_filename = os.path.join(output_path, frame_filename + '.jpg')

        # load object segmentation data
        image_id = int(frame_filename)
        image_name = find_seg_id_v2(frame_filename, all_segments_dict)
        # fname_id = int(str.split(frame_filename, '.')[0])
        segment = all_segments_dict[image_name]

        # load pose data

        pose_json_filename = os.path.join(scan_path, 'predictions', 'pose2d', skeleton_type,
                                              'scan_video_' + str(image_id).zfill(12)+'_keypoints.json')

        # load label data
        action_label = action_list[gt_labels[file_idx]]
        # Draw
        if mode == 'skeleton':
            img = img_pose_skeleton_overlay(img, pose_json_filename, show_numbers, anonimyze=anonimyze,
                                            version=version, skeleton_type=skeleton_type )
        else:
            img = img_pose_mesh_overlay(img, pose_json_filename)

        img = img_seg_overlay_v2(img, segment, dict_colors, cat_dict)

        img = img_action_overlay(img, action_label)
        cv2.imwrite(output_filename, img)
        print('Saved perception demo image ' + rgb_frames[file_idx] + ' to ' + output_filename)

# def get_smpl_faces(smpl_data_dir='./smpl_data/models/', model_type='smplx', gender='neutral',ext='npz'):
#
#     model = smplx.create(smpl_data_dir, model_type=model_type, gender=gender, use_face_contour=False, ext=ext)
#     return model.faces



def export_perception_demo_images_no_seg(input_path, output_path='', device='dev3', scan_name=None,
                                  show_numbers=False, mode='skeleton', anonimyze=False, skeleton_type='openpose'):
    """
    exports images in scan with actions, pose and object segmentation overlay

    Parameters
    ----------
    input_path : path to ikea dataset
    output_path : path to output directory
    device : camera name to use (dev3)
    scan_name : name of scan to export

    Returns
    -------

    """

    output_path = os.path.join(output_path,  scan_name, device, 'perception_demo')
    rgb_frames, dict_tracks, all_segments, all_segments_dict, dict_colors, color_cat, cat_dict, n_frames = \
        get_seg_data(input_path=input_path, output_path=output_path, device=device, scan_name=scan_name)
    scan_path = os.path.join(input_path, scan_name, device)

    # load action info
    gt_json_path = os.path.join(input_path, 'gt_segments.json')

    dataset = Dataset(input_path, action_segments_filename=gt_json_path)
    all_gt_labels = dataset.action_labels
    scan_index = dataset.all_video_list.index(scan_name)
    gt_labels = np.argmax(all_gt_labels[scan_index], axis=1)
    action_list = dataset.action_list

    for file_idx in range(n_frames):
        img = cv2.imread(rgb_frames[file_idx])
        frame_filename = str(file_idx).zfill(6)
        output_filename = os.path.join(output_path, frame_filename + '.jpg')

        # load object segmentation data
        image_id = find_seg_id(frame_filename, all_segments)
        fname_id = int(str.split(frame_filename, '.')[0])
        segment = all_segments_dict[image_id]
        track = dict_tracks[str(fname_id)]

        # load pose data
        pose_json_filename = os.path.join(scan_path, 'predictions', 'pose2d', skeleton_type,
                     'scan_video_' + str(file_idx).zfill(12) + '_keypoints.json')

        # load label data
        action_label = action_list[gt_labels[file_idx]]
        # Draw
        if mode == 'skeleton':
            img = img_pose_skeleton_overlay(img, pose_json_filename, show_numbers, anonimyze=anonimyze,
                                            skeleton_type=skeleton_type )
        else:
            img = img_pose_mesh_overlay(img, pose_json_filename)
        img = img_seg_overlay(img, segment, track, dict_colors, color_cat, cat_dict)
        img = img_action_overlay(img, action_label)
        cv2.imwrite(output_filename, img)
        print('Saved perception demo image ' + rgb_frames[file_idx] + 'to ' + output_filename)



class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class Renderer:
    def __init__(self, resolution=(224, 224), orig_img=False, wireframe=False):
        self.resolution = resolution

        self.faces = get_smpl_faces()
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

    def render(self, img, verts, cam, angle=None, axis=None, mesh_filename=None, color=[1.0, 1.0, 0.9]):

        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)

        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)

        if mesh_filename is not None:
            mesh.export(mesh_filename)

        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)

        sx, sy, tx, ty = cam

        camera = WeakPerspectiveCamera(
            scale=[sx, sy],
            translation=[tx, ty],
            zfar=1000.
        )

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
        image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return image


def extract_frames(scan, dataset_path, out_path, fps=25, video_format='avi'):
    """
    Extract individual frames from a scan video
    Parameters
    ----------
    scan : path to a single video scan (.avi file)
    out_path : output path
    fps : frames per second

    Returns
    -------

    """
    out_dir = scan.replace(dataset_path, out_path)
    os.makedirs(out_dir, exist_ok=True)
    scan_video_filename = os.path.join(scan, 'scan_video.' + video_format)

    print('Extracting frames from: ' + scan_video_filename)

    cap = cv2.VideoCapture(scan_video_filename)
    i = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if 'depth' not in scan:
            cv2.imwrite(os.path.join(out_dir,  str(i).zfill(6) + '.jpg'), frame)
        else:
            frame = get_absolute_depth(frame)
            cv2.imwrite(os.path.join(out_dir, str(i).zfill(6) + '.png'), frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()
    print('Done')