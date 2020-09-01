import os
import shutil
import cv2
import vis_utils
import numpy as np
import json
import random
import sys
sys.path.append('../')
import utils

#TODO implement adjust depth to rgb image size

class FunctionSwitcher:

    def __init__(self, scan_path, file_idx, output_path, name_mapping_dict,
                 modality_save_function={'dev3': 'rgb', 'dev2': 'rgb', 'dev1': 'rgb', 'depth': 'depth',
                                         'seg': 'seg', 'pose': 'pose'}, resize_factor=None):
        self.scan_path = scan_path
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        self.file_idx = file_idx
        self.modality_save_function = modality_save_function
        self.name_mapping_dict = name_mapping_dict
        self.resize_factor = resize_factor

    def switch(self, modality):
        self.modality = modality
        getattr(self, self.modality_save_function[self.modality], lambda: print(" Invalid modality"))()

    def rgb(self):
        filename = os.path.join(self.scan_path, self.modality, 'images', str(self.file_idx).zfill(6) + '.jpg')
        output_filename = os.path.join(self.output_path, self.name_mapping_dict[self.modality] + '.jpg')
        if self.resize_factor is None:
            shutil.copyfile(filename, output_filename)
        else:
            img = cv2.imread(filename)
            h, w, c = img.shape
            h, w = (int(h / self.resize_factor), int(w / self.resize_factor))
            img = cv2.resize(img, dsize=(w, h))  # resizing the images
            cv2.imwrite(output_filename, img)
        print(" Saved {} to {}".format(filename, output_filename))


    def depth(self):
        filename = os.path.join(self.scan_path, 'dev3', self.modality, str(self.file_idx).zfill(6) + '.png')
        output_filename = os.path.join(self.output_path, self.name_mapping_dict[self.modality] + '.png')
        img = cv2.imread(filename, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        if not self.resize_factor is None:
            h, w = img.shape
            h, w = (int(2*h / self.resize_factor), int(2*w / self.resize_factor))
            img = cv2.resize(img, dsize=(w, h))  # resizing the images
        img = vis_utils.get_relative_depth(img)
        cv2.imwrite(output_filename, img)
        print(" Saved {} to {}".format(filename, output_filename))


    def pose(self):
        pose_json_filename = os.path.join(self.scan_path, 'dev3', self.modality, str(self.file_idx).zfill(6) + '.json')
        image_path = os.path.join(self.scan_path, 'dev3', 'images', str(self.file_idx).zfill(6) + '.jpg')
        output_filename = os.path.join(self.output_path, self.name_mapping_dict[self.modality] + '.jpg')
        data = utils.read_pose_json(pose_json_filename)
        img = cv2.imread(image_path)

        j2d = data[0]["joints2d"][:, :2]

        if len(data) > 1:
            j2d = self.get_active_person(data)
        else:
            j2d = np.array(data['people'][0]['pose_keypoints_2d'])  # x,y,confidence

        skeleton_pairs = utils.get_staf_skeleton()
        part_colors = utils.get_pose_colors(mode='bgr')

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

        # # add numbers to the joints
        # for i, point in enumerate(j2d):
        #     if i not in bad_points_idx:
        #         cv2.putText(img, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        #                     (0, 0, 255), 2,
        #                     lineType=cv2.LINE_AA)
        cv2.imwrite(output_filename, img)
        print(" Saved pose visualization to {}".format(output_filename))

    def seg(self):
        scan_name = self.scan_path.split('/')[-1]
        segment_path = os.path.join(self.scan_path, 'dev3', 'seg', scan_name + '.json')
        tracking_path = os.path.join(self.scan_path, 'dev3', 'seg', 'tracklets_interp_' + scan_name + '.txt')
        color_cat = {1: (255, 0, 0), 2: (0, 0, 255), 3: (0, 255, 0), 4: (127, 0, 127), 5: (127, 64, 0), 6: (64, 0, 127),
                     7: (64, 0, 64)}
        cat_dict = {1: 'table_top', 2: 'leg', 3: 'shelf', 4: 'side_panel', 5: 'front_panel', 6: 'bottom_panel',
                    7: 'rear_panel'}
        filename = os.path.join(self.scan_path, 'dev3', 'images', str(self.file_idx).zfill(6) + '.jpg')
        output_filename = os.path.join(self.output_path, self.name_mapping_dict[self.modality] + '.jpg')
        img = cv2.imread(filename)

        all_segments = json.load(open(segment_path))
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

        # Obtain Unique colors for each part
        dict_colors = {}
        max_part = np.max(np.unique(track_id))
        r = random.sample(range(0, 255), max_part)
        g = random.sample(range(0, 255), max_part)
        b = random.sample(range(0, 255), max_part)
        for part_id in np.unique(track_id):
            dict_colors[str(part_id)] = (int(r[part_id - 1]), int(g[part_id - 1]), int(b[part_id - 1]))

        all_segments_dict = {}

        for item in all_segments['annotations']:
            if item['image_id'] not in all_segments_dict:
                all_segments_dict[item['image_id']] = []
            all_segments_dict[item['image_id']].append(item)

        fname = str(self.file_idx).zfill(6) + '.jpg'
        image_id = self.find_id(fname, all_segments)
        fname_id = int(str.split(fname, '.')[0])

        predictions = vis_utils.overlay_segmentation_mask(img, all_segments_dict[image_id], dict_tracks[str(fname_id)],
                                                     dict_colors, color_cat, cat_dict)

        if not self.resize_factor is None:
            h, w, c = predictions.shape
            h, w = (int(h / self.resize_factor), int(w / self.resize_factor))
            predictions = cv2.resize(predictions, dsize=(w, h))  # resizing the images

        cv2.imwrite(output_filename, predictions)
        print(" Saved object segmentation to {}".format( output_filename))

    def find_id(self, image_name, test_data):
        for item in test_data['images']:
            if item['file_name'].find(image_name) != -1:
                return item['id']
        return -1

    def get_active_person(self, data, center=(960, 540)):
        """
        Select the active skeleton in the scene by applying a heuristic of findng the one closest to the center of the frame
        Parameters
        ----------
        data : pose data extracted from json file

        Returns
        -------
        pose: skeleton of the active person in the scene
        """
        pose = None
        min_dtc = 0 # dtc = distance to center
        for person in data['people']:
            current_pose = person['pose_keypoints_2d']
            dtc = self.compute_skeleton_distance_to_center(current_pose, center=center)
            if dtc < min_dtc:
                pose = current_pose
                min_dtc = dtc
        return pose


    def compute_skeleton_distance_to_center(self, skeleton, center=(960, 540)):
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
        diff = skeleton - np.tile(center, len(skeleton[idx]))
        distances = np.sqrt(np.mean(diff ** 2))
        mean_distance = np.mean(distances)

        return mean_distance



output_dir = './example_modalities/'
scan_name = '/mnt/sitzikbs_storage/Datasets/ANU_ikea_dataset/Lack_Side_Table/0007_oak_floor_01_01_2019_08_14_17_17'
image_idx = 2874 # 312
name_mapping_dict = {'dev3': 'top_view', 'dev2': 'front_view', 'dev1': 'side_view', 'depth': 'depth',
                     'seg': 'seg', 'pose': 'pose'}
resize_factor = None

switcher = FunctionSwitcher(scan_name, image_idx, output_dir, name_mapping_dict, resize_factor=resize_factor)
for modality, new_name in name_mapping_dict.items():
    switcher.switch(modality)
