import sqlite3
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import torch
import json


class IKEAActionDataset():
    """
    IKEA Action Dataset class
    """

    def __init__(self, dataset_path, db_filename='ikea_annotation_db_full', action_list_filename='atomic_action_list.txt',
                 action_object_relation_filename='action_object_relation_list.txt',train_filename='train_cross_env.txt',
                 test_filename='test_cross_env.txt', action_segments_filename=None):
        self.dataset_path = dataset_path
        self.db_file = os.path.join(dataset_path, db_filename)
        self.db = sqlite3.connect(self.db_file)
        self.db.row_factory = sqlite3.Row

        self.cursor_vid = self.db.cursor()
        self.cursor_annotations = self.db.cursor()

        # indexing files
        self.action_list_filename = os.path.join(dataset_path, 'indexing_files', action_list_filename)
        self.action_object_relation_filename = os.path.join(dataset_path, 'indexing_files', action_object_relation_filename)

        self.train_filename = os.path.join(dataset_path, 'indexing_files', train_filename)
        self.test_filename = os.path.join(dataset_path, 'indexing_files', test_filename)

        # load action list and object relation list and sort them together
        self.action_list = self.get_list_from_file(self.action_list_filename)
        self.ao_list = self.get_action_object_relation_list()

        self.ao_list = [x for _, x in sorted(zip(self.action_list, self.ao_list))]
        self.action_list.sort()
        self.action_list.insert(0, "NA")  #  0 label for unlabled frames

        self.num_classes = len(self.action_list)
        self.trainset_video_list = self.get_list_from_file(self.train_filename)
        self.testset_video_list = self.get_list_from_file(self.test_filename)
        self.all_video_list = self.testset_video_list + self.trainset_video_list

        # GT files
        if action_segments_filename is not None:
            self.segment_json_filename = action_segments_filename
            self.action_labels = self.get_actions_labels_from_json(self.segment_json_filename, mode='gt')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.db.close()

    def close(self):
        self.db.close()

    def get_list_from_file(self, filename):
        """
        retrieve a list of lines from a .txt file
        :param :
        :return: list of atomic actions
        """
        with open(filename) as f:
            line_list = f.read().splitlines()
        # line_list.sort()
        return line_list

    def get_action_object_relation_list(self):
        # read the action-object relation list file and return a list of integers
        with open(self.action_object_relation_filename, 'r') as file:
            a_o_relation_list = file.readlines()
            a_o_relation_list = [x.rstrip('\n').split() for x in a_o_relation_list]
            a_o_relation_list = [[int(y) for y in x] for x in a_o_relation_list]
        return a_o_relation_list

    def get_annotated_videos_table(self, device='all'):
        """
        fetch the annotated videos table from the database
        :param : device: string ['all', 'dev1', 'dev2', 'dev3']
        :return: annotated videos table
        """
        if device == 'all':
            return_table = self.cursor_vid.execute('''SELECT * FROM videos WHERE annotated = 1''')
        else:
            return_table = self.cursor_vid.execute('''SELECT * FROM videos WHERE annotated = 1 AND camera = ?''',
                                                   (device,))
        return return_table

    def get_video_annotations_table(self, video_idx):
        """
        fetch the annotation table of a specific video
        :param :  video_idx: index of the desired video
        :return: video annotations table
        """
        return self.cursor_annotations.execute('''SELECT * FROM annotations WHERE video_id = ?''', (video_idx,))

    def get_video_table(self, video_idx):
        """
        fetch the video information row from the video table in the database
        :param :  video_idx: index of the desired video
        :return: video information table row from the databse
        """
        return self.cursor_annotations.execute('''SELECT * FROM videos WHERE id = ?''', (video_idx,))

    def get_annotation_table(self):
        """
        :return: full annotations table (for all videos)
        """
        return self.cursor_annotations.execute('''SELECT * FROM annotations ''')

    def get_action_id(self, atomic_action_id, object_id):
        """
        find the action id of an atomic action-object pair, returns None if not in the set
        :param action_id: int id of atomic action
        :param object_id: int id of object
        :return: int compound action id | None if not in set
        """
        idx = None
        for i, ao_pair in enumerate(self.ao_list):
            if ao_pair[0] == atomic_action_id and ao_pair[1] == object_id:
                idx = i + 1  # +1 to allow the NA first action label
                break
        return idx

    def get_video_id_from_video_path(self, video_path, device='dev3'):
        """
        return video id for a given video path
        :param video_path: path to video (including device and image dir)
        :return: video_id: id of the video
        """
        rows = self.cursor_vid.execute('''SELECT * FROM videos WHERE video_path = ? AND camera = ?''',
                                (video_path, device)).fetchall()
        if len(rows) > 1:
            raise ValueError("more than one video with the desired specs - check database")
        else:
            output = rows[0]["id"]
        return output

    def get_video_name_from_id(self, video_id):
        """
        return video name for a given video id
        :param video_id: id of video

        :return: video_name: name of the video
        """
        rows = self.cursor_vid.execute('''SELECT * FROM videos WHERE id = ?''',
                                (video_id, )).fetchall()
        if len(rows) > 1:
            raise ValueError("more than one video with the desired specs - check database")
        else:
            # output = os.path.join(rows[0]["furniture"], rows[0]["video_name"])
            output = rows[0]["video_path"].split('dev', 1)[0][:-1]
        return output

    def squeeze_action_names(self, thresh=3):
        """
        shortens the class names for visualization

        :param class_names: class name
        :return: class_names: shortened class names
        """
        class_names = self.action_list
        class_names = [substring.split(" ") for substring in class_names]
        for i, cls in enumerate(class_names):
            if len(cls) <= thresh:
                class_names[i] = " ".join(cls)
            else:
                class_names[i] = " ".join(cls[0:thresh]) + "..."
        return class_names

    def get_actions_labels_from_json(self, json_filename, device='dev3', mode='gt'):
        """

         Loads a label segment .json file (ActivityNet format
          http://activity-net.org/challenges/2020/tasks/anet_localization.html) and converts to frame labels for evaluation

        Parameters
        ----------
        json_filename : output .json file name (full path)
        device: camera view to use
        Returns
        -------
        frame_labels: one_hot frame labels (allows multi-label)
        """
        labels = []
        with open(json_filename, 'r') as json_file:
            json_dict = json.load(json_file)

        if mode == 'gt':
            video_results = json_dict["database"]
        else:
            video_results = json_dict["results"]
        for scan_path in video_results:
            video_path = os.path.join(scan_path, device, 'images')
            video_idx = self.get_video_id_from_video_path(video_path, device=device)
            video_info = self.get_video_table(video_idx).fetchone()
            n_frames = video_info["nframes"]
            current_labels = np.zeros([n_frames, self.num_classes])
            if mode == 'gt':
                segments = video_results[scan_path]['annotation']
            else:
                segments = video_results[scan_path]
            for segment in segments:
                action_idx = self.action_list.index(segment["label"])
                start = segment['segment'][0]
                end = segment['segment'][1]
                current_labels[start:end, action_idx] = 1
            labels.append(current_labels)
        return labels


# This dataset loads videos (not single frames) and cuts them into clips
class IKEAActionVideoClipDataset(IKEAActionDataset):
    def __init__(self, dataset_path, db_filename='ikea_annotation_db_full', action_list_filename='atomic_action_list.txt',
                 action_object_relation_filename='action_object_relation_list.txt', train_filename='train_cross_env.txt',
                 test_filename='test_cross_env.txt', transform=None, set='test', camera='dev3', frame_skip=1,
                 frames_per_clip=64, resize=None, mode='vid', input_type='rgb'):
        super().__init__(dataset_path=dataset_path, db_filename=db_filename, action_list_filename=action_list_filename,
                 action_object_relation_filename=action_object_relation_filename, train_filename=train_filename,
                         test_filename=test_filename)
        self.mode = mode
        self.transform = transform
        self.set = set
        self.camera = camera
        self.frame_skip = frame_skip
        self.frames_per_clip = frames_per_clip
        self.resize = resize
        self.input_type = input_type
        if self.set == 'train':
            self.video_list = self.trainset_video_list
        elif self.set == 'test':
            self.video_list = self.testset_video_list
        else:
            raise ValueError("Invalid set name")

        self.video_set = self.get_video_frame_labels()
        self.clip_set, self.clip_label_count = self.get_clips()


    def get_video_frame_labels(self):
        # Extract the label data from the database
        # outputs a dataset structure of (video_path, multi-label per-frame, number of frames in the video)
        video_table = self.get_annotated_videos_table(device=self.camera)
        vid_list = []
        for row in video_table:
            n_frames = int(row["nframes"])
            video_path = row['video_path']
            if self.input_type == 'depth':
                video_path = video_path.replace('images', 'depth')
            video_name = os.path.join(video_path.split('/')[0], video_path.split('/')[1])
            if self.mode == 'vid':
                video_full_path = os.path.join(self.dataset_path, video_path, 'scan_video.avi')
            else:
                video_full_path = os.path.join(self.dataset_path, video_path)
            if not video_name in self.video_list:
                continue
            if n_frames < 66 * self.frame_skip:  # check video length
                continue
            if not os.path.exists(video_full_path):  # check if frame folder exists
                continue

            label = np.zeros((self.num_classes, n_frames), np.float32) # allow multi-class representation
            label[0, :] = np.ones((1, n_frames), np.float32)   # initialize all frames as background|transition
            video_id = row['id']
            annotation_table = self.get_video_annotations_table(video_id)
            for ann_row in annotation_table:
                atomic_action_id = ann_row["atomic_action_id"]  # map the labels
                object_id = ann_row["object_id"]
                action_id = self.get_action_id(atomic_action_id, object_id)
                end_frame = ann_row['ending_frame'] if ann_row['ending_frame'] < n_frames else n_frames
                if action_id is not None:
                    label[:, ann_row['starting_frame']:end_frame] = 0  # remove the background label
                    label[action_id, ann_row['starting_frame']:end_frame] = 1

            vid_list.append(
                (video_full_path, label, n_frames))  # 0 = duration - irrelevant for initial tests, used for start
        return vid_list

    def get_clips(self):
        # extract equal length video clip segments from the full video dataset
        clip_dataset = []
        label_count = np.zeros(self.num_classes)
        for i, data in enumerate(self.video_set):
            n_frames = data[2]
            n_clips = int(n_frames / (self.frames_per_clip * self.frame_skip))
            remaining_frames = n_frames % (self.frames_per_clip * self.frame_skip)
            for j in range(0, n_clips):
                for k in range(0, self.frame_skip):
                    start = j * self.frames_per_clip * self.frame_skip + k
                    end = (j + 1) * self.frames_per_clip * self.frame_skip
                    label = data[1][:, start:end:self.frame_skip]
                    label_count = label_count + np.sum(label, axis=1)
                    frame_ind = np.arange(start, end, self.frame_skip).tolist()
                    clip_dataset.append((data[0], label, frame_ind, self.frames_per_clip, i, 0))
            if not remaining_frames == 0:
                frame_pad =  self.frames_per_clip - remaining_frames
                start = n_clips * self.frames_per_clip * self.frame_skip + k
                end = start + remaining_frames
                label = data[1][:, start:end:self.frame_skip]
                label_count = label_count + np.sum(label, axis=1)
                label = data[1][:, start-frame_pad:end:self.frame_skip]
                frame_ind = np.arange(start-frame_pad, end, self.frame_skip).tolist()
                clip_dataset.append((data[0], label, frame_ind, self.frames_per_clip, i, frame_pad))
        return clip_dataset, label_count

    def load_rgb_frames(self, video_full_path, frame_ind):
        # load video file and extract the frames
        frames = []
            # Open the video file
        if self.mode == 'vid':
            cap = cv2.VideoCapture(video_full_path)
        for i in frame_ind:
            if self.mode == 'vid':  # load from video file
                cap.set(1, i)
                ret, img = cap.read()
            else:  # load from image folder
                if self.input_type == 'rgb':
                    img_filename = os.path.join(video_full_path, str(i).zfill(6) + '.jpg')
                else:
                    img_filename = os.path.join(video_full_path, str(i).zfill(6) + '.png')
                img = cv2.imread(img_filename)
                    # img = cv2.imread(img_filename, cv2.IMREAD_ANYDEPTH).astype(np.float32)
            try:
                # if self.input_type == 'rgb':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # else:
                #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            except:
                # debugging
                raise ValueError("error occured while loading frame {} from video {}".format(i, video_full_path))

            if self.resize is not None:
                w, h, c = img.shape
                if w < self.resize or h < self.resize:
                    d = self.resize - min(w, h)
                    sc = 1 + d / min(w, h)
                    img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
                img = cv2.resize(img, dsize=(self.resize, self.resize))  # resizing the images
                img = (img / 255.) * 2 - 1
            frames.append(img)
        if self.mode == 'vid':
            cap.release()
        return np.asarray(frames, dtype=np.float32)

    def video_to_tensor(self, pic):
        """Convert a ``numpy.ndarray`` to tensor.
        Converts a numpy.ndarray (T x H x W x C)
        to a torch.FloatTensor of shape (C x T x H x W)

        Args:
             pic (numpy.ndarray): Video to be converted to tensor.
        Returns:
             Tensor: Converted video.
        """
        return torch.tensor(pic.transpose([3, 0, 1, 2]), dtype=torch.float32)

    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.clip_set)

    def __getitem__(self, index):
        # 'Generate one sample of data'
        video_full_path, labels, frame_ind, n_frames_per_clip, vid_idx, frame_pad = self.clip_set[index]

        imgs = self.load_rgb_frames(video_full_path, frame_ind)
        imgs = self.transform(imgs)

        return self.video_to_tensor(imgs), torch.from_numpy(labels), vid_idx, frame_pad


# This dataset loads each full video (frames) separately as a dataset
class IkeaSingleVideoActionDataset(IKEAActionDataset):
    def __init__(self, dataset_path, db_filename='ikea_annotation_db_full', action_list_filename='atomic_action_list.txt',
                 action_object_relation_filename='action_object_relation_list.txt', train_filename='ikea_trainset.txt',
                 test_filename='ikea_testset.txt', transform=None, set='test', example=0, camera='dev3'):
        super().__init__(dataset_path=dataset_path, db_filename=db_filename, action_list_filename=action_list_filename,
                 action_object_relation_filename=action_object_relation_filename, train_filename=train_filename,
                         test_filename=test_filename)
        self.transform = transform
        self.transform_no_norm = transforms.Compose(transform.transforms[0:-1]) # removes the last transformation - assumes normalization last

        self.set = set
        self.example = example
        self.camera = camera
        if self.set == 'train':
            self.video_path = self.trainset_video_list[self.example]
        elif self.set == 'test':
            self.video_path = self.testset_video_list[self.example]
        furniture_type = os.path.split(self.video_path)[0]
        self.video_name = os.path.split(self.video_path)[1]
        self.video_full_path = os.path.join(dataset_path, self.video_path, self.camera, 'images')

        # get list of image files to load files
        files = [os.path.join(self.video_full_path, file_i) for file_i in os.listdir(self.video_full_path)
                 if os.path.isfile(os.path.join(self.video_full_path, file_i)) and file_i.endswith(".jpg")]
        files.sort()
        self.frame_list = files

        # get list of labels
        self.video_idx = self. get_video_id_from_video_path(self.video_path, device=self.camera)
        self.frame_labels = self.get_video_labels(self.video_idx)


    def get_video_labels(self, video_idx):

        video_annotations = self.get_video_annotations_table(video_idx)
        vid_row = self.cursor_vid.execute('''SELECT * FROM videos WHERE id = ?''', (video_idx, )).fetchall()

        n_frames = vid_row[0]["nframes"]

        frame_list = np.arange(0, n_frames)
        label_list = np.zeros_like(frame_list)

        for ann_row in video_annotations:
            action_id = self.get_action_id(ann_row["atomic_action_id"], ann_row["object_id"])   # no need to +1 because table index starts at 1
            if action_id is not None:
                end_frame = ann_row['ending_frame'] if ann_row['ending_frame'] <= n_frames else n_frames
                label_list[ann_row['starting_frame']:end_frame] = action_id

        return label_list

    def __len__(self):
        # 'Denotes the total number of samples'
        return len(self.frame_list)

    def __getitem__(self, index):
        # 'Generates one sample of data'
        # Select sample
        image_name = self.frame_list[index]
        image_label = np.int32(self.frame_labels[index])

        original_image = Image.open(image_name)

        if self.transform is not None:
            image = self.transform(original_image)
            original_image = self.transform_no_norm(original_image)
        else:
            original_image = np.array(original_image)
            image = original_image


        return image, image_label, original_image


# This dataset loads all videos (frames) as a dataset (maintaining video index lookup table)
class IkeaAllSingleVideoActionDataset(IKEAActionDataset):
    def __init__(self, dataset_path, db_filename='ikea_annotation_db_full', action_list_filename='atomic_action_list.txt',
                 action_object_relation_filename='action_object_relation_list.txt', train_filename='train_cross_env.txt',
                 test_filename='test_cross_env.txt', transform=None, set='test', camera='dev3'):
        super().__init__(dataset_path=dataset_path, db_filename=db_filename, action_list_filename=action_list_filename,
                 action_object_relation_filename=action_object_relation_filename, train_filename=train_filename,
                         test_filename=test_filename)

        self.transform = transform

        self.set = set
        self.camera = camera
        if self.set == 'train':
            self.video_list = self.trainset_video_list
        elif self.set == 'test':
            self.video_list = self.testset_video_list
            if transform is not None:
                self.transform_no_norm = transforms.Compose(
                    transform.transforms[0:-1])  # removes the last transformation - assumes normalization last

        self.frame_list = []
        self.frame_labels = []

        self.idx_lut = [] #global index to video-frame index look up table
        for example in range(len(self.video_list)):
            video_path = self.video_list[example]
            furniture_type = os.path.split(video_path)[0]
            video_name = os.path.split(video_path)[1]
            video_full_path = os.path.join(dataset_path, video_path, self.camera, 'images')

            # get list of image files to load files
            files = [os.path.join(video_full_path, file_i) for file_i in os.listdir(video_full_path)
                     if os.path.isfile(os.path.join(video_full_path, file_i)) and file_i.endswith(".jpg")]
            files.sort()
            self.frame_list.append(files)  # if example == 0 else np.concatenate([self.frame_list, files])

            # get list of labels
            video_path_full = os.path.join(video_path, self.camera, 'images')
            video_idx = self. get_video_id_from_video_path(video_path_full, device=self.camera)
            frame_labels = self.get_video_labels(video_idx)
            self.frame_labels.append(frame_labels) #= np.concatenate([self.frame_labels, frame_labels])
            lut = np.array([example*np.ones_like(frame_labels), np.arange(len(frame_labels))])
            self.idx_lut = lut if example == 0 else np.concatenate([self.idx_lut, lut], axis=1)


    def get_video_labels(self, video_idx):

        video_annotations = self.get_video_annotations_table(video_idx)
        vid_row = self.cursor_vid.execute('''SELECT * FROM videos WHERE id = ?''', (video_idx, )).fetchall()
        n_frames = vid_row[0]["nframes"]

        frame_list = np.arange(0, n_frames)
        label_list = np.zeros_like(frame_list)

        for ann_row in video_annotations:
            action_id = self.get_action_id(ann_row["atomic_action_id"], ann_row["object_id"])   # no need to +1 because table index starts at 1
            if action_id is not None:
                end_frame = ann_row['ending_frame'] if ann_row['ending_frame'] <= n_frames else n_frames
                label_list[ann_row['starting_frame']:end_frame] = action_id

        return label_list

    def __len__(self):
        # 'Denotes the total number of samples'
        count = sum([len(listElem) for listElem in self.frame_labels])
        return count

    def __getitem__(self, idx):
        # 'Generates one sample of data'
        # Select sample
        video_index, frame_index = self.idx_lut[:, idx]
        image_name = self.frame_list[video_index][frame_index]
        image_label = np.int32(self.frame_labels[video_index][frame_index])

        original_image = Image.open(image_name)

        if self.transform is not None:
            image = self.transform(original_image)
            original_image = self.transform_no_norm(original_image)
        else:
            original_image = np.array(original_image)
            image = original_image

        return image, image_label, original_image, video_index


# This dataset loads per frame poses
class IKEAPoseActionVideoClipDataset(IKEAActionVideoClipDataset):
    def __init__(self, dataset_path, db_filename='ikea_annotation_db_full', action_list_filename='atomic_action_list.txt',
                 action_object_relation_filename='action_object_relation_list.txt', train_filename='train_cross_env.txt',
                 test_filename='test_cross_env.txt', transform=None, set='test', camera='dev3', frame_skip=1, mode='img',
                 frames_per_clip=64, pose_path='predictions/pose2d/keypoint_rcnn', arch='HCN'):
        super().__init__(dataset_path=dataset_path, db_filename=db_filename, action_list_filename=action_list_filename,
                 action_object_relation_filename=action_object_relation_filename, train_filename=train_filename,
                         test_filename=test_filename, transform=transform, set=set, camera=camera, frame_skip=frame_skip,
                         frames_per_clip=frames_per_clip, mode=mode)
        self.pose_path = pose_path
        self.arch = arch

    # Use this function when GT is fixed - remove the custom collate function after fix
    def load_poses(self, video_full_path, frame_ind):
        pose_seq = []
        video_full_path = video_full_path.replace('images', self.pose_path)
        for i in frame_ind:
            pose_json_filename = os.path.join(video_full_path,
                                              'scan_video_' + str(i).zfill(12) + '_keypoints' + '.json')
            # data = utils.read_pose_json(pose_json_filename)
            with open(pose_json_filename) as json_file:
                data = json.load(json_file)
            data = data['people']
            if len(data) > 1:
                pose = self.get_active_person(data, center=(960, 540), min_bbox_area=20000)
            else:
                pose = np.array(data[0]['pose_keypoints_2d'])  # x,y,confidence

            pose = pose.reshape(-1, 3)
            if self.arch == 'ST_GCN':
                pose = np.delete(pose, 8, 0)
            pose_seq.append(pose)

        pose_seq = torch.tensor(pose_seq, dtype=torch.float32)
        pose_seq = pose_seq[:, :, 0:2].unsqueeze(-1)  # format: frames, joints, coordinates, N_people
        pose_seq = pose_seq.permute(2, 0, 1, 3)  # format: coordinates, frames, joints, N_people

        return pose_seq

    def get_active_person(self, people, center=(960, 540), min_bbox_area=20000):
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
            current_pose = np.array(person['pose_keypoints_2d'])
            joints_2d = np.reshape(current_pose, (-1, 3))[:, :2]
            if 'boxes' in person.keys():
                # maskrcnn
                bbox = person['boxes']
            else:
                # openpose
                idx = np.where(joints_2d.any(axis=1))[0]
                bbox = [np.min(joints_2d[idx, 0]),
                        np.min(joints_2d[idx, 1]),
                        np.max(joints_2d[idx, 0]),
                        np.max(joints_2d[idx, 1])]

            A = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # bbox area
            bbox_center = (bbox[0] + (bbox[2] - bbox[0]) / 2, bbox[1] + (bbox[3] - bbox[1]) / 2)  # bbox center

            dtc = np.sqrt(np.sum((np.array(bbox_center) - np.array(center)) ** 2))
            if dtc < min_dtc:
                closest_pose = current_pose
                if A > min_bbox_area:
                    pose = closest_pose
                    min_dtc = dtc
        # if all bboxes are smaller than threshold, take the closest
        if pose is None:
            pose = closest_pose
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
        diff = skeleton - np.tile(center, [len(skeleton[idx]), 1])
        distances = np.sqrt(np.sum(diff ** 2, 1))
        mean_distance = np.mean(distances)

        return mean_distance


    def __getitem__(self, index):
        # 'Generate one sample of data'
        video_full_path, labels, frame_ind, n_frames_per_clip, vid_idx, frame_pad = self.clip_set[index]
        try:
            poses = self.load_poses(video_full_path, frame_ind)
            return poses, torch.from_numpy(labels), vid_idx, frame_pad
        except Exception as e:
            print('defective input')
            print(e)
            return None


# This dataset loads per frame poses and images
class IKEACombinedDataset(IKEAPoseActionVideoClipDataset):
    def __init__(self, dataset_path, db_filename='ikea_annotation_db_full', action_list_filename='atomic_action_list.txt',
                 action_object_relation_filename='action_object_relation_list.txt', train_filename='train_cross_env.txt',
                 test_filename='test_cross_env.txt', transform=None, set='test', camera='dev3', frame_skip=1, mode='img',
                 frames_per_clip=64, pose_path='predictions/pose2d/openpose_pytorch_ft_all', arch='HCN'):
        super().__init__(dataset_path=dataset_path, db_filename=db_filename, action_list_filename=action_list_filename,
                 action_object_relation_filename=action_object_relation_filename, train_filename=train_filename,
                         test_filename=test_filename, transform=transform, set=set, camera=camera, frame_skip=frame_skip,
                         frames_per_clip=frames_per_clip, mode=mode, arch=arch, pose_path=pose_path)

    def __getitem__(self, index):
        # 'Generate one sample of data'
        video_full_path, labels, frame_ind, n_frames_per_clip, vid_idx, frame_pad = self.clip_set[index]
        try:
            poses = self.load_poses(video_full_path, frame_ind)
            imgs = self.load_rgb_frames(video_full_path, frame_ind)
            imgs = self.transform(imgs)
            return self.video_to_tensor(imgs), poses, torch.from_numpy(labels), vid_idx, frame_pad

        except Exception as e:
            print('defective pose input')
            print(e)
            return None


if __name__ == "__main__":
    dataset_path = '/mnt/sitzikbs_storage/Datasets/ANU_ikea_dataset_smaller'

    db_file = 'ikea_annotation_db_full'
    train_file = 'ikea_trainset.txt'
    test_file = 'ikea_testset.txt'
    action_list_file = 'atomic_action_list.txt'
    action_object_relation_file = 'action_object_relation_list.txt'
    dataset = IKEAActionDataset(dataset_path, db_file, action_list_file, action_object_relation_file, train_file, test_file)

    print(dataset.action_list)