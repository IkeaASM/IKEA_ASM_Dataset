import numpy as np


def compute_video_duration_stats(dataset):
    """
    Traverse the videos in the dataset and count the number of frames
    """
    video_table = dataset.get_annotated_videos_table(device='all')
    n_videos = 0
    total_n_frames = 0
    frames_per_video_list = []
    for row in video_table:
        total_n_frames += row['nframes']
        frames_per_video_list.append(row['nframes'])
        n_videos += 1

    return n_videos, total_n_frames, frames_per_video_list


def compute_action_duration_stats(dataset):
    """
    Traverse the annotations in the dataset and count the number of frames in each action
    """
    video_table = dataset.get_annotation_table()

    frames_per_action_list = []
    for row in video_table:
        action_id = dataset.get_action_id(row['atomic_action_id'], row['object_id'])
        if action_id is not None:  # only count valid actions
            frames_per_action_list.append(row['ending_frame'] - row['starting_frame'])
    return frames_per_action_list


def compute_action_train_test_proportions(dataset):
    """
    Traverse the annotations in the dataset and count the number of clips for each action
    """
    action_list = dataset.action_list
    n_actions = len(action_list)
    video_table = dataset.get_annotation_table()
    train_action_count = np.zeros(n_actions)
    test_action_count = np.zeros(n_actions)
    for row in video_table:
        video_id = row['video_id']
        video_name = dataset.get_video_name_from_id(video_id)
        action_id = dataset.get_action_id(row['atomic_action_id'], row['object_id'])
        if action_id is not None:  # only count valid actions
            if video_name in dataset.trainset_video_list:
                train_action_count[action_id] = train_action_count[action_id] + 1
            elif video_name in dataset.testset_video_list:
                test_action_count[action_id] = test_action_count[action_id] + 1
            else:
                pass
                # print('video not in train or test set...')
    return train_action_count, test_action_count


def compute_occurrence_action_train_test_proportions(dataset):
    """
    Traverse the videos in the dataset and count the number of action occurrences
    i.e. if an action is in the video (no matter how many times) count it once.
    """
    action_list = dataset.action_list
    n_actions = len(action_list)
    video_table = dataset.get_annotated_videos_table(device='dev3').fetchall()
    train_action_count = np.zeros(n_actions)
    test_action_count = np.zeros(n_actions)
    for row in video_table:
        video_id = row['id']
        video_name = dataset.get_video_name_from_id(video_id)
        video_annotation_table = dataset.get_video_annotations_table(video_id).fetchall()
        vid_occurences = [False] * n_actions
        for ann_row in video_annotation_table:
            action_id = dataset.get_action_id(ann_row['atomic_action_id'], ann_row['object_id'])
            if action_id is not None: # only count valid actions
                if not vid_occurences[action_id]:  #dont count the same action in the same video twice
                    vid_occurences[action_id] = True
                    if video_name in dataset.trainset_video_list:
                        train_action_count[action_id] = train_action_count[action_id] + 1
                    elif video_name in dataset.testset_video_list:
                        test_action_count[action_id] = test_action_count[action_id] + 1
    return train_action_count, test_action_count
