# global utils for all action recognition abselines
import numpy as np
import json
import matplotlib.pyplot as plt

def squeeze_class_names(class_names):
    """
    shortens the class names for visualization

    :param class_names: class name
    :return: class_names: shortened class names
    """

    class_names = [substring.split(" ") for substring in class_names]
    for i, cls in enumerate(class_names):
        if len(cls) <= 4:
            class_names[i] = " ".join(cls)
        else:
            class_names[i] = " ".join(cls[0:4]) + "..."
    return class_names


def find_label_segments_and_scores(input, mode="logits"):
    """
    finds the range of label segments in a given labels array and computes the segment score as max logit average

    Parameters
    ----------
    logits : numpy array of per frame logits
    mode : logits | labels
    Returns
    -------
    ranges : list of segments ranges
    scores : segment score (logits average)
    segment_label : the label of the corresponding segment
    """
    if mode == "logits":
        logits = input
        labels = np.argmax(logits, axis=1)
    else:
        labels = input

    diff = np.diff(labels)
    iszero = np.concatenate(([0], np.equal(diff, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    segments = np.where(absdiff == 1)[0].reshape(-1, 2)
    scores = []
    segment_label = []
    for segment in segments:
        segment_label.append(labels[segment[0]])
        if mode == "logits":
            scores.append(np.mean(logits[segment[0]:segment[1], labels[segment[0]]]))
        else:
            scores.append(1.0)
    return segments, np.array(scores), segment_label


def convert_frame_logits_to_segment_json(logits, json_filename, video_name_list, action_list, mode="logits",
                                         details=""):
    """
    convert dataset per frame logits to scored segments and save to .json file in the ActivityNet format
    for action localization evaluation
    http://activity-net.org/challenges/2020/tasks/anet_localization.html

    Parameters
    ----------
    logits : per frame logits or labels (depending on mode)
    json_filename : output .json file name (full path)
    video_name_list : list of video names
    action_list : list of action labels
    mode : logits | labels
     Returns
    -------
    """
    json_dict_to_write = {"version": "VERSION 1.3"}

    results = {}
    for i, vid_logits in enumerate(logits):
        segments, scores, segment_labels = find_label_segments_and_scores(vid_logits, mode=mode)
        list_of_result_dicts = []
        vid_name = video_name_list[i]
        for j, segment in enumerate(segments):
            list_of_result_dicts.append({"label": action_list[segment_labels[j]], "score": float(scores[j]),
                                         "segment": segment.tolist()})
        results[vid_name] = list_of_result_dicts
    json_dict_to_write["results"] = results
    json_dict_to_write["external_data"] = {"details": details}
    with open(json_filename, 'w') as outfile:
        json.dump(json_dict_to_write, outfile)

def convert_db_to_segment_json(logits, json_filename, video_name_list, action_list, mode="logits",
                                         details="", subset=["testing"]):
    """
    convert dataset per frame labels to segments and save to .json file in the ActivityNet format
    for action localization evaluation
    http://activity-net.org/challenges/2020/tasks/anet_localization.html

    Parameters
    ----------
    logits : per frame logits or labels (depending on mode)
    json_filename : output .json file name (full path)
    video_name_list : list of video names
    action_list : list of action labels
    mode : logits | labels
    subset : list of strings containing training | testing corresponding to the example subset association
    Returns
    -------
    """
    json_dict_to_write = {"version": "VERSION 1.3"}

    database = {}
    for i, vid_logits in enumerate(logits):
        vid_name = video_name_list[i]
        database[vid_name] = {}
        database[vid_name]["subset"] = subset[i]
        segments, _, segment_labels = find_label_segments_and_scores(vid_logits, mode=mode)
        list_of_result_dicts = []
        for j, segment in enumerate(segments):
            list_of_result_dicts.append({"label": action_list[segment_labels[j]], "segment": segment.tolist()})
        database[vid_name]["annotation"] = list_of_result_dicts
    json_dict_to_write["database"] = database
    with open(json_filename, 'w') as outfile:
        json.dump(json_dict_to_write, outfile)

# def convert_segment_json_to_frame_labels(json_filename, dataset):
#     """
#      Loads a label segment .json file (ActivityNet format
#       http://activity-net.org/challenges/2020/tasks/anet_localization.html) and converts to frame labels for evaluation
#
#     Parameters
#     ----------
#     json_filename : output .json file name (full path)
#     video_name_list : list of video names
#     action_list : list of action labels
#     Returns
#     -------
#     frame_labels: one_hot grame labels (allows multi-label)
#     """
#     labels = []
#     with open(json_filename, 'r') as json_file:
#         json_dict = json.load(json_file)
#     video_results = json_dict["results"]
#     for video_name in video_results:
#         n_frames = get_nframes_from_db(db_filename, video_name )
#         labels
#     return labels

def plot_class_acc_comparison(acc_mat,
                          class_names=None,
                          methods_name=None,
                          title=None,
                          cmap=None):
    """
    given a matrix of rows for methods and columns for per class performance - plot an image

    Arguments
    ---------
    acc_mat:           per class accuracy matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    methods_name: given method names

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    """
    if cmap is None:
        cmap = plt.get_cmap('Blues')
    plt.figure(figsize=(26, 26))
    plt.matshow(acc_mat, cmap=cmap, fignum=1)
    if title is not None:
        plt.title(title)
    # plt.colorbar()
    if class_names is not None:
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90)
        # plt.yticks(tick_marks, class_names)
    if methods_name is not None:
        tick_marks = np.arange(len(acc_mat))
        plt.yticks(tick_marks, methods_name)

    plt.xlim([0 - 0.5, acc_mat.shape[1] - 0.5])
    plt.ylim([0 - 0.5, acc_mat.shape[0] - 0.5])
    plt.gca().invert_yaxis()

    thresh = acc_mat.max() / 2
    for (i, j), z in np.ndenumerate(acc_mat):
            plt.text(j, i, "{:0.2f}".format(z),
                     ha="center", va="center",
                     color="white" if z > thresh else "black")
    # plt.tight_layout()
    plt.ylabel('Method')
    plt.xlabel('Accuracy per class')
    plt.tight_layout()
    # plt.show()
    return plt.gcf(), plt.gca()


def plot_confusion_matrix(cm,
                          target_names,
                          title=None,
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """


    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        cm_sum[cm_sum == 0] = 1
        cm = cm.astype('float') / cm_sum

    plt.figure(figsize=(26, 26))
    plt.matshow(cm, cmap=cmap, fignum=1)
    if title is not None:
        plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    plt.xlim([0 - 0.5, cm.shape[1] - 0.5])
    plt.ylim([0 - 0.5, cm.shape[0] - 0.5])
    plt.gca().invert_yaxis()

    if normalize:
        thresh = 0.4
    else:
        thresh = cm.max() / 2

    for (i, j), z in np.ndenumerate(cm):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(z),
                     ha="center", va="center",
                     color="white" if z > thresh else "black", fontsize=12)
        else:
            plt.text(j, i, "{:0.2f,}".format(z),
                     ha="center", va="center",
                     color="white" if z > thresh else "black")


    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    return plt.gcf(), plt.gca()


############################################ Pose processing utils ########################################
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
    data = {}
    for track_id_str in json_data.keys():
        track_id = int(track_id_str)
        data[track_id] = {}
        for labels in json_data[track_id_str].keys():
            data[track_id][labels] = np.array(json_data[track_id_str][labels]) # Convert lists to numpy arrays
    return data


def get_staf_joint_names():
    # using OpenPose BODY_21 parts, from staf openpose repo: https://github.com/soulslicer/openpose/blob/staf/include/openpose/pose/poseParametersRender.hpp
    return [
        'OP Nose',  # 0,
        'OP Neck',  # 1,
        'OP RShoulder',  # 2,
        'OP RElbow',  # 3,
        'OP RWrist',  # 4,
        'OP LShoulder',  # 5,
        'OP LElbow',  # 6,
        'OP LWrist',  # 7,
        'OP MidHip',  # 8,
        'OP RHip',  # 9,
        'OP RKnee',  # 10,
        'OP RAnkle',  # 11,
        'OP LHip',  # 12,
        'OP LKnee',  # 13,
        'OP LAnkle',  # 14,
        'OP REye',  # 15,
        'OP LEye',  # 16,
        'OP REar',  # 17,
        'OP LEar',  # 18,
        'Neck (LSP)',  # 19,
        'Top of Head (LSP)',  # 20,
    ]


def get_staf_skeleton():
    """
    Returns
    -------
    list of parts (skeleton key point pairs)
    """
    return np.array([[0, 1],
                     [1, 2],
                     [2, 3],
                     [3, 4],
                     [1, 5],
                     [5, 6],
                     [6, 7],
                     [1, 8],
                     [9, 8],
                     [9, 10],
                     [10, 11],
                     [8, 12],
                     [12, 13],
                     [13, 14],
                     [0, 15],
                     [0, 16],
                     [15, 17],
                     [16, 18]
                     ])


def get_pose_colors(mode='rgb'):
    """

    Parameters
    ----------
    mode : rgb | bgr color format to return

    Returns
    -------
    list of part colors for skeleton visualization
    """
    # colormap from OpenPose: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/3c9441ae62197b478b15c551e81c748ac6479561/include/openpose/pose/poseParametersRender.hpp
    colors = np.array(
        [
            [255., 0., 85.],
            # [255., 0., 0.],
            [255., 85., 0.],
            [255., 170., 0.],
            [255., 255., 0.],
            [170., 255., 0.],
            [85., 255., 0.],
            [0., 255., 0.],
            [255., 0., 0.],
            [0., 255., 85.],
            [0., 255., 170.],
            [0., 255., 255.],
            [0., 170., 255.],
            [0., 85., 255.],
            [0., 0., 255.],
            [255., 0., 170.],
            [170., 0., 255.],
            [255., 0., 255.],
            [85., 0., 255.],

            [0., 0., 255.],
            [0., 0., 255.],
            [0., 0., 255.],
            [0., 255., 255.],
            [0., 255., 255.],
            [0., 255., 255.]])
    if mode == 'rgb':
        return colors
    elif mode == 'bgr':
        colors[:, [0, 2]] = colors[:, [2, 0]]
        return colors
    else:
        raise ValueError('Invalid color mode, please specify rgb or bgr')


def accume_per_video_predictions(vid_idx, frame_pad, pred_labels_per_video, logits_per_video, pred_labels,
                                 logits, frames_per_clip):
    """
    This is a helper function to accumulate the predictions of teh different batches into a single list
    containing the predictions for each video separately. It is used in all of the test files except the frame based
    (no sequence)
    Parameters
    ----------
    vid_idx : list of video index corresponding for each element in the batch
    frame_pad : list of number of padded frames per element in the batch
    pred_labels_per_video : predicted labels per video - accumulated from previous batch
    logits_per_video : logits per video - accumulated from previous batch
    pred_labels : the current batch predictions
    logits : the current batch logits
    frames_per_clip : number of frames per clip (int)

    Returns
    -------
        pred_labels_per_video : predicted labels per video - accumulated from previous batch
        logits_per_video : logits per video - accumulated from previous batch
    """

    for i in range(len(vid_idx)):
        batch_vid_idx = vid_idx[i].item()
        batch_frame_pad = frame_pad[i].item()

        pred_labels_per_video[batch_vid_idx].extend(pred_labels[i*frames_per_clip:(i+1)*frames_per_clip])
        if not batch_frame_pad == 0:
            pred_labels_per_video[batch_vid_idx] = pred_labels_per_video[batch_vid_idx][0:-batch_frame_pad]


        logits_per_video[batch_vid_idx].extend(logits[i*frames_per_clip:(i+1)*frames_per_clip])
        if not batch_frame_pad == 0:
            logits_per_video[batch_vid_idx] = logits_per_video[batch_vid_idx][0:-batch_frame_pad]

    return pred_labels_per_video, logits_per_video

