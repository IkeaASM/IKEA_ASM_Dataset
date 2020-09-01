import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
import os
import json

def vis_segment_vid(vid_path, gt_labels, action_list, bar_height=0.25, output_path=None, output_filename=None, fps=25):
    """
    Visualize a video with an action segmentation bar plot under it
    :param vid_path: path to video to load
    :param labels: per frame action labels (one hot vectors, allows multi-class per frame)
    :param n_frames: number of frames in the video
    :param action_list: list of action labels
    """
    margin = 30

    gt_labels_argmax = np.argmax(gt_labels, axis=0)
    segment_ranges = find_label_segmetns(gt_labels_argmax)
    legend_labels = np.trim_zeros(np.unique(gt_labels_argmax))

    cap = cv2.VideoCapture(vid_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fig = plt.figure()
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(5, 1)
    ax1 = fig.add_subplot(gs[:-1, :])
    plt.axis('off')
    ax2 = fig.add_subplot(gs[-1, :])
    plt.axis('off')
    ax2.set_xlim([0 - margin, n_frames + margin])
    ax2.set_ylim([-0.5, 0.5])

    plot_segments(ax2, gt_labels_argmax, segment_ranges, action_list, bar_height=bar_height)

    patch = patches.Rectangle([0, -0.5*bar_height*1.25], 10, bar_height*1.25, fc='k')
    flag, frame = cap.read()
    img = ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax2_action_txt = ax2.text(0.5 * n_frames, 0.5, '', horizontalalignment='center')

    def init():
        ax2_action_txt.set_text('')
        ax2.add_patch(patch)
        img.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return patch, img, ax2_action_txt

    def animate(i):
        flag, frame = cap.read()
        if flag:
            img.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            patch.set_xy([i, -0.5*bar_height*1.25])
            ax2_action_txt.set_text(action_list[gt_labels_argmax[i]])
        else:
            plt.close()
            cap.release()

        return patch, img, ax2_action_txt

    anim = animation.FuncAnimation(fig, animate,
                                   init_func=init,
                                   frames=n_frames,
                                   interval=1,
                                   blit=False)
    if output_path is None:
        plt.show()
    else:
        output_filename = 'output.mp4' if output_filename is None else output_filename
        output_filename = os.path.join(output_path, output_filename)
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(output_filename, writer=writer)


def vis_segment_vid_compare(vid_path, gt_labels, pred_labels, action_list, bar_height=0.25, title1='gt', title2='pred'):
    """
    Visualize a video with an action segmentation bar plot under it
    :param vid_path: path to video to load
    :param labels: per frame action labels (one hot vectors, allows multi-class per frame)
    :param n_frames: number of frames in the video
    :param action_list: list of action labels
    """

    #TODO insert titles to bar plots
    margin = 30

    gt_labels_argmax = np.argmax(gt_labels, axis=0)
    pred_labels_argmax =  np.argmax(pred_labels, axis=0)

    segment_ranges = find_label_segmetns(gt_labels_argmax)
    pred_segment_range = find_label_segmetns(pred_labels_argmax)

    cap = cv2.VideoCapture(vid_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fig = plt.figure()
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(8, 1)
    ax1 = fig.add_subplot(gs[:-2, :])
    plt.axis('off')
    ax2 = fig.add_subplot(gs[-2, :])
    plt.axis('off')
    ax2.set_xlim([0 - margin, n_frames + margin])
    ax2.set_ylim([-0.5, 0.5])
    plot_segments(ax2, gt_labels_argmax, segment_ranges, action_list, bar_height=bar_height)

    ax3 = fig.add_subplot(gs[-1, :])
    plt.axis('off')
    ax3.set_xlim([0 - margin, n_frames + margin])
    ax3.set_ylim([-0.5, 0.5])
    plot_segments(ax3, pred_labels_argmax, pred_segment_range, action_list, bar_height=bar_height)

    patch = patches.Rectangle([0, -0.5*bar_height*1.25], 10, bar_height*1.25, fc='k')
    patch3 = patches.Rectangle([0, -0.5 * bar_height * 1.25], 10, bar_height * 1.25, fc='k')
    flag, frame = cap.read()
    img = ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax2_action_txt = ax2.text(0.5 * n_frames, 0.5, '', horizontalalignment='center')

    def init():
        ax2_action_txt.set_text('')
        ax2.add_patch(patch)
        ax3.add_patch(patch3)
        img.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return patch, patch3, img, ax2_action_txt

    def animate(i):
        flag, frame = cap.read()
        if flag:
            img.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            patch.set_xy([i, -0.5*bar_height*1.25])
            patch3.set_xy([i, -0.5 * bar_height * 1.25])
            ax2_action_txt.set_text(action_list[gt_labels_argmax[i]])
        else:
            plt.close()
            cap.release()

        return patch, patch3, img, ax2_action_txt

    anim = animation.FuncAnimation(fig, animate,
                                   init_func=init,
                                   frames=n_frames,
                                   interval=1,
                                   blit=False)
    plt.show()


def find_label_segmetns(labels):
    """

    finds the range of label segments in a given labels array

    Parameters
    ----------
    labels : numpy array of per frame labels

    Returns
    -------
    ranges : list of segments ranges
    """
    diff = np.diff(labels)
    iszero = np.concatenate(([0], np.equal(diff, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def plot_segments(ax, labels, segment_ranges, action_list, bar_height=0.4):
    """

    plot a video segmentation bar plot

    Parameters
    ----------
    ax2 : matplotlib axes to plit in
    gt_labels : numpy array of per frame labels
    segment_ranges : label segment start and end
    action_list : full action class list

    Returns
    -------

    """
    n_frames = len(labels)
    n_classes = len(action_list)

    # generate color map
    cmap = plt.cm.get_cmap('rainbow', n_classes)
    action_color = [cmap(x) for x in range(n_classes)]
    action_color[0] = [1., 1., 1.] #set first color to white

    seen_labels = []
    # ax.barh(0, [0-10, n_frames+10], align='center', height=bar_height*1.15, color='w', edgecolor=[0.5, 0.5, 0.5])
    for i, segment in enumerate(segment_ranges):
        # bar_start = 0 if i == 0 else segment_ranges[i-1][1]
        cls = labels[segment[0]]
        if cls not in seen_labels:
            seen_labels.append(cls)
            ax.barh(0, segment, left=segment[0], align='center', height=bar_height, color=action_color[cls], label=cls)
        else:
            ax.barh(0, segment, left=segment[0], align='center', height=bar_height, color=action_color[cls])

    rect = patches.Rectangle([0, -0.5 * bar_height], n_frames, bar_height, fill=False, edgecolor=[0.5, 0.5, 0.5])
    ax.add_patch(rect)
    # legend_entries = [action_list[label] for label in seen_labels]
    # plt.legend(legend_entries, loc="lower left", ncol=len(seen_labels))


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

    relative_depth_frame = cv2.convertScaleAbs(img, alpha=(255.0/max_depth_val), beta=-255.0*min_depth_val/max_depth_val)
    # relative_depth_frame = cv2.cvtColor(relative_depth_frame, cv2.COLOR_GRAY2BGR)
    relative_depth_frame = cv2.applyColorMap(relative_depth_frame, cv2.COLORMAP_JET)
    return relative_depth_frame


def overlay_segmentation_mask(image, predictions, part_tracks, dict_colors, color_cat, cat_dict):
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


