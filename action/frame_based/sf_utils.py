import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

import sys
sys.path.append('../')
from IKEAActionDataset import IkeaAllSingleVideoActionDataset as Dataset

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def export_visualization(imgs, logits, gt_labels, vis_output_path, class_names, im_path):
    """
    exports an image with results overlay.
    :param logits: predicted logits
    :param gt_labels: ground truth labels
    :param vis_output_path: path to save the output images

    """
    fig_h = plt.figure()

    ax1 = plt.subplot(122)
    plt.imshow(imgs.transpose(1, 2, 0))
    pred = np.argmax(logits)
    plt.title('pred: ' + str(class_names[pred]) + ', GT: ' + str(class_names[gt_labels]))
    plt.axis('off')

    ax2 = plt.subplot(121)
    pos = np.arange(len(class_names))
    rects = ax2.barh(pos, logits,
                     align='center',
                     height=0.5,
                     tick_label=class_names)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.set_title('Logits')
    # plt.show()
    plt.savefig(os.path.join(vis_output_path, os.path.basename(im_path)), bbox_inches = "tight")


def convert_to_action_vis_img(imgs, logits, gt_labels, class_names):
    """
     takes a batch of images and action logits and converts to PIL batch of images
     viuslizing the gt, predicted and logits
    :param img:
    :param logits:
    :param gt_labels:
    :param class_names:
    :return: frames: list of PIL imags
    """
    frames = []
    pos = np.arange(len(class_names))
    matplotlib.rcParams.update({'font.size': 10})
    for i, img in enumerate(imgs):
        fig_h = plt.figure()

        # plot image
        ax1 = plt.subplot(122)
        plt.imshow(img.transpose(1, 2, 0))
        pred = np.argmax(logits[i])
        title_color = 'green' if pred == gt_labels[i] else 'red'
        plt.title('pred: ' + str(class_names[pred]) + ',\n GT: ' + str(class_names[gt_labels[i]]),
                  fontdict={'color': title_color, 'fontsize': 10})
        plt.axis('off')

        # plot light bar blot
        ax2 = plt.subplot(121)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        rects = ax2.barh(pos, logits[i],
                         align='center',
                         height=0.5,
                         tick_label=class_names)
        ax2.set_title('Logits')
        ax2.set_xlim([0.0, 1.0])
        fig_h.tight_layout()
        canvas = matplotlib.backends.backend_agg.FigureCanvas(fig_h)
        fig_h.canvas.draw()
        frames.append(np.array(canvas.renderer.buffer_rgba()))
        plt.close()
    return frames


def export_video_from_frames(imgs, output_filename, fps=30):
    """
    exports a video from a list of PIL images
    :param imgs: list of numpy arrays of images [t x h x w x 4]
    :return:
    """
    import cv2

    nframes, h, w, _ = imgs.shape
    videodims = (w, h)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video_writer = cv2.VideoWriter(output_filename, fourcc, fps, videodims)
    for img in imgs:
        video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGBA2BGR))
    video_writer.release()



if __name__ == '__main__':
    results_path = '/mnt/sitzikbs_storage/PycharmProjects/ikea_asm_dataset-dev/action/action_baseline_single_frame_action_recognition/log/weighted_sampler/vgg/results/'
    results_filename = os.path.join(results_path, 'pred.npy')
    json_filename = os.path.join(results_path, 'segments.json')
    pred_data = np.load(results_filename, allow_pickle=True).item()
    pred_labels = pred_data['pred_labels']
    logits = pred_data['logits']


    db_filename= '/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller_video/ikea_annotation_db_full'
    dataset_path= '/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller_video/'
    train_filename = 'ikea_trainset.txt'
    testset_filename = 'ikea_testset.txt'
    dataset = Dataset(dataset_path, db_filename=db_filename, train_filename=train_filename,
                      transform=None, set='test', camera='dev3', frame_skip=1,
                      frames_per_clip=64, resize=None)

    video_name_list = dataset.set_video_list
    convert_frame_logits_to_segment_json(logits, json_filename, video_name_list, dataset.action_list)