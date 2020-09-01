import torch
import numpy as np
def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""

    batch_size = target.size(0)
    pred = torch.argmax(output, dim=1)
    pred = pred.squeeze()
    correct = pred.eq(target.expand_as(pred))
    acc = correct.view(-1).float().sum(0) * 100 / (batch_size)
    return acc


def sliding_accuracy(logits, target, slider_length):
    '''
        compute the accuracy while averaging over slider_length frames
        implemented to accumulate at the begining of the sequence and give the average for the last frame in the slider
    '''

    n_examples = target.size(0)
    pred = torch.zeros_like(logits)
    for i in range(logits.size(2)):
        pred[:, :, i] = torch.mean(logits[:, :, np.max([0, i - slider_length]):i + 1], dim=2)

    pred = torch.argmax(pred, dim=1)
    pred = pred.squeeze().view(-1)
    correct = pred.eq(target)
    acc = correct.view(-1).float().sum(0) * 100 / n_examples
    return acc, pred


def accuracy_v2(output, target):
    """Computes the precision@k for the specified values of k"""

    batch_size = target.size(0)
    n_frames = target.size(1)
    correct = output.eq(target.expand_as(output))
    acc = correct.view(-1).float().sum(0) * 100 / (batch_size*n_frames)
    return acc


def accuracy_topk(output, target, topk=(1,)):
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


def post_process_logits(per_frame_logits, average=False, num_frames_to_avg=12, threshold = 0.7):
    if average:
        last_frame_logits = torch.mean(per_frame_logits[:, :, -num_frames_to_avg - 1:-1], dim=2)
        label_ind = torch.argmax(last_frame_logits, dim=1).item()
        last_frame_logits = torch.nn.functional.softmax(last_frame_logits, dim=1).squeeze()
    else:
        per_frame_logits = torch.nn.functional.softmax(per_frame_logits, dim=1)
        _, pred = per_frame_logits.topk(1, 1, True, True)
        label_ind = pred.squeeze()[-1].item()
        last_frame_logits = per_frame_logits[0, :, -1].squeeze()

    if last_frame_logits[label_ind] < threshold:
        label_ind = 0

    return label_ind, last_frame_logits


def make_weights_for_balanced_classes(clip_set, label_count):
    """ compute the weight per clip for the weighted random sampler"""
    n_clips = len(clip_set)
    nclasses = len(label_count)
    N = label_count.sum()
    weight_per_class = [0.] * nclasses
    for i in range(nclasses):
        weight_per_class[i] = N/float(label_count[i])

    weight = [0] * n_clips
    for idx, clip in enumerate(clip_set):
        clip_label_sum = clip[1].sum(axis=1)
        if clip_label_sum.sum() == 0:
            print("zero!!!")
        ratios = clip_label_sum / clip_label_sum.sum()
        weight[idx] = np.dot(weight_per_class, ratios)
    return weight