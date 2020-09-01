# Author: Yizhak Ben-Shabat (Itzik), 2020
# <sitzikbs at gmail dot com>
# test frame based action recognition methods on IKEA ASM dataset and save the predictions

import numpy as np
from torch.utils import data
import torch
import torch.nn as nn
import sys
sys.path.append('../')
from IKEAActionDataset import IkeaAllSingleVideoActionDataset as Dataset
# from vgg import vgg19
import torchvision.models as models
import torchvision.transforms as transforms
import os
import utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='../../../ikea_asm_dataset/ANU_ikea_dataset_smaller/',
                    help='path to raw image IKEA ASM dataset dir after images were resized')
parser.add_argument('--test_filename', type=str, default='test_cross_env.txt',
                    help='file name containnig the list of test files (within the dataset directory)')
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--arch', type=str, default='resnet50', help='architecture: resnet18|resnet50')
parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU, 999 to use all available gpus')
args = parser.parse_args()

gpu_idx = args.gpu_idx
batch_size = args.batch_size
dataset_path = args.dataset_path
test_filename = args.test_filename
arch = args.arch

logdir = os.path.join('./log/', arch)
output_path = os.path.join(logdir, 'results')
os.makedirs(output_path, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
torch.cuda.set_device(0)
device = torch.device("cuda:%d" % 0)


def validate(val_set, val_loader, model, output_path):

    pred_output_filename = os.path.join(output_path, 'pred.npy')
    json_output_filename = os.path.join(output_path, 'action_segments.json')
    # switch to evaluate mode
    model.eval()

    pred_labels_accum = np.array([])
    logits_accum = []
    pred_labels_per_video = []
    logits_per_video = []
    last_vid_idx = 0
    with torch.no_grad():
        for step, (im_data, im_class, im_path, vid_idx) in enumerate(val_loader):
            print('Processing video batch [{}/{}]'.format(step, len(val_loader)))
            im_data = im_data.to(device)
            vid_idx = vid_idx.detach().cpu().numpy()
            # compute output
            output = model(im_data)
            pred_labels = torch.argmax(output, 1).detach().cpu().numpy()
            logits = torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy().tolist()

            # accumulate results per video
            if not vid_idx[0] == vid_idx[-1]:  # video index changes in the middle of the batch
                print('Finished video {}'.format(vid_idx[0]))
                idx_change = np.where(vid_idx[:-1] != vid_idx[1:])[0][0] + 1  # find where the video index changes
                pred_labels_accum = np.concatenate([pred_labels_accum, pred_labels[:idx_change]])
                pred_labels_per_video.append(pred_labels_accum)
                pred_labels_accum = np.array(pred_labels[idx_change:])

                logits_accum.append(logits[:idx_change])
                logits_per_video.append(np.vstack(logits_accum))
                logits_accum = logits[idx_change:]
            elif not last_vid_idx == vid_idx[0]:  # video index changes after a full batch
                print('Finished video {}'.format(last_vid_idx))
                pred_labels_per_video.append(pred_labels_accum)
                pred_labels_accum = np.array(pred_labels)

                logits_per_video.append(np.vstack(logits_accum))
                logits_accum = logits
            elif step == len(val_loader)-1:  # last batch
                print('Finished video {} - the last batch'.format(last_vid_idx))
                pred_labels_accum = np.concatenate([pred_labels_accum, pred_labels])
                pred_labels_per_video.append(pred_labels_accum)

                logits_accum.append(logits)
                logits_per_video.append(np.vstack(logits_accum))
            else:
                pred_labels_accum = np.concatenate([pred_labels_accum, pred_labels])
                logits_accum.append(logits)
            last_vid_idx = vid_idx[-1]

    np.save(pred_output_filename, {'pred_labels': pred_labels_per_video, 'logits': logits_per_video})
    utils.convert_frame_logits_to_segment_json(logits_per_video, json_output_filename, val_set.video_list,
                                               val_set.action_list)

def main():

    res = 256
    center_crop = 224
    val_transform = transforms.Compose([
        transforms.Resize(res),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
        transforms.Normalize( mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Generators
    val_set = Dataset(dataset_path=dataset_path, set='test', transform=val_transform, test_filename=test_filename)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)

    num_classes = 33
    if arch == 'vgg':
        # use vgg architecture
        model_ft = vgg19(pretrained=False)  ##### Model Structure Here
        model_ft.classifier[6] = nn.Linear(4096, num_classes)  # change last layer to fit the number of classes
    elif 'resnet' in arch:
        # use resnet architecture
        model_ft = models.__dict__[arch](pretrained=False)
        if arch =='resnet18' or arch =='resnet34':
            model_ft.fc = nn.Linear(512, num_classes)
        elif arch == 'resnet50' or arch == 'resnet101' or arch == 'resnet152':
            model_ft = models.__dict__[arch](pretrained=False)
            model_ft.fc = nn.Linear(2048, num_classes)
    else:
        raise ValueError("unsupported architecture")

    try:
        model_ft.load_state_dict(torch.load(os.path.join(logdir, 'best_classifier.pth')))
    except:
        model_ft = nn.DataParallel(model_ft)
        model_ft.load_state_dict(torch.load(os.path.join(logdir, 'best_classifier.pth')))
    model_ft.to(device)

    # evaluate on validation set
    model_ft.eval()
    validate(val_set, val_loader, model_ft, output_path)


if __name__ == '__main__':
    main()
    os.system('python3 ../evaluation/evaluate.py --results_path {} --mode frame'.format(output_path))
