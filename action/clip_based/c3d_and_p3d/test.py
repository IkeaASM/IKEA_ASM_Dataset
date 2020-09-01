# Author: Yizhak Ben-Shabat (Itzik), 2020
# test clip based P3d and C3D methods on IKEA ASM dataset

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import argparse
import sys
sys.path.append('../../') # for ikea datast data loader
import utils
sys.path.append('../i3d') # for utils and video transforms
import i3d_utils
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import videotransforms
import numpy as np
import c3d
import p3d
from IKEAActionDataset import IKEAActionVideoClipDataset as Dataset


parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, default='rgb', help='rgb | depth or flow, flow is currently unsupported in the dataset')
parser.add_argument('-arch', type=str, default='p3d', help='select method: c3d | p3d ')
parser.add_argument('-frame_skip', type=int, default=1, help='reduce fps by skippig frames')
parser.add_argument('-batch_size', type=int, default=8, help='number of clips per batch')
parser.add_argument('-db_filename', type=str,
                    default='/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller/ikea_annotation_db_full',
                    help='database file')
parser.add_argument('-model_path', type=str, default='./log/',
                    help='path to model save dir')
parser.add_argument('-device', default='dev3', help='which camera to load')
parser.add_argument('-model', type=str, default='best_classifier.pt', help='path to model save dir')
parser.add_argument('-dataset_path', type=str,
                    default='/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller/', help='path to dataset')
args = parser.parse_args()


def run(dataset_path, db_filename, model_path, output_path, frames_per_clip=16, mode='rgb',
        testset_filename='test_cross_env.txt', trainset_filename='train_cross_env.txt', frame_skip=1,
        batch_size=8, device='dev3', model_name='p3d'):

    pred_output_filename = os.path.join(output_path, 'pred.npy')
    json_output_filename = os.path.join(output_path, 'action_segments.json')

    # setup dataset
    img_size = 112 if model_name == 'c3d' else 160
    test_transforms = transforms.Compose([videotransforms.CenterCrop(img_size)])

    test_dataset = Dataset(dataset_path, db_filename=db_filename, test_filename=testset_filename,
                           train_filename=trainset_filename, transform=test_transforms, set='test', camera=device,
                           frame_skip=frame_skip, frames_per_clip=frames_per_clip, resize=None, mode='img',
                           input_type=mode)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6,
                                                  pin_memory=True)

    # setup the model
    num_classes = test_dataset.num_classes
    if model_name == 'c3d':
        model = c3d.C3D()
        model.load_state_dict(torch.load('c3d.pickle'))
        model.replace_logits(num_classes)
    elif model_name == 'p3d':
        model = p3d.P3D199(pretrained=False, modality='RGB', num_classes=num_classes)
    else:
        raise ValueError("unsupported model")

    checkpoints = torch.load(model_path)
    model.load_state_dict(checkpoints["model_state_dict"]) # load trained model
    model.cuda()
    model = nn.DataParallel(model)

    n_examples = 0

    # Iterate over data.
    avg_acc = []
    pred_labels_per_video = [[] for i in range(len(test_dataset.video_list))]
    logits_per_video = [[] for i in range(len(test_dataset.video_list))]

    for test_batchind, data in enumerate(test_dataloader):
        model.train(False)
        # get the inputs
        inputs, labels, vid_idx, frame_pad = data

        # wrap them in Variable
        inputs = Variable(inputs.cuda(), requires_grad=True)
        labels = Variable(labels.cuda())

        t = inputs.size(2)
        logits = model(inputs)
        logits = torch.nn.functional.interpolate(logits.unsqueeze(-1), t, mode='linear', align_corners=True)

        acc = i3d_utils.accuracy_v2(torch.argmax(logits, dim=1), torch.argmax(labels, dim=1))
        avg_acc.append(acc.item())
        n_examples += batch_size
        print('batch Acc: {}, [{} / {}]'.format(acc.item(), test_batchind, len(test_dataloader)))
        logits = logits.permute(0, 2, 1)
        logits = logits.reshape(inputs.shape[0] * frames_per_clip, -1)
        pred_labels = torch.argmax(logits, 1).detach().cpu().numpy()
        logits = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy().tolist()

        pred_labels_per_video, logits_per_video = \
            utils.accume_per_video_predictions(vid_idx, frame_pad, pred_labels_per_video, logits_per_video, pred_labels,
                                               logits, frames_per_clip)

    pred_labels_per_video = [np.array(pred_video_labels) for pred_video_labels in pred_labels_per_video]
    logits_per_video = [np.array(pred_video_logits) for pred_video_logits in logits_per_video]
    np.save(pred_output_filename, {'pred_labels': pred_labels_per_video, 'logits': logits_per_video})
    utils.convert_frame_logits_to_segment_json(logits_per_video, json_output_filename, test_dataset.video_list,
                                               test_dataset.action_list)


if __name__ == '__main__':
    model_dir = os.path.join(args.model_path, args.arch)
    output_path = os.path.join(model_dir, 'results')
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(model_dir, args.model)
    run(dataset_path=args.dataset_path, db_filename=args.db_filename, model_path=model_path,
        output_path=output_path, frame_skip=args.frame_skip,  mode=args.mode, batch_size=args.batch_size,
        device=args.device, model_name=args.arch)
    os.system('python3 ../../evaluation/evaluate.py --results_path {} --mode vid'.format(output_path))
