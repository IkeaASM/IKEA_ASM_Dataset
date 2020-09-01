# Author: Yizhak Ben-Shabat (Itzik), 2020
# <sitzikbs at gmail dot com>
# test pose based action recognition methods on IKEA ASM dataset

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import argparse
import itertools
import sys
sys.path.append('../') # for data loader
sys.path.append('../clip_based/i3d/')  # for utils and video transforms
import i3d_utils
import utils
import torch
from torch.autograd import Variable
from torchvision import transforms
import videotransforms
import numpy as np
import HCN
import st_gcn
from IKEAActionDataset import IKEAPoseActionVideoClipDataset as Dataset


parser = argparse.ArgumentParser()
parser.add_argument('--frame_skip', type=int, default=1, help='reduce fps by skipping frames')
parser.add_argument('--batch_size', type=int, default=128, help='number of clips per batch')
parser.add_argument('--frames_per_clip', type=int, default=128, help='number of frames in a sequence')
parser.add_argument('--arch', type=str, default='HCN', help='ST_GCN | HCN indicates which architecture to use')
parser.add_argument('--db_filename', type=str,
                    default='/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller/ikea_annotation_db_full',
                    help='database file')
parser.add_argument('--model_path', type=str, default='./log/HCN_128/',
                    help='path to model save dir')
parser.add_argument('--device', default='dev3', help='which camera to load')
parser.add_argument('--model', type=str, default='best_classifier.pth',
                    help='path to model save dir')
parser.add_argument('--dataset_path', type=str,
                    default='/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller/', help='path to dataset')
parser.add_argument('--pose_relative_path', type=str, default='predictions/pose2d/openpose',
                    help='path to pose dir within the dataset dir')
args = parser.parse_args()


def run(dataset_path, db_filename, model_path, output_path, frames_per_clip=16,
        testset_filename='test_cross_env.txt', trainset_filename='train_cross_env.txt', frame_skip=1,
        batch_size=8, device='dev3', arch='HCN', pose_path='predictions/pose2d/openpose'):

    pred_output_filename = os.path.join(output_path, 'pred.npy')
    json_output_filename = os.path.join(output_path, 'action_segments.json')

    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    test_dataset = Dataset(dataset_path, db_filename=db_filename, test_filename=testset_filename,
                           train_filename=trainset_filename, transform=test_transforms, set='test', camera=device,
                           frame_skip=frame_skip, frames_per_clip=frames_per_clip, mode='img', pose_path=pose_path,
                           arch=arch)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6,
                                                  pin_memory=True)

    # setup the model
    num_classes = test_dataset.num_classes

    if arch == 'HCN':
        model = HCN.HCN(in_channel=2, num_joint=19, num_person=1, out_channel=64, window_size=frames_per_clip,
                        num_class=num_classes)
    elif arch == 'ST_GCN':
        graph_args = {'layout': 'openpose', 'strategy': 'spatial'}  # layout:'ntu-rgb+d'
        model = st_gcn.Model(in_channels=2, num_class=num_classes, graph_args=graph_args,
                             edge_importance_weighting=True, dropout=0.5)
    else:
        raise ValueError("Unsupported architecture: please select HCN | ST_GCN")

    checkpoints = torch.load(model_path)
    model.load_state_dict(checkpoints["model_state_dict"]) # load trained model
    model.cuda()
    # model = nn.DataParallel(model)

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
        # logits = F.interpolate(logits, t, mode='linear', align_corners=True)  # b x classes x frames

        acc = i3d_utils.accuracy_v2(torch.argmax(logits, dim=1), torch.argmax(labels, dim=1))

        avg_acc.append(acc.item())
        n_examples += batch_size
        print('batch Acc: {}, [{} / {}]'.format(acc.item(), test_batchind, len(test_dataloader)))
        logits = logits.permute(0, 2, 1)  # [ batch, classes, frames] -> [ batch, frames, classes]
        logits = logits.reshape(inputs.shape[0] * frames_per_clip, -1)
        pred_labels = torch.argmax(logits, 1).detach().cpu().numpy().tolist()
        logits = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy().tolist()

        pred_labels_per_video, logits_per_video = \
            utils.accume_per_video_predictions(vid_idx, frame_pad,pred_labels_per_video, logits_per_video, pred_labels,
                                     logits, frames_per_clip)

    pred_labels_per_video = [np.array(pred_video_labels) for pred_video_labels in pred_labels_per_video]
    logits_per_video = [np.array(pred_video_logits) for pred_video_logits in logits_per_video]

    np.save(pred_output_filename, {'pred_labels': pred_labels_per_video,
                                   'logits': logits_per_video})
    utils.convert_frame_logits_to_segment_json(logits_per_video, json_output_filename, test_dataset.video_list,
                                               test_dataset.action_list)


if __name__ == '__main__':
    # need to add argparse
    output_path = os.path.join(args.model_path, 'results')
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(args.model_path, args.model)
    run(dataset_path=args.dataset_path, db_filename=args.db_filename, model_path=model_path,
        output_path=output_path, frame_skip=args.frame_skip,  batch_size=args.batch_size,
        device=args.device, arch=args.arch, pose_path=args.pose_relative_path, frames_per_clip=args.frames_per_clip)
    os.system('python3 ../evaluation/evaluate.py --results_path {} --mode vid'.format(output_path))
