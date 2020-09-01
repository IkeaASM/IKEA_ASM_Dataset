# Author: Yizhak Ben-Shabat (Itzik), 2020
# <sitzikbs at gmail dot com>
# train pose based action recognition  methods on IKEA ASM dataset

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   

import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import HCN
import copy
import st_gcn
sys.path.append('../') # for data loader
sys.path.append('../clip_based/i3d/')  # for utils and video transforms
import i3d_utils as utils

from IKEAActionDataset import IKEAPoseActionVideoClipDataset as Dataset
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--frame_skip', type=int, default=1, help='reduce fps by skippig frames')
parser.add_argument('--batch_size', type=int, default=128, help='number of clips per batch')
parser.add_argument('--n_epochs', type=int, default=3000, help='number of epochs')
parser.add_argument('--steps_per_update', type=int, default=1, help='number of steps per backprop update')
parser.add_argument('--frames_per_clip', type=int, default=32, help='number of frames per clip')
parser.add_argument('--db_filename', type=str, default='ikea_annotation_db_full',
                    help='database file name within dataset path')
parser.add_argument('--arch', type=str, default='HCN', help='which architecture to use')
parser.add_argument('--logdir', type=str, default='./log/debug/', help='path to model save dir')
parser.add_argument('--dataset_path', type=str,
                    default='/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller/', help='path to dataset')
parser.add_argument('--load_mode', type=str, default='img', help='dataset loader mode to load videos or images: '
                                                                 'vid | img')
parser.add_argument('--camera', type=str, default='dev3', help='dataset camera view: dev1 | dev2 | dev3 ')
parser.add_argument('--refine', action="store_true", help='flag to refine the model')
parser.add_argument('--refine_epoch', type=int, default=0, help='refine model from this epoch')
parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU, 999 to use all available gpus')
parser.add_argument('--input_type', type=str, default='rgb', help='depth | rgb ')
parser.add_argument('--pose_relative_path', type=str, default='predictions/pose2d/openpose',
                    help='path to pose dir within the dataset dir')
args = parser.parse_args()


def run(init_lr=0.0001, max_steps=5e3, frames_per_clip=64, dataset_path='/media/sitzikbs/6TB/ANU_ikea_dataset/',
        train_filename='train_cross_env.txt', testset_filename='test_cross_env.txt',
        db_filename='../ikea_dataset_frame_labeler/ikea_annotation_db', logdir='',
        frame_skip=1, batch_size=8, camera='dev3', refine=False, refine_epoch=0, load_mode='img',
         pose_path='predictions/pose2d/openpose', arch='HCN', steps_per_update=1):


    os.makedirs(logdir, exist_ok=True)

    # setup dataset
    train_transforms = None
    test_transforms = None
    train_dataset = Dataset(dataset_path, db_filename=db_filename, train_filename=train_filename,
                 transform=train_transforms, set='train', camera=camera, frame_skip=frame_skip,
                            frames_per_clip=frames_per_clip, mode=load_mode, pose_path=pose_path, arch=arch)
    print("Number of clips in the dataset:{}".format(len(train_dataset)))
    weights = utils.make_weights_for_balanced_classes(train_dataset.clip_set, train_dataset.clip_label_count)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                                   num_workers=6, pin_memory=False)

    test_dataset = Dataset(dataset_path, db_filename=db_filename, train_filename=train_filename,
                           test_filename=testset_filename, transform=test_transforms, set='test', camera=camera,
                           frame_skip=frame_skip, frames_per_clip=frames_per_clip, mode=load_mode,
                           pose_path=pose_path, arch=arch)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6,
                                                  pin_memory=False)

    # setup the model
    num_classes = train_dataset.num_classes
    if arch == 'HCN':
        model = HCN.HCN(in_channel=2, num_joint=19, num_person=1, out_channel=64, window_size=frames_per_clip,
                        num_class=num_classes)
    elif arch == 'ST_GCN':
        graph_args = {'layout': 'openpose', 'strategy': 'spatial'} #ntu-rgb+d
        model = st_gcn.Model(in_channels=2, num_class=num_classes, graph_args=graph_args,
                             edge_importance_weighting=True, dropout=0.5)
    else:
        raise ValueError("Unsupported architecture: please select HCN | ST_GCN")

    if refine:
        if refine_epoch == 0:
            raise ValueError("You set the refine epoch to 0. No need to refine, just retrain.")
        refine_model_filename = os.path.join(logdir, str(refine_epoch).zfill(6)+'.pt')
        checkpoint = torch.load(refine_model_filename)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.cuda()
    # model = nn.DataParallel(model)

    lr = init_lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1E-6)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [1000, 2000, 3000, 4000])
    # criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification

    if refine:
        lr_sched.load_state_dict(checkpoint["lr_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    train_writer = SummaryWriter(os.path.join(logdir, 'train'))
    test_writer = SummaryWriter(os.path.join(logdir, 'test'))

    num_steps_per_update = steps_per_update  # accum gradient - try to have number of examples per update match original code 8*5*4
    # eval_steps  = 5
    steps = 0
    # train it
    n_examples = 0
    train_num_batch = len(train_dataloader)
    test_num_batch = len(test_dataloader)
    refine_flag = True
    best_acc = 0

    while steps < max_steps:#for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, max_steps))
        print('-' * 10)
        if steps <= refine_epoch and refine and refine_flag:
            lr_sched.step()
            steps += 1
            n_examples += len(train_dataset.clip_set)
            continue
        else:
            refine_flag = False
        # Each epoch has a training and validation phase

        test_batchind = -1
        test_fraction_done = 0.0
        test_enum = enumerate(test_dataloader, 0)
        tot_loss = 0.0
        num_iter = 0
        optimizer.zero_grad()

        # Iterate over data.
        avg_acc = []
        for train_batchind, data in enumerate(train_dataloader):

            num_iter += 1
            # get the inputs
            inputs, labels, vid_idx, frame_pad = data

            # wrap them in Variable
            inputs = Variable(inputs.cuda(), requires_grad=True)
            labels = Variable(labels.cuda())
            labels = torch.argmax(labels, dim=1)

            logits = model(inputs)
            t = inputs.size(2)
            per_frame_logits = torch.nn.functional.interpolate(logits.unsqueeze(-1), t, mode='linear', align_corners=True)
            probs = torch.nn.functional.softmax(per_frame_logits, dim=1)


            loss = nn.CrossEntropyLoss()(per_frame_logits, labels)

            tot_loss += loss.item()
            loss.backward()

            acc = utils.accuracy_v2(torch.argmax(per_frame_logits, dim=1), labels)

            avg_acc.append(acc.item())
            train_fraction_done = (train_batchind + 1) / train_num_batch
            print('[{}] train Acc: {}, Loss: {:.4f} [{} / {}]'.format(steps, acc.item(), loss.item(), train_batchind, len(train_dataloader)))
            if (num_iter == num_steps_per_update or train_batchind == len(train_dataloader)-1) :
                n_steps = num_steps_per_update
                if train_batchind == len(train_dataloader)-1:
                    n_steps = num_iter
                n_examples += batch_size*n_steps
                print('updating the model...')
                print('train Total Loss: {:.4f}'.format(tot_loss / n_steps))
                optimizer.step()
                optimizer.zero_grad()
                train_writer.add_scalar('loss', tot_loss / n_steps, n_examples)
                train_writer.add_scalar('Accuracy', np.mean(avg_acc), n_examples)
                train_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], n_examples)
                num_iter = 0
                tot_loss = 0.

            if test_fraction_done <= train_fraction_done and test_batchind + 1 < test_num_batch:
                model.train(False)  # Set model to evaluate mode
                test_batchind, data = next(test_enum)
                inputs, labels, vid_idx, frame_pad = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda(), requires_grad=True)
                labels = Variable(labels.cuda())
                labels = torch.argmax(labels, dim=1)

                with torch.no_grad():
                    logits = model(inputs)
                    t = inputs.size(2)
                    per_frame_logits = torch.nn.functional.interpolate(logits.unsqueeze(-1), t, mode='linear',
                                                                       align_corners=True)
                    probs = torch.nn.functional.softmax(per_frame_logits, dim=1)

                    loss = nn.CrossEntropyLoss()(per_frame_logits, labels)

                    acc = utils.accuracy_v2(torch.argmax(per_frame_logits, dim=1), labels)


                print('[{}] test Acc: {}, Loss: {:.4f} [{} / {}]'.format(steps, acc.item(), loss.item(), test_batchind,
                                                                     len(test_dataloader)))
                test_writer.add_scalar('loss', loss.item(), n_examples)
                test_writer.add_scalar('Accuracy', acc.item(), n_examples)
                test_fraction_done = (test_batchind + 1) / test_num_batch
                model.train(True)
        if steps % 100 == 0:
            # save model
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_state_dict": lr_sched.state_dict()},
                       logdir + str(steps).zfill(6) + '.pt')

        # remember best prec@1 and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        if (is_best):
            model_tmp = copy.deepcopy(model.state_dict())
            model.load_state_dict(model_tmp)
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_state_dict": lr_sched.state_dict()}, os.path.join(logdir, 'best_classifier.pth'))

        steps += 1
        lr_sched.step()
    train_writer.close()
    test_writer.close()


if __name__ == '__main__':

    if not args.gpu_idx == 999:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)  # non-functional
        torch.cuda.set_device(0)

    # need to add argparse
    print("Starting training ...")
    print("Using data from {}".format(args.camera))
    run(dataset_path=args.dataset_path, logdir=args.logdir,
        frame_skip=args.frame_skip, db_filename=args.db_filename, batch_size=args.batch_size, max_steps=args.n_epochs, camera=args.camera,
        refine=args.refine, refine_epoch=args.refine_epoch, load_mode=args.load_mode, pose_path=args.pose_relative_path,
        arch=args.arch, frames_per_clip=args.frames_per_clip, steps_per_update=args.steps_per_update)
