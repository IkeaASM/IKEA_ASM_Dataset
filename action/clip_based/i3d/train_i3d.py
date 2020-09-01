# Author: Yizhak Ben-Shabat (Itzik), 2020
# train I3D on the ikea ASM dataset

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse
import i3d_utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import videotransforms
import numpy as np
from pytorch_i3d import InceptionI3d
sys.path.append('../../')
from IKEAActionDataset import IKEAActionVideoClipDataset as Dataset
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='rgb', help='rgb or flow')
parser.add_argument('--frame_skip', type=int, default=1, help='reduce fps by skippig frames')
parser.add_argument('--steps_per_update', type=int, default=20, help='number of steps per backprop update')
parser.add_argument('--frames_per_clip', type=int, default=64, help='number of frames in a clip sequence')
parser.add_argument('--batch_size', type=int, default=8, help='number of clips per batch')
parser.add_argument('--db_filename', type=str, default='ikea_annotation_db_full',
                    help='database file name within dataset path')
parser.add_argument('--logdir', type=str, default='./log/debug/', help='path to model save dir')
parser.add_argument('--dataset_path', type=str,
                    default='/home/sitzikbs/Datasets/ANU_ikea_dataset_smaller/', help='path to dataset')
parser.add_argument('--load_mode', type=str, default='img', help='dataset loader mode to load videos or images: '
                                                                 'vid | img')
parser.add_argument('--camera', type=str, default='dev2', help='dataset camera view: dev1 | dev2 | dev3 ')
parser.add_argument('--refine', action="store_true", help='flag to refine the model')
parser.add_argument('--refine_epoch', type=int, default=0, help='refine model from this epoch')
parser.add_argument('--input_type', type=str, default='rgb', help='depth | rgb | flow support will be added later')
parser.add_argument('--pretrained_model', type=str, default='charades', help='charades | imagenet')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
args = parser.parse_args()


def run(init_lr=0.001, max_steps=64e3, frames_per_clip=16, mode='rgb', dataset_path='/media/sitzikbs/6TB/ANU_ikea_dataset/',
        train_filename='train_cross_env.txt', testset_filename='test_cross_env.txt',
        db_filename='../ikea_dataset_frame_labeler/ikea_annotation_db', logdir='',
        frame_skip=1, batch_size=8, camera='dev3', refine=False, refine_epoch=0, load_mode='vid',
        input_type='rgb', pretrained_model='charades', steps_per_update=1):


    os.makedirs(logdir, exist_ok=True)

    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    train_dataset = Dataset(dataset_path, db_filename=db_filename, train_filename=train_filename,
                 transform=train_transforms, set='train', camera=camera, frame_skip=frame_skip,
                            frames_per_clip=frames_per_clip, resize=None, mode=load_mode, input_type=input_type)
    print("Number of clips in the dataset:{}".format(len(train_dataset)))
    weights = utils.make_weights_for_balanced_classes(train_dataset.clip_set, train_dataset.clip_label_count)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                                   num_workers=6, pin_memory=True)

    test_dataset = Dataset(dataset_path, db_filename=db_filename, train_filename=train_filename,
                           test_filename=testset_filename, transform=test_transforms, set='test', camera=camera,
                           frame_skip=frame_skip, frames_per_clip=frames_per_clip, resize=None, mode=load_mode,
                           input_type=input_type)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=6,
                                                  pin_memory=True)

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_' + pretrained_model + '.pt'))
    else:
        i3d = InceptionI3d(157, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_' + pretrained_model + '.pt'))

    num_classes = train_dataset.num_classes
    i3d.replace_logits(num_classes)

    for name, param in i3d.named_parameters():  # freeze i3d parameters
        if 'logits' in name:
            param.requires_grad = True
        elif 'Mixed_5c' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    if refine:
        if refine_epoch == 0:
            raise ValueError("You set the refine epoch to 0. No need to refine, just retrain.")
        refine_model_filename = os.path.join(logdir, str(refine_epoch).zfill(6)+'.pt')
        checkpoint = torch.load(refine_model_filename)
        i3d.load_state_dict(checkpoint["model_state_dict"])

    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr

    optimizer = optim.Adam(i3d.parameters(), lr=lr, weight_decay=1E-6)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30, 40])

    if refine:
        lr_sched.load_state_dict(checkpoint["lr_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    train_writer = SummaryWriter(os.path.join(logdir, 'train'))
    test_writer = SummaryWriter(os.path.join(logdir, 'test'))

    num_steps_per_update = 4 * 5 # accum gradient - try to have number of examples per update match original code 8*5*4
    # eval_steps  = 5
    steps = 0
    # train it
    n_examples = 0
    train_num_batch = len(train_dataloader)
    test_num_batch = len(test_dataloader)
    refine_flag = True

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
        tot_loc_loss = 0.0
        tot_cls_loss = 0.0
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

            t = inputs.size(2)
            per_frame_logits = i3d(inputs)
            per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear', align_corners=True)

            # compute localization loss
            loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
            tot_loc_loss += loc_loss.item()

            # compute classification loss (with max-pooling along time B x C x T)
            cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
            tot_cls_loss += cls_loss.item()

            loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update

            tot_loss += loss.item()
            loss.backward()

            acc = utils.accuracy_v2(torch.argmax(per_frame_logits, dim=1), torch.argmax(labels, dim=1))
            # acc = utils.accuracy(per_frame_logits, labels)
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
                train_writer.add_scalar('cls loss', tot_cls_loss / n_steps, n_examples)
                train_writer.add_scalar('loc loss', tot_loc_loss / n_steps, n_examples)
                train_writer.add_scalar('Accuracy', np.mean(avg_acc), n_examples)
                train_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], n_examples)
                num_iter = 0
                tot_loss = 0.

            if test_fraction_done <= train_fraction_done and test_batchind + 1 < test_num_batch:
                i3d.train(False)  # Set model to evaluate mode
                test_batchind, data = next(test_enum)
                inputs, labels, vid_idx, frame_pad = data

                # wrap them in Variable
                inputs = Variable(inputs.cuda(), requires_grad=True)
                labels = Variable(labels.cuda())

                with torch.no_grad():
                    per_frame_logits = i3d(inputs)
                    per_frame_logits = F.interpolate(per_frame_logits, t, mode='linear', align_corners=True)

                    # compute localization loss
                    loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)

                    # compute classification loss (with max-pooling along time B x C x T)
                    cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                                                                  torch.max(labels, dim=2)[0])

                    loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                    acc = utils.accuracy_v2(torch.argmax(per_frame_logits, dim=1), torch.argmax(labels, dim=1))


                print('[{}] test Acc: {}, Loss: {:.4f} [{} / {}]'.format(steps, acc.item(), loss.item(), test_batchind,
                                                                     len(test_dataloader)))
                test_writer.add_scalar('loss', loss.item(), n_examples)
                test_writer.add_scalar('cls loss', loc_loss.item(), n_examples)
                test_writer.add_scalar('loc loss', cls_loss.item(), n_examples)
                test_writer.add_scalar('Accuracy', acc.item(), n_examples)
                test_fraction_done = (test_batchind + 1) / test_num_batch
                i3d.train(True)
        if steps % 2 == 0:
            # save model
            torch.save({"model_state_dict": i3d.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_state_dict": lr_sched.state_dict()},
                       logdir + str(steps).zfill(6) + '.pt')

        steps += 1
        lr_sched.step()
    train_writer.close()
    test_writer.close()


if __name__ == '__main__':
    # need to add argparse
    print("Starting training ...")
    print("Using data from {}".format(args.camera))
    run(init_lr=args.lr, mode=args.mode, dataset_path=args.dataset_path, logdir=args.logdir,
        frame_skip=args.frame_skip, db_filename=args.db_filename, batch_size=args.batch_size, camera=args.camera,
        refine=args.refine, refine_epoch=args.refine_epoch, load_mode=args.load_mode, input_type=args.input_type,
        pretrained_model=args.pretrained_model, steps_per_update=args.steps_per_update,
        frames_per_clip=args.frames_per_clip)
