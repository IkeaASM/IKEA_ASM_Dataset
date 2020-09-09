# Mask R-CNN for 2D Human Pose
#
# For training, use a flattened directory structure split into test and train folders.
# For testing (i.e. using --save_results), use the standard structured dataset directory without test/train splits.
#
# Adapted from PyTorch ImageNet example:
# https://github.com/pytorch/examples/blob/master/imagenet/main.py
#
# Dylan Campbell <dylan.campbell@anu.edu.au>

import argparse
import os
import io
import sys
import random
import shutil
import time
import copy
import json
import warnings
from PIL import Image
import matplotlib.pyplot as plt

from ikea_pose_dataset import IKEAKeypointRCNNDataset, IKEAKeypointRCNNTestDataset, IKEAKeypointRCNNVideoTestDataset
from joint_ids import get_ikea_joint_names, get_ikea_joint_hflip_names, get_ikea_connectivity

from pdb import set_trace as st

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import torch.utils.tensorboard as tb

# mp.set_sharing_strategy('file_system') # for error: too many open files


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N',
                    help='mini-batch size (default: 2), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--batch-size-val', default=None, type=int,
                    help='mini-batch size (default: training batch-size)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--logdir', default='./temp/', type=str,
                    help='path to where the logs are saved')
# parser.add_argument('--save_results', dest='save_results', action='store_true',
#                     help='Evaluate model on training and validation sets and save results.')
parser.add_argument('--save_results', default='', type=str,
                    help='Evaluate model on training and validation sets and save results ["frames", "video"].')
parser.add_argument('--out_data_dir', default='/home/djcam/Documents/HDD/datasets/ikea/ikea_asm/', type=str,
                    help='path to where results will be saved')
parser.add_argument('--pck_threshold', type=float, default=10.0,
                    help='threshold for PCK measure in pixels')
parser.add_argument('--image_scale_factor', type=float, default=1.0,
                    help='factor by which dataset images have been scaled down')
parser.add_argument('--camera_id', type=str, default='dev3',
                    help='camera device ID for dataset (GT annotations for dev3 only) [for save_results only]')

best_mpjpe = 10000.0

def main():
    args = parser.parse_args()

    if not args.batch_size_val:
        args.batch_size_val = args.batch_size
    args.writer = tb.SummaryWriter(log_dir=args.logdir) if args.logdir else None

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_mpjpe
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # os.environ['TORCH_HOME'] = './pre-trained_weights/'
    model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True, num_classes=2, num_keypoints=17, pretrained_backbone=True)

    # Freeze some weights:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.roi_heads.parameters(): # box head, box predictor, keypoint head, keypoint predictor
        param.requires_grad = True

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_mpjpe = checkpoint['best_mpjpe']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data augmentation code
    image_transform_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomApply([transforms.Lambda(jpeg_compression)], p=0.1),
        transforms.RandomGrayscale(p=0.01),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # RCNN already does this internally!
    ])
    image_transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # RCNN already does this internally!
    ])
    image_and_annotation_transform_train = RandomHorizontalFlip(p=0.5)
    image_and_annotation_transform_test = None

    if args.save_results == 'frames':
        train_dataset = IKEAKeypointRCNNTestDataset(args.data, split='train', cam=args.camera_id, image_transform=image_transform_test)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
        save_results(train_loader, model, args)

        test_dataset = IKEAKeypointRCNNTestDataset(args.data, split='test', cam=args.camera_id, image_transform=image_transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
        save_results(test_loader, model, args)
        return
    elif args.save_results == 'video':
        # from moviepy.editor import VideoFileClip
        # from moviepy.video.io.VideoFileClip import VideoFileClip
        train_dataset = IKEAKeypointRCNNVideoTestDataset(args.data, split='train', cam=args.camera_id)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        save_results_video(train_loader, model, args)

        test_dataset = IKEAKeypointRCNNVideoTestDataset(args.data, split='test', cam=args.camera_id)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        save_results_video(test_loader, model, args)
        return

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')

    train_dataset = IKEAKeypointRCNNDataset(traindir, image_transform=image_transform_train, preprocess=image_and_annotation_transform_train)
    print(f"Number of train images: {len(train_dataset)}")

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)

    val_dataset = IKEAKeypointRCNNDataset(valdir, image_transform=image_transform_test, preprocess=image_and_annotation_transform_test)
    print(f"Number of test images: {len(val_dataset)}")

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size_val, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

    if args.evaluate:
        epoch = 0
        validate(val_loader, model, epoch, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, optimizer, epoch, args)

        # evaluate on validation set
        mpjpe = validate(val_loader, model, epoch, args)

        # remember best mpjpe and save checkpoint
        is_best = mpjpe < best_mpjpe
        best_mpjpe = min(mpjpe, best_mpjpe)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_mpjpe': best_mpjpe,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.logdir, f"checkpoint_{epoch}.pth.tar")


def train(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    loss_classifier = AverageMeter('loss_classifier', ':.4f')
    loss_box_reg = AverageMeter('loss_box_reg', ':.4f')
    loss_keypoint = AverageMeter('loss_keypoint', ':.4f')
    loss_objectness = AverageMeter('loss_objectness', ':.4f')
    loss_rpn_box_reg = AverageMeter('loss_rpn_box_reg', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, loss_keypoint, loss_classifier, loss_objectness, loss_box_reg, loss_rpn_box_reg],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            for j in range(len(images)):
                images[j] = images[j].cuda(args.gpu, non_blocking=True)
        for j in range(len(targets)):
            targets[j]["boxes"] = targets[j]["boxes"].cuda(args.gpu, non_blocking=True)
            targets[j]["labels"] = targets[j]["labels"].cuda(args.gpu, non_blocking=True)
            targets[j]["keypoints"] = targets[j]["keypoints"].cuda(args.gpu, non_blocking=True)

        output = model(images, targets)

        loss = output["loss_keypoint"] + output["loss_classifier"] + output["loss_objectness"] + output["loss_box_reg"] + output["loss_rpn_box_reg"]

        losses.update(loss.item(), len(images))
        loss_keypoint.update(output["loss_keypoint"].item(), len(images))
        loss_classifier.update(output["loss_classifier"].item(), len(images))
        loss_objectness.update(output["loss_objectness"].item(), len(images))
        loss_box_reg.update(output["loss_box_reg"].item(), len(images))
        loss_rpn_box_reg.update(output["loss_rpn_box_reg"].item(), len(images))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        if args.writer:
            global_step = epoch * len(train_loader) + i
            args.writer.add_scalar('loss_train', losses.val, global_step=global_step)
            args.writer.add_scalar('loss_keypoint_train', loss_keypoint.val, global_step=global_step)
            args.writer.add_scalar('loss_classifier_train', loss_classifier.val, global_step=global_step)
            args.writer.add_scalar('loss_objectness_train', loss_objectness.val, global_step=global_step)
            args.writer.add_scalar('loss_box_reg_train', loss_box_reg.val, global_step=global_step)
            args.writer.add_scalar('loss_rpn_box_reg_train', loss_rpn_box_reg.val, global_step=global_step)


def validate(val_loader, model, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    mpjpe_meter = AverageMeter('MPJPE', ':.1f')
    pck_meter = AverageMeter('PCK', ':.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, mpjpe_meter, pck_meter],
        prefix='Test: ')

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, targets) in enumerate(val_loader):
            if args.gpu is not None:
                for j in range(len(images)):
                    images[j] = images[j].cuda(args.gpu, non_blocking=True)
            for j in range(len(targets)):
                # targets[j]["boxes"] = targets[j]["boxes"].cuda(args.gpu, non_blocking=True)
                # targets[j]["labels"] = targets[j]["labels"].cuda(args.gpu, non_blocking=True)
                targets[j]["keypoints"] = targets[j]["keypoints"].cuda(args.gpu, non_blocking=True)
            ids = [target["id"] for target in targets]

            output = model(images)

            for j, output in enumerate(output):
                keypoints_gt = targets[j]["keypoints"]
                confidences_gt = keypoints_gt[:,:,2].squeeze() # 17

                keypoints = output["keypoints"] # Most confident detection first
                # label = output["labels"] # Most confident detection first
                # score = output["scores"] # Most confident detection first
                if len(keypoints) > 0: # Skip if no keypoints detected
                    errors = torch.norm(keypoints[:,:,:2] - keypoints_gt[:,:,:2], dim=2) # Kx17 per joint errors
                    best_index = errors.sum(dim=1).argmin()
                    errors = errors[best_index, :] # 17
                    visibility = keypoints[best_index,:,2] # 17
                    # print(f'best_index: {best_index}')

                    # If using small dataset, scale keypoints by 3 to get original coordinates
                    if 'small' in args.data:
                        errors *= args.image_scale_factor

                    # MPJPE & PCK:
                    # Sum up per-joint errors if visible and GT is confident
                    for k, error in enumerate(errors):
                        if confidences_gt[k] == 3 and visibility[k] > 0.0:
                            mpjpe_meter.update(error, 1)

                        # PCK:
                        if confidences_gt[k] == 3:
                            if error <= args.pck_threshold: # In pixels
                                pck_meter.update(1, 1)
                            else:
                                pck_meter.update(0, 1)

                    # do_plot = True
                    do_plot = False
                    if do_plot:
                        plot_skeleton(images[j].cpu(), keypoints[best_index, :, :])
                else:
                    # PCK:
                    num_gt_joints = (confidences_gt == 3).sum().item() # confident GT only
                    pck_meter.update(0, num_gt_joints)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        if args.writer:
            args.writer.add_scalar('mpjpe_val', mpjpe_meter.avg, global_step=epoch)
            args.writer.add_scalar('pck_val', pck_meter.avg, global_step=epoch)

    return mpjpe_meter.avg

def save_results(loader, model, args):
    progress = ProgressMeter(len(loader), [], prefix='Test: ')
    model.eval()
    with torch.no_grad():
        for i, (images, image_paths) in enumerate(loader):
            if args.gpu is not None:
                for j in range(len(images)):
                    images[j] = images[j].cuda(args.gpu, non_blocking=True)

            outputs = model(images)

            # Plots:
            # do_plot = True
            do_plot = False
            if do_plot:
                import visualize_maskrcnn_predictions as vis_preds
                top_predictions = vis_preds.select_top_predictions(outputs[0], 0.7)
                top_predictions = {k:v.cpu() for k, v in top_predictions.items()}
                cv_img = (images[0].cpu().numpy().transpose((1, 2, 0)) * 255).astype("uint8")
                result = cv_img.copy()
                result = vis_preds.overlay_boxes(result, top_predictions)
                result = vis_preds.overlay_keypoints(result, top_predictions)
                result = vis_preds.overlay_class_names(result, top_predictions)
                plt.imshow(result)
                plt.show()

            for j, output in enumerate(outputs):
                keypoints_all = output["keypoints"].cpu() # Nx17x3 (most confident detection first)
                keypoints_scores_all = output["keypoints_scores"].cpu() # Nx17 (most confident detection first)
                boxes = output["boxes"].cpu() # Nx4
                labels = output["labels"].cpu() # N
                scores = output["scores"].cpu() # N
                image_path = image_paths[j]

                # If using small dataset, scale keypoints by 3 to get original coordinates
                if 'small' in args.data:
                    keypoints_all[:, :, :2] *= args.image_scale_factor

                # Rearrange into dictionary for writing:
                # OPENPOSE ANNOTATION STYLE
                people = []
                for k, keypoints in enumerate(keypoints_all):
                    if labels[k] == 1:
                        keypoint_visibility = keypoints[:, 2].tolist()
                        keypoints_scores = keypoints_scores_all[k, :]
                        keypoints[:, 2] = keypoints_scores # Swap visibility and keypoint scores
                        keypoints = keypoints.reshape(-1).tolist()
                        box = boxes[k].tolist()
                        score = scores[k].item()
                        person = {"person_id": [-1], "pose_keypoints_2d": keypoints, "keypoint_visibility": keypoint_visibility, "boxes": box, "score": score}
                        people.append(person)
                if not people: # Output all zero if no detection
                    people = [{"person_id": [-1], "pose_keypoints_2d": [0.]*17*3, "keypoint_visibility": [0.]*17, "boxes": [0.]*4, "score": 0.}]
                output_dict = {}
                output_dict["format"] = "ikea"
                output_dict["people"] = people

                image_path_split = image_path.split('/') # eg <root>/Lack_TV_Bench/0007_white_floor_08_04_2019_08_28_10_47/dev3/images/000100.png
                furniture_type = image_path_split[-5]
                experiment_id = image_path_split[-4]
                cam_id = image_path_split[-3]
                image_filename = image_path_split[-1]
                frame_str, ext = os.path.splitext(image_filename)
                json_name = f"scan_video_000000{frame_str}_keypoints.json"

                # output_path = os.path.join(args.out_data_dir, furniture_type, experiment_id, cam_id, 'predictions', 'pose2d', 'keypoint_rcnn_pt')
                output_path = os.path.join(args.out_data_dir, furniture_type, experiment_id, cam_id, 'predictions', 'pose2d', 'keypoint_rcnn_ft')
                os.makedirs(output_path, exist_ok=True)
                json_file = os.path.join(output_path, json_name)
                print(f"Writing: {json_file}")
                with open(json_file, 'w') as f:
                    json.dump(output_dict, f)

            if i % args.print_freq == 0:
                progress.display(i)

def save_results_video(loader, model, args):
    from moviepy.video.io.VideoFileClip import VideoFileClip
    model.eval()
    with torch.no_grad():
        for i, video_paths in enumerate(loader): # Must be a batch size of 1 (video)
            print(f"Processing {i} of {len(loader)}: {video_paths[0]}")
            clip = VideoFileClip(video_paths[0])
            for frame_id, frame in enumerate(clip.iter_frames()): # HxWx3 numpy array
                frame = transforms.ToTensor()(frame)
                if args.gpu is not None:
                    images = [frame.cuda(args.gpu, non_blocking=True)]

                outputs = model(images)

                # Plots:
                # do_plot = True
                do_plot = False
                if do_plot:
                    import visualize_maskrcnn_predictions as vis_preds
                    top_predictions = vis_preds.select_top_predictions(outputs[0], 0.7)
                    top_predictions = {k:v.cpu() for k, v in top_predictions.items()}
                    cv_img = (images[0].cpu().numpy().transpose((1, 2, 0)) * 255).astype("uint8")
                    result = cv_img.copy()
                    result = vis_preds.overlay_boxes(result, top_predictions)
                    result = vis_preds.overlay_keypoints(result, top_predictions)
                    result = vis_preds.overlay_class_names(result, top_predictions)
                    plt.imshow(result)
                    plt.show()

                for j, output in enumerate(outputs):
                    keypoints_all = output["keypoints"].cpu() # Nx17x3 (most confident detection first)
                    keypoints_scores_all = output["keypoints_scores"].cpu() # Nx17 (most confident detection first)
                    boxes = output["boxes"].cpu() # Nx4
                    labels = output["labels"].cpu() # N
                    scores = output["scores"].cpu() # N
                    video_path = video_paths[j]

                    # If using small dataset, scale keypoints by 3 to get original coordinates
                    if 'small' in args.data:
                        keypoints_all[:, :, :2] *= args.image_scale_factor

                    # Rearrange into dictionary for writing:
                    # OPENPOSE ANNOTATION STYLE
                    people = []
                    for k, keypoints in enumerate(keypoints_all):
                        if labels[k] == 1:
                            keypoint_visibility = keypoints[:, 2].tolist()
                            keypoints_scores = keypoints_scores_all[k, :]
                            keypoints[:, 2] = keypoints_scores # Swap visibility and keypoint scores
                            keypoints = keypoints.reshape(-1).tolist()
                            box = boxes[k].tolist()
                            score = scores[k].item()
                            person = {"person_id": [-1], "pose_keypoints_2d": keypoints, "keypoint_visibility": keypoint_visibility, "boxes": box, "score": score}
                            people.append(person)
                    if not people: # Output all zero if no detection
                        people = [{"person_id": [-1], "pose_keypoints_2d": [0.]*17*3, "keypoint_visibility": [0.]*17, "boxes": [0.]*4, "score": 0.}]
                        print(f"No people detected in frame {frame_id} of {video_path}")
                    output_dict = {}
                    output_dict["format"] = "ikea"
                    output_dict["people"] = people

                    video_path_split = video_path.split('/') # eg <root>/Lack_TV_Bench/0007_white_floor_08_04_2019_08_28_10_47/dev3/images/scan_video.avi
                    furniture_type = video_path_split[-5]
                    experiment_id = video_path_split[-4]
                    cam_id = video_path_split[-3]
                    json_name = f"scan_video_000000{frame_id:06d}_keypoints.json"

                    output_path = os.path.join(args.out_data_dir, furniture_type, experiment_id, cam_id, 'predictions', 'pose2d', 'keypoint_rcnn_ft_all')
                    os.makedirs(output_path, exist_ok=True)
                    json_file = os.path.join(output_path, json_name)
                    # print(f"Writing: {json_file}")
                    with open(json_file, 'w') as f:
                        json.dump(output_dict, f)

            clip.close()

def save_checkpoint(state, is_best, path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(path, filename))
    if is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def collate_fn(batch):
        return tuple(zip(*batch))


def plot_skeleton(image, keypoints, boxes=None, normalized=True):
    if isinstance(image, torch.Tensor):
        if normalized:
            mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)
            std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1)
            image = (image * std) + mean # Unnormalize image
        image = image.numpy().transpose((1, 2, 0))
    plt.imshow(image)
    positions = keypoints[:, :2]
    visibility = keypoints[:, 2]
    for joint_index, position in enumerate(positions):
        if visibility[joint_index] > 0.0:
            plt.scatter(position[0].item(), position[1].item(), c='w')
    for limb in get_ikea_connectivity():
        if visibility[limb[0]] > 0.0 and visibility[limb[1]] > 0.0:
            plt.plot([positions[limb[0], 0].item(), positions[limb[1], 0].item()], [positions[limb[0], 1].item(), positions[limb[1], 1].item()], c='w')
    if isinstance(boxes, torch.Tensor):
        plt.plot([boxes[0], boxes[2]], [boxes[1], boxes[1]], c='w')
        plt.plot([boxes[0], boxes[2]], [boxes[3], boxes[3]], c='w')
        plt.plot([boxes[0], boxes[0]], [boxes[1], boxes[3]], c='w')
        plt.plot([boxes[2], boxes[2]], [boxes[1], boxes[3]], c='w')
    plt.show()


# Data augmentation code:

def jpeg_compression(image):
    f = io.BytesIO()
    image.save(f, 'jpeg', quality=50)
    return Image.open(f)

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, annotations):
        if random.random() > self.p:
            image = transforms.functional.hflip(image)
            # Handle keypoints:
            annotations = copy.deepcopy(annotations)
            annotations["keypoints"][:, :, 0] = float(image.width) - annotations["keypoints"][:, :, 0] - 1.0 # 1x17x3 tensor
            # Flip keypoints:
            joint_names = get_ikea_joint_names()
            joint_names_hflip = get_ikea_joint_hflip_names()
            keypoints = torch.zeros_like(annotations["keypoints"])
            for source_i in range(annotations["keypoints"].size(1)):
                xyc = annotations["keypoints"][:, source_i, :]
                source_name = joint_names[source_i]
                target_name = joint_names_hflip.get(source_name)
                if target_name:
                    target_i = joint_names.index(target_name)
                else:
                    target_i = source_i
                keypoints[:, target_i, :] = xyc
            annotations["keypoints"] = keypoints
            # Handle boxes:
            x1 = annotations["boxes"][:, 0].item()
            x2 = annotations["boxes"][:, 2].item()
            # Swap x1 and x2
            annotations["boxes"][:, 0] = float(image.width) - x2 - 1.0 # 1x4 tensor
            annotations["boxes"][:, 2] = float(image.width) - x1 - 1.0 # 1x4 tensor
            # plot_skeleton(image, annotations["keypoints"].squeeze(), annotations["boxes"].squeeze(), normalized=False)
        return image, annotations

if __name__ == '__main__':
    main()
