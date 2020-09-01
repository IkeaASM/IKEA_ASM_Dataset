# Author: Yizhak Ben-Shabat (Itzik), 2020
# <sitzikbs at gmail dot com>
# train frame based methods on IKEA ASM dataset
from torch.utils import data
import torch
import torch.nn as nn
import torch.optim as optim
import copy
# from vgg import vgg19
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
import sf_utils as utils
import os
from tensorboardX import SummaryWriter
import argparse

#execute: python3 train.py --arch resnet18 --logdir ./log/ --batch_size 512 --gpu_idx 999 --data_sampler weighted

# This code uses Pytorch pretrained models for ResNet. Alternatively, you can optionally download your own pretrained models
# os.environ['TORCH_HOME'] = '~/pretrained_models/torch/'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default='.../../../ANU_ikea_dataset_smaller_ImageFolder/',
                        help='ImageFolder structure of the dataset')
    parser.add_argument('--arch', type=str, default='resnet18', help='architecture: resnet18|resnet50')
    parser.add_argument('--logdir', type=str, default='./log/debug/', help='training log folder')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--lr_steps', type=int, default=[5, 25, 50], help='steps to reduce learning rate')
    parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU, 999 to use all available gpus')
    parser.add_argument('--refine', action="store_true", help='flag to refine the model')
    parser.add_argument('--refine_epoch', type=int, default=0, help='refine model from this epoch')
    parser.add_argument('--data_sampler', type=str, default='weighted',
                        help='weighted | random: accomodate data imbalance')
    return parser.parse_args()


def make_weights_for_balanced_classes(images, nclasses):
    """ compute the weight per image for the weighted random sampler"""
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def main(opt):

    lr1 = opt.lr
    lr_steps = opt.lr_steps
    gpu_idx = opt.gpu_idx
    batch_size = opt.batch_size
    arch = opt.arch
    logdir = os.path.join(opt.logdir, arch)
    os.makedirs(logdir, exist_ok=True)
    n_epochs = opt.n_epochs

    train_writer = SummaryWriter(os.path.join(logdir, 'train'))
    test_writer = SummaryWriter(os.path.join(logdir, 'test'))

    if not gpu_idx == 999:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)  # non-functional
        torch.cuda.set_device(0)
    device = torch.device("cuda:0" )


    best_prec1 = 0

    res = 256
    center_crop = 224
    train_transform = transforms.Compose([
        transforms.Resize(res),
        transforms.CenterCrop(center_crop),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # because inpus dtype is PIL Image
        transforms.Normalize( mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(res),
        transforms.CenterCrop(center_crop),
        transforms.ToTensor(),
        transforms.Normalize( mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Data loaders
    dataset_path = opt.dataset_path

    training_set = torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'train'), transform=train_transform)
    if opt.data_sampler == 'weighted':
        train_sampler_weights = make_weights_for_balanced_classes(training_set.imgs, len(training_set.classes))
        train_sampler_weights = torch.DoubleTensor(train_sampler_weights)
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_sampler_weights, len(train_sampler_weights))
        train_loader = data.DataLoader(training_set, sampler=train_sampler, batch_size=batch_size, num_workers=8,
                                       pin_memory=True)
    else:
        train_loader = data.DataLoader(training_set, batch_size=batch_size, num_workers=5, pin_memory=True)


    val_set = torchvision.datasets.ImageFolder(os.path.join(dataset_path, 'test'), transform=val_transform)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)

    num_classes = 33
    if opt.arch == 'vgg':
        # use vgg architecture
        # model_ft = vgg19(pretrained=True)  ##### Model Structure Here
        model_ft = models.__dict__['vgg19'](pretrained=True)
        model_ft.classifier[6] = nn.Linear(4096, num_classes)  # change last layer to fit the number of classes
    elif 'resnet' in opt.arch:
        # use resnet architecture
        model_ft = models.__dict__[opt.arch](pretrained=True)
        if opt.arch =='resnet18' or opt.arch =='resnet34':
            model_ft.fc = nn.Linear(512, num_classes)
        elif opt.arch == 'resnet50' or opt.arch == 'resnet101' or opt.arch == 'resnet152':
            model_ft = models.__dict__[opt.arch](pretrained=True)
            model_ft.fc = nn.Linear(2048, num_classes)
    else:
        raise ValueError("unsupported architecture")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model_ft = nn.DataParallel(model_ft)

    if opt.refine:
        if opt.refine_epoch == 0:
            raise ValueError("You set the refine epoch to 0. No need to refine, just retrain.")
        refine_model_filename = os.path.join(logdir, 'classifier{}.pth' .format(opt.refine_epoch))
        model_ft.load_state_dict(torch.load(refine_model_filename))

    model_ft.to(device)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model_ft.parameters(), lr=lr1, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_steps,
                                         gamma=0.1)  # milestones in number of optimizer iterations

    refine_flag = True
    for epoch in range(1, n_epochs):
        if epoch <= opt.refine_epoch and opt.refine and refine_flag:
            scheduler.step()
            continue
        else:
            refine_flag = False
        # adjust_learning_rate(optimizer, epoch, lr1, lr_steps)

        train_fraction_done = 0.0

        test_batchind = -1
        test_fraction_done = 0.0
        test_enum = enumerate(val_loader, 0)

        # train for one epoch
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top3 = utils.AverageMeter()

        model_ft.train()
        optimizer.zero_grad()

        train_num_batch = len(train_loader)
        test_num_batch = len(val_loader)

        for train_batchind, (im_data, im_class) in enumerate(train_loader):

            model_ft.train
            im_data = im_data.to(device)
            im_class = im_class.to(device)
            # batch_size = im_data.shape[0]

            optimizer.zero_grad()
            output = model_ft(im_data)

            # measure accuracy and record loss
            prec1, prec3 = utils.accuracy(output.data.detach(), im_class, topk=(1, 3))
            loss = criterion(output, im_class)
            loss.backward()

            # compute gradient and do SGD step
            optimizer.step()

            losses.update(loss.item(), im_data.size(0))
            top1.update(prec1.item(), im_data.size(0))
            top3.update(prec3.item(), im_data.size(0))

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                epoch, train_batchind + 1, len(train_loader) + 1, loss=losses, top1=top1, top3=top3))
            train_fraction_done = (train_batchind + 1) / train_num_batch
            train_writer.add_scalar('loss', losses.val,
                                    (epoch + train_fraction_done) * train_num_batch * batch_size)
            train_writer.add_scalar('top1', top1.val,
                                    (epoch + train_fraction_done) * train_num_batch * batch_size)
            train_writer.add_scalar('top3', top3.val,
                                    (epoch + train_fraction_done) * train_num_batch * batch_size)

            train_fraction_done = (train_batchind + 1) / train_num_batch

            # evaluate on a fraction of the validation set
            if test_fraction_done <= train_fraction_done and test_batchind+1 < test_num_batch:
                test_losses = utils.AverageMeter()
                test_top1 = utils.AverageMeter()
                test_top3 = utils.AverageMeter()

                # switch to evaluate mode
                model_ft.eval()
                test_batchind, (im_data, im_class) = next(test_enum)
                with torch.no_grad():
                    im_data = im_data.to(device)
                    im_class = im_class.to(device)

                    # compute output
                    output = model_ft(im_data)
                    test_loss = criterion(output, im_class)
                    # measure accuracy and record loss
                    prec1, prec3 = utils.accuracy(output.data, im_class, topk=(1, 3))
                    test_losses.update(test_loss.item(), im_data.size(0))
                    test_top1.update(prec1.item(), im_data.size(0))
                    test_top3.update(prec3.item(), im_data.size(0))
                    print('Test: [{0}/{1}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@3 {top3.val:.3f} ({top3.avg:.3f})\t'
                        .format(
                        test_batchind, len(val_loader), loss=test_losses, top1=test_top1, top3=test_top3))
                    test_writer.add_scalar('loss', test_losses.val, (epoch + train_fraction_done) * train_num_batch * batch_size)
                    test_writer.add_scalar('top1', test_top1.val, (epoch + train_fraction_done) * train_num_batch * batch_size)
                    test_writer.add_scalar('top3', test_top3.val, (epoch + train_fraction_done) * train_num_batch * batch_size)
                    test_writer.add_scalar('lr', optimizer.param_groups[0]['lr'],
                                           (epoch + train_fraction_done) * train_num_batch * batch_size)
                test_fraction_done = (test_batchind + 1) / test_num_batch

        scheduler.step()

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if (epoch + 1) % 2 == 0:
            # save model
            model_tmp = copy.deepcopy(model_ft.state_dict())
            model_ft.load_state_dict(model_tmp)
            torch.save(model_ft.state_dict(), os.path.join(logdir, 'classifier' + str(epoch) + '.pth'))
        if (is_best):
            model_tmp = copy.deepcopy(model_ft.state_dict())
            model_ft.load_state_dict(model_tmp)
            torch.save(model_ft.state_dict(), os.path.join(logdir, 'best_classifier.pth'))



if __name__ == '__main__':
    train_opt = parse_arguments()
    main(train_opt)
