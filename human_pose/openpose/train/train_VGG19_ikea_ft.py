import argparse
import time
import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.utils.tensorboard as tb

from pdb import set_trace as st

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from lib.network.rtpose_vgg import get_model, use_vgg
from lib.datasets import coco, transforms, datasets
from lib.datasets import dataset_ikea
from lib.config import cfg, update_config
from lib.utils.paf_to_pose import paf_to_pose_cpp

DATA_DIR = '/home/djcam/Documents/datasets/ikea_asm_small'

IMAGE_DIR_TRAIN = os.path.join(DATA_DIR, 'train')
IMAGE_DIR_VAL = os.path.join(DATA_DIR, 'test')


def train_cli(parser):
    group = parser.add_argument_group('dataset and loader')
    group.add_argument('--train-image-dir', default=IMAGE_DIR_TRAIN)
    group.add_argument('--val-image-dir', default=IMAGE_DIR_VAL)
    group.add_argument('--loader-workers', default=8, type=int,
                       help='number of workers for data loading')
    group.add_argument('--batch-size', default=32, type=int,
                       help='batch size')
    group.add_argument('--lr', '--learning-rate', default=1., type=float,
                    metavar='LR', help='initial learning rate')
    group.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    group.add_argument('--weight-decay', '--wd', default=0.000, type=float,
                    metavar='W', help='weight decay (default: 1e-4)') 
    group.add_argument('--nesterov', dest='nesterov', default=True, type=bool)     
    group.add_argument('--print_freq', default=20, type=int, metavar='N',
                    help='number of iterations to print the training statistics')    
                   
                                         
def train_factory(args, preprocess, target_transforms):
    train_datas = [dataset_ikea.IKEAPose2dDataset(
        root=args.train_image_dir,
        preprocess=preprocess,
        image_transform=transforms.image_transform_rtpose_train,
        target_transforms=target_transforms,
    )]

    train_data = torch.utils.data.ConcatDataset(train_datas)
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    val_data = dataset_ikea.IKEAPose2dDataset(
        root=args.val_image_dir,
        preprocess=preprocess,
        image_transform=transforms.image_transform_rtpose,
        target_transforms=target_transforms,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=True)

    return train_loader, val_loader, train_data, val_data

def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    train_cli(parser)
    parser.add_argument('-o', '--output', default=None,
                        help='output file')
    parser.add_argument('--stride-apply', default=1, type=int,
                        help='apply and reset gradients every n batches')
    parser.add_argument('--epochs', default=75, type=int,
                        help='number of epochs to train')
    parser.add_argument('--freeze-base', default=0, type=int,
                        help='number of epochs to train with frozen base')
    parser.add_argument('--pre-lr', type=float, default=1e-4,
                        help='pre learning rate')
    parser.add_argument('--update-batchnorm-runningstatistics',
                        default=False, action='store_true',
                        help='update batch norm running statistics')
    parser.add_argument('--square-edge', default=368, type=int,
                        help='square edge of input images')
    parser.add_argument('--ema', default=1e-3, type=float,
                        help='ema decay constant')
    parser.add_argument('--debug-without-plots', default=False, action='store_true',
                        help='enable debug but dont plot')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')                        
    parser.add_argument('--model_path', default='', type=str, metavar='DIR',
                    help='path to where the model saved')
    parser.add_argument('--log_dir', default='./logs/', type=str,
                    help='path to where the logs are saved')
    parser.add_argument('--resume', default='', type=str,
                    help='path to where checkpoint is saved')
    parser.add_argument('--save_results', dest='save_results', action='store_true',
                    help='evaluate model on training and validation sets and save results')
    parser.add_argument('--out_data_dir', default='/mnt/DE9656C296569B3B/datasets/ikea/ikea_asm/', type=str,
                        help='path to where results will be saved')
    args = parser.parse_args()

    if not args.model_path:
        args.model_path = args.log_dir

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True

    args.writer = tb.SummaryWriter(log_dir=args.log_dir) if args.log_dir else None
        
    return args

args = cli()

print("Loading dataset...")
# load train data
preprocess = transforms.Compose([
        transforms.Normalize(),
        transforms.RandomApply(transforms.HFlip(), 0.5),
        transforms.RescaleRelative(),
        transforms.Crop(args.square_edge), # CHANGE args.square_edge (default 368) to larger for larger images!
        transforms.CenterPad(args.square_edge),
    ])
train_loader, val_loader, train_data, val_data = train_factory(args, preprocess, target_transforms=None)


def build_names():
    names = []
    for j in range(1, 7):
        for k in range(1, 3):
            names.append('loss_stage%d_L%d' % (j, k))
    return names


def get_loss(saved_for_loss, heat_temp, vec_temp):

    names = build_names()
    saved_for_log = OrderedDict()
    criterion = nn.MSELoss(reduction='mean').cuda()
    total_loss = 0

    for j in range(6):
        pred1 = saved_for_loss[2 * j]
        pred2 = saved_for_loss[2 * j + 1] 


        # Compute losses
        loss1 = criterion(pred1, vec_temp)
        loss2 = criterion(pred2, heat_temp) 

        total_loss += loss1
        total_loss += loss2
        # print(total_loss)

        # Get value from Variable and save for log
        saved_for_log[names[2 * j]] = loss1.item()
        saved_for_log[names[2 * j + 1]] = loss2.item()

    saved_for_log['max_ht'] = torch.max(
        saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['min_ht'] = torch.min(
        saved_for_loss[-1].data[:, 0:-1, :, :]).item()
    saved_for_log['max_paf'] = torch.max(saved_for_loss[-2].data).item()
    saved_for_log['min_paf'] = torch.min(saved_for_loss[-2].data).item()

    return total_loss, saved_for_log
         

def train(train_loader, model, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    meter_dict = {}
    for name in build_names():
        meter_dict[name] = AverageMeter()
    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()    
    meter_dict['max_paf'] = AverageMeter()    
    meter_dict['min_paf'] = AverageMeter()
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (img, heatmap_target, paf_target, image_ids) in enumerate(train_loader):
        # measure data loading time
        #writer.add_text('Text', 'text logged at step:' + str(i), i)

        #for name, param in model.named_parameters():
        #    writer.add_histogram(name, param.clone().cpu().data.numpy(),i)        
        data_time.update(time.time() - end)

        img = img.cuda()
        heatmap_target = heatmap_target.cuda()
        paf_target = paf_target.cuda()
        # compute output
        _,saved_for_loss = model(img)
        
        total_loss, saved_for_log = get_loss(saved_for_loss, heatmap_target, paf_target)
        
        for name,_ in meter_dict.items():
            meter_dict[name].update(saved_for_log[name], img.size(0))
        losses.update(total_loss, img.size(0))

        del saved_for_loss
        del saved_for_log
        # torch.cuda.empty_cache()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        del total_loss
        # torch.cuda.empty_cache()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print_string = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(train_loader))
            print_string +='Data time {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(data_time=data_time)
            print_string +='Batch time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(batch_time=batch_time)
            print_string += 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss=losses)

            for name, value in meter_dict.items():
                print_string+='{name}: {loss.val:.4f} ({loss.avg:.4f})\t'.format(name=name, loss=value)
            print(print_string)
        if args.writer:
            global_step = epoch * len(train_loader) + i
            args.writer.add_scalar('loss_train', losses.val, global_step=global_step)
    return losses.avg  
        
        
def validate(val_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    meter_dict = {}
    for name in build_names():
        meter_dict[name] = AverageMeter()
    meter_dict['max_ht'] = AverageMeter()
    meter_dict['min_ht'] = AverageMeter()    
    meter_dict['max_paf'] = AverageMeter()    
    meter_dict['min_paf'] = AverageMeter()
    # switch to train mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (img, heatmap_target, paf_target, image_ids) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            img = img.cuda()
            heatmap_target = heatmap_target.cuda()
            paf_target = paf_target.cuda()
            
            # compute output
            _,saved_for_loss = model(img)
            
            total_loss, saved_for_log = get_loss(saved_for_loss, heatmap_target, paf_target)
                   
            #for name,_ in meter_dict.items():
            #    meter_dict[name].update(saved_for_log[name], img.size(0))
                
            losses.update(total_loss.item(), img.size(0))

            del saved_for_loss
            del saved_for_log
            del total_loss
            # torch.cuda.empty_cache()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()  
            if i % args.print_freq == 0:
                print_string = 'Epoch: [{0}][{1}/{2}]\t'.format(epoch, i, len(val_loader))
                print_string +='Data time {data_time.val:.3f} ({data_time.avg:.3f})\t'.format( data_time=data_time)
                print_string +='Batch time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(batch_time=batch_time)
                print_string += 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss=losses)

                for name, value in meter_dict.items():
                    print_string+='{name}: {loss.val:.4f} ({loss.avg:.4f})\t'.format(name=name, loss=value)
                print(print_string)
    
    if args.writer:
        args.writer.add_scalar('loss_val', losses.avg, global_step=epoch)
    return losses.avg

def save_results(loader, model, args):
    # Get cropped image coordinate conversion factors
    orig_image_w, orig_image_h = loader.dataset.image_size
    padding = int(args.square_edge / 2.0)
    x_offset, y_offset = 0, 0
    if orig_image_w > args.square_edge:
        x_offset = torch.tensor(orig_image_w / 2.0 - padding)
        x_offset = torch.clamp(x_offset, min=0, max=orig_image_w - args.square_edge).item()
    if orig_image_h > args.square_edge:
        y_offset = torch.tensor(orig_image_h / 2.0 - padding)
        y_offset = torch.clamp(y_offset, min=0, max=orig_image_h - args.square_edge).item()
    cropped_image_w = min(args.square_edge, orig_image_w - x_offset)
    cropped_image_h = min(args.square_edge, orig_image_h - y_offset)

    model.eval()
    with torch.no_grad():
        for i, (img, heatmap_target, paf_target, image_ids) in enumerate(loader):
            img = img.cuda()

            outputs, _ = model(img)


            pafs = outputs[-2].cpu().data.numpy().transpose(0, 2, 3, 1)
            heatmaps = outputs[-1].cpu().data.numpy().transpose(0, 2, 3, 1)

            for j, (paf, heatmap) in enumerate(zip(pafs, heatmaps)):
                id = image_ids[j]
                id_tokens = id.split('_') # eg Lack_TV_Bench_0007_white_floor_08_04_2019_08_28_10_47_dev3_000100
                furniture_type = '_'.join(id_tokens[:3])
                experiment_id = '_'.join(id_tokens[3:-2])
                cam_id = id_tokens[-2]
                frame_str = id_tokens[-1]
                json_name = f"scan_video_000000{frame_str}_keypoints.json"

                humans = paf_to_pose_cpp(heatmap, paf, cfg)

                # Save joints in OpenPose format
                out_image_h, out_image_w = 1080, 1920
                people = []
                for i, human in enumerate(humans):
                    keypoints = []
                    for j in range(18):
                        if j == 8:
                            keypoints.extend([0, 0, 0]) # Add extra joint (midhip) to correspond to body_25
                        if j not in human.body_parts.keys():
                            keypoints.extend([0, 0, 0])
                        else:
                            body_part = human.body_parts[j]
                            keypoints.extend([(cropped_image_w * body_part.x + x_offset) * out_image_w / orig_image_w, (cropped_image_h * body_part.y + y_offset) * out_image_h / orig_image_h, body_part.score])
                    person = {
                        "person_id":[i-1],
                        "pose_keypoints_2d":keypoints
                    }
                    people.append(person)
                    
                    # do_plot = True
                    do_plot = False
                    if do_plot:
                        image_path = os.path.join(args.out_data_dir, furniture_type, experiment_id, cam_id, 'images', frame_str + '.png')
                        out_image = Image.open(image_path)
                        keypoints_np = np.array(keypoints).reshape(-1, 3)
                        plot_skeleton(out_image, keypoints_np)
                people_dict = {"people":people}
                people_dict["format"] = "body19"

                output_path = os.path.join(args.out_data_dir, furniture_type, experiment_id, cam_id, 'predictions', 'pose2d', 'openpose_ft')
                os.makedirs(output_path, exist_ok=True)
                json_file = os.path.join(output_path, json_name)
                print(f"Writing: {json_file}")
                with open(json_file, 'w') as f:
                    json.dump(people_dict, f)

def plot_skeleton(image, keypoints):
    plt.imshow(image)
    positions = keypoints[:, :2]
    visibility = keypoints[:, 2]
    for joint_index, position in enumerate(positions):
        if visibility[joint_index] > 0.0:
            plt.scatter(position[0].item(), position[1].item(), c='w')
    # for limb in get_ikea_connectivity():
    #     if visibility[limb[0]] > 0.0 and visibility[limb[1]] > 0.0:
    #         plt.plot([positions[limb[0], 0].item(), positions[limb[1], 0].item()], [positions[limb[0], 1].item(), positions[limb[1], 1].item()], c='w')
    plt.show()

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


###################################################################
start_epoch = 0
best_val_loss = np.inf
model = get_model('vgg19')

if args.resume:
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_loss']
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
else:
    model.load_state_dict(torch.load('pose_model.pth'))
    model = torch.nn.DataParallel(model).cuda()

# Fix weights: model0, model1_1, model1_2, ..., model6_1, model6_2
# for param in model.module.model0.parameters():
#     param.requires_grad = False
# for param in model.module.model1_1.parameters():
#     param.requires_grad = False
# for param in model.module.model1_2.parameters():
#     param.requires_grad = False
# for param in model.module.model2_1.parameters():
#     param.requires_grad = False
# for param in model.module.model2_2.parameters():
#     param.requires_grad = False
# for param in model.module.model3_1.parameters():
#     param.requires_grad = False
# for param in model.module.model3_2.parameters():
#     param.requires_grad = False
# for param in model.module.model4_1.parameters():
#     param.requires_grad = False
# for param in model.module.model4_2.parameters():
#     param.requires_grad = False
# for param in model.module.model5_1.parameters():
#     param.requires_grad = False
# for param in model.module.model5_2.parameters():
#     param.requires_grad = False
# for param in model.module.model6_1.parameters():
#     param.requires_grad = False
# for param in model.module.model6_2.parameters():
#     param.requires_grad = False

trainable_vars = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.SGD(trainable_vars, lr=args.lr,
                           momentum=args.momentum,
                           weight_decay=args.weight_decay,
                           nesterov=args.nesterov)

if args.resume:
    optimizer.load_state_dict(checkpoint['optimizer'])

if args.save_results:
    args.cfg = './experiments/vgg19_368x368_sgd.yaml'
    args.opts = []
    update_config(cfg, args)

    preprocess_test = transforms.Compose([
        transforms.Normalize(),
        transforms.CenterCrop(args.square_edge), # Square centered crop might miss person, but model expects this format
        transforms.CenterPad(args.square_edge),
    ])
    train_data = dataset_ikea.IKEAPose2dDataset(
        root=args.train_image_dir,
        preprocess=preprocess_test,
        image_transform=transforms.image_transform_rtpose, # test time
        target_transforms=None,
    )
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=False)
    val_data = dataset_ikea.IKEAPose2dDataset(
        root=args.val_image_dir,
        preprocess=preprocess_test,
        image_transform=transforms.image_transform_rtpose,
        target_transforms=None,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers, drop_last=False)
    save_results(train_loader, model, args)
    save_results(val_loader, model, args)
    quit()

model_save_filename = os.path.join(args.model_path, 'best_pose.pth')
for epoch in range(start_epoch, args.epochs):
    # train for one epoch
    train_loss = train(train_loader, model, optimizer, epoch)

    # evaluate on validation set
    val_loss = validate(val_loader, model, epoch)

    is_best = val_loss<best_val_loss
    best_val_loss = min(val_loss, best_val_loss)
    if is_best:
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_loss': best_val_loss,
            'optimizer': optimizer.state_dict(),
        }, model_save_filename)
    if epoch % 1 == 0:
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_loss': best_val_loss,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.model_path, f"model_epoch_{epoch:02}.pth"))
          
if args.writer:
    args.writer.close()