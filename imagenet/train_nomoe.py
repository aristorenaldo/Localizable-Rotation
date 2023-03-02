import argparse
import os
import random
import time
import warnings
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import load_tinyimagenet_dataset, prepare_folders, shuffle_channel, save_checkpoint, accuracy, AverageMeter
from torch.utils.tensorboard import SummaryWriter

import models

model_names = ['Nomoe']
backbone_names = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]

parser = argparse.ArgumentParser(description='PyTorch Tiny-Imagenet Nomoe Training')

parser.add_argument('-a', '--arch', metavar='ARCH', default='Nomoe',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: Nomoe)')
parser.add_argument('-b', '--backbone', metavar='Backbone', default='resnet18', 
                    choices=backbone_names,
                    help='model backbone: ' +
                        ' | '.join(backbone_names) +
                        ' (default: resnet18)')
parser.add_argument('--exp_str', default='0', type=str, 
                    help='number to indicate which experiment it is')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')

parser.add_argument('-bs', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
                    
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--root_log',type=str, default='log')
parser.add_argument('--root_model', type=str, default='checkpoint')
parser.add_argument('--r_ratio', default=0.1, type=float, help='ratio')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

def main():
    args = parser.parse_args()
    args.store_name = '_'.join(['tiny-imagenet', args.arch, args.exp_str])
    args.root_log = args.root_log + '/' + str(int(args.r_ratio * 100))
    prepare_folders(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                        'This will turn on the CUDNN deterministic setting, '
                        'which can slow down your training considerably! '
                        'You may see unexpected behavior when restarting '
                        'from checkpoints.')

    best_acc1 = 0

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    else:
        warnings.warn('Please use GPU for training or modify the code to train without GPU')
        return

    # check transformation


    # ceate model
    print("=> creating model '{}'".format(args.arch))
    model =  models.__dict__[args.arch](backbone=args.backbone)
    
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                args.momentum,
                                weight_decay=args.weight_decay)


    # resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return 
    
    cudnn.benchmark = True

    # data loader

    mean = [x / 255 for x in [127.5, 127.5, 127.5]] # 0.5
    std = [x / 255 for x in [127.5, 127.5, 127.5]] # 0.5

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)
    ])

    train_dataset, val_dataset = load_tinyimagenet_dataset(train_transform, val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=100, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # training log (txt)
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.txt'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        CE = nn.CrossEntropyLoss().cuda(args.gpu)
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

        # train and validate 1 epoch
        train(train_loader, model, criterion, optimizer, epoch, args, log_training, tf_writer, CE)
        acc1 = validate(val_loader, model, criterion, epoch, args, log_training, tf_writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        log_training.write(output_best + '\n')
        log_training.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

def train(train_loader, model, criterion, optimizer, epoch, args, log, tf_writer, CE):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to train mode
    model.train()

    end = time.time()
    output=''
    for b_i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        flip_label = torch.randint(2, size=(input.size(0),))
        sc_label = torch.randint(6, size=(input.size(0),))

        idx = torch.randint(4, size=(input.size(0),))
        idx2 = torch.randint(4, size=(input.size(0),))
        r = input.size(2) // 2
        r2 = input.size(2)
        for i in range(input.size(0)):
            if idx[i] == 0:
                w1 = 0
                w2 = r
                h1 = 0
                h2 = r
            elif idx[i] == 1:
                w1 = 0
                w2 = r
                h1 = r
                h2 = r2
            elif idx[i] == 2:
                w1 = r
                w2 = r2
                h1 = 0
                h2 = r
            elif idx[i] == 3:
                w1 = r
                w2 = r2
                h1 = r
                h2 = r2

            # Fliplr
            if flip_label[i]:
                input[i][:, w1:w2, h1:h2] = torch.fliplr(input[i][:, w1:w2, h1:h2])
            # lorot E
            input[i][:, w1:w2, h1:h2] = torch.rot90(
                    input[i][:, w1:w2, h1:h2], 
                    idx2[i], 
                    [1, 2]
                )
            # shuffle channel
            input[i][:, w1:w2, h1:h2] = shuffle_channel(
                    input[i][:, w1:w2, h1:h2],
                    sc_label[i]
            )

        idx = idx.cuda()
        idx2 = idx2.cuda()
        rot_label = idx * 4 + idx2
        rot_label = rot_label.cuda()
        flip_label = flip_label.cuda()
        sc_label = sc_label.cuda()

        del idx
        del idx2

        # compute output
        # if args.use_gating:
        #     # use moe
        #     if args.use_flip and args.use_sc:
        #         # Moe1
        #         output, rot_output, flip_output, sc_output, gn_output = model(input)
        #     else:
        #         # moe1flip/sc
        #         output, rot_output, f_s_output, gn_output = model(input)
        # else:
        #     # Lorot or vanilla
        #     if args.use_rot:
        #         # lorot
        #         output, rot_output = model(input)
        #     else:
        #         # vanilla
        #         output = model(input)
        output, rot_output, flip_output, sc_output = model(input)

        # loss classifier
        loss = criterion(output, target)

        # compute loss ssl
        # if args.use_gating:
        #     # use moe
        #     rot_loss = CE(rot_output, rot_label)
        #     if args.use_flip and args.use_sc:
        #         # Moe1 all
        #         flip_loss = CE(flip_output, flip_label)
        #         sc_loss = CE(sc_output, sc_label)
        #     elif args.use_flip:
        #         # moe1flip
        #         flip_loss =CE(f_s_output, flip_label)
        #     else:
        #         # moe1sc
        #         sc_loss = CE(f_s_output, sc_label)
        # else:
        #     # Lorot or vanilla
        #     if args.use_rot:
        #         # lorot
        #         rot_loss = CE(rot_output, rot_label)
        rot_loss = CE(rot_output, rot_label)
        flip_loss = CE(flip_output, flip_label)
        sc_loss = CE(sc_output, sc_label)
        
            
        # rot_loss = CE(rot_output, rot_label)
        # flip_loss = CE(flip_output, flip_label)
        # sc_loss = CE(sc_output, sc_label)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        loss_ssl = rot_loss+flip_loss+sc_loss
        # if args.use_gating:
        #     gn_softmax = nn.Softmax()(gn_output.mean(dim=0))
        #     loss_ssl = gn_softmax[0].item() * rot_loss

        #     if args.use_flip and args.use_sc:
        #         # moe all
        #         loss_ssl += (gn_softmax[1].item() * flip_loss
        #                     + gn_softmax[2].item() * sc_loss)
        #     elif args.use_flip:
        #         # moe1flip
        #         loss_ssl += (gn_softmax[1].item() * flip_loss)
        #     else:
        #         # moe1sc
        #         loss_ssl += (gn_softmax[1].item() * sc_loss)
        # else:
        #     # Lorot or vanilla
        #     if args.use_rot:
        #         # lorot
        #         loss_ssl = rot_loss
        #     else:
                # loss_ssl=0
        
        loss = loss + args.r_ratio * loss_ssl

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if b_i % args.print_freq == 0 or b_i == (len(train_loader)-1):
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, b_i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))  # TODO
            print('\r'+output, end='')
            log.write(output + '\n')
            log.flush()
    print()

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

def validate(val_loader, model, criterion, epoch, args, log=None, tf_writer=None, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0 or i == (len(val_loader)-1):
                output = ('Test: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print('\r'+output, end='')
        print()
        # cf = confusion_matrix(all_targets, all_preds).astype(float)
        # cls_cnt = cf.sum(axis=1)
        # cls_hit = np.diag(cf)
        # cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                .format(flag=flag, top1=top1, top5=top5, loss=losses))
        # out_cls_acc = '%s Class Accuracy: %s'%(flag,(np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))
        print(output)
        # print(out_cls_acc)
        if log is not None:
            log.write(output + '\n')
            log.flush()

        tf_writer.add_scalar('loss/test_'+ flag, losses.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
        # tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i):x for i, x in enumerate(cls_acc)}, epoch)

    return top1.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed 0.1 every 75 epochs"""
    last_lr = optimizer.param_groups[-1]['lr']
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch % 75 == 0 :
        lr = last_lr * 0.1
    else:
        lr = last_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()