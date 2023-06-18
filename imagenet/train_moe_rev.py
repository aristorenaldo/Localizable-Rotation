
import os
import random
import time
import warnings
import sys
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms


from utils import load_tinyimagenet_dataset, accuracy, AverageMeter, SslTransform, TBLog
from torch.utils.tensorboard import SummaryWriter

import models
from config_utils import ConfigObj

def save_checkpoint(args, state, is_best):
    # filename = "%s/%s/ckpt.pth.tar" % (args.root_model, args.store_name)
    filename = os.path.join(args.save, 'ckpt.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace("pth.tar", "best.pth.tar"))

def main(args):

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    else:
        raise Exception(f'{args.save} is exist, please change the save_name or delete')
    if not os.path.exists(args.tb_dir):
        os.makedirs(args.tb_dir)
    if not os.path.isdir(args.save):
        raise Exception('%s is not a dir' % args.save)
   

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


    # ceate model
    print("=> creating model '{}'".format(args.arch))
    model =  models.__dict__[args.arch](backbone=args.backbone)
    
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                args.momentum,
                                weight_decay=args.weight_decay)
    
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 3, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0, verbose=True)


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

    # mean = [x / 255 for x in [127.5, 127.5, 127.5]] # 0.5
    # std = [x / 255 for x in [127.5, 127.5, 127.5]] # 0.5

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        SslTransform(args.arch),
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
    log_training = open(os.path.join(args.save, 'log_train.txt'), 'w')
    with open(os.path.join(args.save, 'args.txt'), 'w') as f:
        f.write(str(args))
    # tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    tb_log = TBLog(args.tb_dir, args.save_name)

    for epoch in range(args.start_epoch, args.epochs):

        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

        # train and validate 1 epoch
        train(train_loader, model, criterion, optimizer, epoch, args, log_training, tb_log)
        lr_scheduler.step()
        acc1 = validate(val_loader, model, criterion, epoch, args, log_training, tb_log)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        tb_log.update({'acc/test_top1_best': best_acc1}, epoch)
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
    
    log_training.close()

def train(train_loader, model, criterion, optimizer, epoch, args, log, tb_log):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to train mode
    model.train()

    end = time.time()
    output=''
    for b_i, ((input, ssl_lb), target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        ssl_lb = torch.stack(ssl_lb).cuda(args.gpu) if isinstance(ssl_lb, (tuple, list)) else ssl_lb.cuda(args.gpu) 

        # output, rot_output, flip_output, sc_output, gate = model(input)

        # gate = F.softmax(gate, dim=1).mean(dim=0)
        # # loss classifier
        # loss = criterion(output, target)
        # ssl_loss = (
        #     CE(rot_output, ssl_lb[0]) * gate[0].item()+
        #     CE(flip_output, ssl_lb[1]) * gate[1].item()+
        #     CE(sc_output, ssl_lb[2]) * gate[2].item()
        # )
        # gate_dict = {}
        if args.arch == "Moe1": # use rot flip sc
            sup_output, rot_output, flip_output, sc_output, gate_output = model(input)
            gate = F.softmax(gate_output, dim=1).mean(dim=0)
            ssl_loss = (F.cross_entropy(rot_output, ssl_lb[0], reduction='mean') * gate[0].item()+
                        F.cross_entropy(flip_output, ssl_lb[1], reduction='mean') * gate[1].item()+
                        F.cross_entropy(sc_output, ssl_lb[2], reduction='mean') * gate[2].item())
            # gate_dict['gate'] = {
            #     'rot': gate[0].detach(),
            #     'flip': gate[1].detach(),
            #     'sc': gate[2].detach()
            # }
            
        if args.arch == "Moe1flip": # use rot flip sc
            sup_output, rot_output, flip_output, gate_output = model(input)
            gate = F.softmax(gate_output, dim=1).mean(dim=0)
            ssl_loss = (F.cross_entropy(rot_output, ssl_lb[0], reduction='mean') * gate[0].item()+
                        F.cross_entropy(flip_output, ssl_lb[1], reduction='mean') * gate[1].item())
            # gate_dict['gate'] = {
            #     'rot': gate[0].detach(),
            #     'flip': gate[1].detach(),
            # }

        if args.arch == "Moe1sc": # use rot flip sc
            sup_output, rot_output, sc_output, gate_output = model(input)
            gate = F.softmax(gate_output, dim=1).mean(dim=0)
            ssl_loss = (F.cross_entropy(rot_output, ssl_lb[0], reduction='mean') * gate[0].item()+
                        F.cross_entropy(sc_output, ssl_lb[1], reduction='mean') * gate[1].item())
            # gate_dict['gate'] = {
            #     'rot': gate[0].detach(),
            #     'sc': gate[1].detach()
            # }

        elif args.arch == "Nomoe": # use 2 task
            sup_output, rot_output, flip_output, sc_output = model(input)
            ssl_loss = (F.cross_entropy(rot_output, ssl_lb[0], reduction='mean') +
                        F.cross_entropy(flip_output, ssl_lb[1], reduction='mean') +
                        F.cross_entropy(sc_output, ssl_lb[2], reduction='mean') )
            
        elif args.arch == "Lorot":
            sup_output, rot_output = model(input)
            ssl_loss = F.cross_entropy(rot_output, ssl_lb, reduction='mean')
        
        elif args.arch == 'vanilla':
            sup_output = model(input)
            ssl_loss = 0
        
        loss = criterion(sup_output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(sup_output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        loss = loss + args.ssl_ratio * ssl_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log gating
        # tb_log.update(gate_dict, epoch*len(train_loader) + b_i)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if b_i % args.print_freq == 0 or b_i == (len(train_loader)-1):
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, b_i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))  # TODO
            print('\r'+output, end='')
            log.write(output + '\n')
            log.flush()
    print()
    # if args.use_gating:
    #     output = f"Epoch: {epoch}, Gated Network Weight Gate = "
    #     for i in range(0, gate.shape[0]):
    #         output += f"[{i}]:{gate[i].item():.2f} "
    #     print(output)
    #     log.write(output + '\n')
    #     log.flush()

    tb_dict = {}
    tb_dict['loss/train'] = losses.avg
    tb_dict['loss/train'] = top1.avg
    tb_dict['acc/train_top5'] = top5.avg
    tb_dict['lr'] = optimizer.param_groups[-1]['lr']
    tb_log.update(tb_dict, epoch)

    # tf_writer.add_scalar('loss/train', losses.avg, epoch)
    # tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    # tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    # tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

def validate(val_loader, model, criterion, epoch, args, log=None, tb_log=None, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to evaluate mode
    model.eval()

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


            if i % args.print_freq == 0 or i == (len(val_loader)-1):
                output_log = ('Test: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print('\r'+output_log, end='')
        print()

        output_log = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                .format(flag=flag, top1=top1, top5=top5, loss=losses))
        print(output_log)
        if log is not None:
            log.write(output_log + '\n')
            log.flush()

        tb_dict = {
            'loss/test_'+ flag: losses.avg,
            'acc/test_' + flag + '_top1': top1.avg,
            'acc/test_' + flag + '_top5': top5.avg
        }

        tb_log.update(tb_dict, epoch)

        # tf_writer.add_scalar('loss/test_'+ flag, losses.avg, epoch)
        # tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        # tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)


    return top1.avg

def path_correction(path):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)),path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Adversarial Pertubation Training for Gated SSL')
    parser.add_argument('--exp_str', '-e', type=str, help='exp_str', required=True)
    parser.add_argument('--path', '-p', type=str, help='config path')
    cli_parser = parser.parse_args()

    config = ConfigObj(path_correction('config/tinyimagenet_default.yaml'), cli_parser.path)
    args = config.get()

    assert args.arch in ['Moe1', 'Lorot', 'Nomoe', 'Moe1flip', 'Moe1sc', 'vanilla']
    # set save_name
    args.save_name = f"tinyimagenet_{args.arch}_{cli_parser.exp_str}_sslratio_{int(args.ssl_ratio*100)}"
    args.save = os.path.join(args.save,args.save_name)
    # args.tb_dir = os.path.join(args.tb_dir, args.save_name)

    main(args)