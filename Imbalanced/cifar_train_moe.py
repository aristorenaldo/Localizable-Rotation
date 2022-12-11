import argparse
import os
import random
import sys
import time
import warnings

import models
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
from losses import FocalLoss, LDAMLoss
from sklearn.metrics import confusion_matrix
from tensorboardX import SummaryWriter
from utils import *

##2022_CVPR_LoRot-E
model_names = sorted(
    name
    for name in models.__dict__
    if not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch Cifar Training")
parser.add_argument("--dataset", default="cifar10", help="dataset setting")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet32",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet32)",
)
parser.add_argument("--loss_type", default="CE", type=str, help="loss type")
parser.add_argument("--imb_type", default="exp", type=str, help="imbalance type")
parser.add_argument("--imb_factor", default=0.01, type=float, help="imbalance factor")
parser.add_argument(
    "--train_rule",
    default="None",
    type=str,
    help="data sampling strategy for train loader",
)
parser.add_argument(
    "--rand_number", default=0, type=int, help="fix random number for data sampling"
)
parser.add_argument(
    "--exp_str", default="0", type=str, help="number to indicate which experiment it is"
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs", default=200, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b", "--batch-size", default=128, type=int, metavar="N", help="mini-batch size"
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=2e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--print-freq",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--pretrained", dest="pretrained", action="store_true", help="use pre-trained model"
)
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
parser.add_argument("--root_log", type=str, default="log")
parser.add_argument("--root_model", type=str, default="checkpoint")
parser.add_argument("--r_ratio", default=0.1, type=float, help="ratio")
# parser.add_argument(
#     "-m",
#     "--method",
#     type=str,
#     required=True,
#     help="Experiment it will be doing, Full explanation in experiments md, the available value is rot, fliplr, and sc",
# )
best_acc1 = 0


# ya
def main():
    args = parser.parse_args()
    args.store_name = "_".join(
        [
            args.dataset,
            args.arch,
            args.loss_type,
            args.train_rule,
            args.imb_type,
            str(args.imb_factor),
            args.exp_str,
        ]
    )
    # args.method = args.method.split(" ")
    args.root_log = args.root_log + "/" + str(int(args.r_ratio * 100))
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if args.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 100 if args.dataset == "cifar100" else 10
    use_norm = True if args.loss_type == "LDAM" else False
    model = models.__dict__[args.arch](num_classes=num_classes, use_norm=use_norm)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location="cuda:0")
            args.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    if args.dataset == "cifar10":
        train_dataset = IMBALANCECIFAR10(
            root="./data",
            imb_type=args.imb_type,
            imb_factor=args.imb_factor,
            rand_number=args.rand_number,
            train=True,
            download=True,
            transform=transform_train,
        )
        val_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_val
        )
    elif args.dataset == "cifar100":
        train_dataset = IMBALANCECIFAR100(
            root="./data",
            imb_type=args.imb_type,
            imb_factor=args.imb_factor,
            rand_number=args.rand_number,
            train=True,
            download=True,
            transform=transform_train,
        )
        val_dataset = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform_val
        )
    else:
        warnings.warn("Dataset is not listed")
        return
    cls_num_list = train_dataset.get_cls_num_list()
    print("cls num list:")
    print(cls_num_list)
    args.cls_num_list = cls_num_list

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # init log for training

    log_training = open(
        os.path.join(args.root_log, args.store_name, "log_train.csv"), "w"
    )
    log_testing = open(
        os.path.join(args.root_log, args.store_name, "log_test.csv"), "w"
    )
    with open(os.path.join(args.root_log, args.store_name, "args.txt"), "w") as f:
        f.write(str(args))
    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        if args.train_rule == "None":
            train_sampler = None
            per_cls_weights = None
        elif args.train_rule == "Resample":
            train_sampler = ImbalancedDatasetSampler(train_dataset)
            per_cls_weights = None
        elif args.train_rule == "Reweight":
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = (
                per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            )
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == "DRW":
            train_sampler = None
            idx = epoch // 160
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = (
                per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            )
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        else:
            warnings.warn("Sample rule is not listed")
        CE = nn.CrossEntropyLoss().cuda(args.gpu)
        if args.loss_type == "CE":
            criterion = nn.CrossEntropyLoss().cuda(args.gpu)
            # criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == "LDAM":
            criterion = LDAMLoss(
                cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights
            ).cuda(args.gpu)
            # CE = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == "Focal":
            criterion = FocalLoss(weight=per_cls_weights, gamma=1).cuda(args.gpu)
        else:
            warnings.warn("Loss type is not listed")
            return

        # train for one epoch
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            args,
            log_training,
            tf_writer,
            CE,
        )

        # evaluate on validation set
        acc1 = validate(
            val_loader, model, criterion, epoch, args, log_testing, tf_writer
        )

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar("acc/test_top1_best", best_acc1, epoch)
        output_best = "Best Prec@1: %.3f\n" % (best_acc1)
        print(output_best)
        log_testing.write(output_best + "\n")
        log_testing.flush()

        save_checkpoint(
            args,
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_acc1": best_acc1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
        )


def train(train_loader, model, criterion, optimizer, epoch, args, log, tf_writer, CE):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    # switch to train mode
    model.train()

    end = time.time()
    for b_i, (input_image, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input_image = input_image.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        region_label = torch.randint(4, size=(input_image.size(0),))
        
        # idx_rotation = torch.randint(4, size=(input_image.size(0),))
        flip_label = torch.randint(4, size=(input_image.size(0),))
        
        sc_label= torch.randint(6, size=(input_image.size(0),))
        
        r = input_image.size(2) // 2
        r2 = input_image.size(2) 
        # regi, fliplabel, sclabel = 0, 0, 0
        for i in range(input_image.size(0)):
            if region_label[i] == 0:
                w1 = 0
                w2 = r
                h1 = 0
                h2 = r
            elif region_label[i] == 1:
                w1 = 0
                w2 = r
                h1 = r
                h2 = r2
            elif region_label[i] == 2:
                w1 = r
                w2 = r2
                h1 = 0
                h2 = r
            elif region_label[i] == 3:
                w1 = r
                w2 = r2
                h1 = r
                h2 = r2
            input_image[i][:, w1:w2, h1:h2] = flip_image(
                    input_image[i][:, w1:w2, h1:h2],
                    flip_label[i]
                )
            input_image[i][:, w1:w2, h1:h2] = shuffle_channel(
                    input_image[i][:, w1:w2, h1:h2],
                    sc_label[i]
                )
            # if "fliplr" in args.method:
            #     input_image[i][:, w1:w2, h1:h2] = torch.fliplr(
            #         input_image[i][:, w1:w2, h1:h2]
            #     )
            #     fliplabel = idx * 4
            #     fliplabel = fliplabel.cuda()
            # if "rot" in args.method:
            #     input_image[i][:, w1:w2, h1:h2] = torch.rot90(
            #         input_image[i][:, w1:w2, h1:h2], idx_rotation[i], [1, 2]
            #     )
            #     rotlabel = idx * 4 + idx_rotation
            #     rotlabel = rotlabel.cuda()
            # if "sc" in args.method:
            #     input_image[i][:, w1:w2, h1:h2] = shuffle_channel(
            #         input_image[i][:, w1:w2, h1:h2], idx_shuffle_channel[i]
            #     )
            #     sclabel = idx * 4 + idx_shuffle_channel
            #     sclabel = sclabel.cuda()

        region_label = region_label.cuda()
        flip_label = flip_label.cuda()
        sc_label = sc_label.cuda()
        # rot_output, flip_output, sc_output = 0, 0, 0
        # rotloss, fliploss, scloss = 0, 0, 0
        # compute output
        if args.arch.startswith('Moe'):
            output, region_output, flip_output, sc_output, gn_output = model(input_image)
        else:
            output, region_output, flip_output, sc_output = model(input_image)
        # output,  flip_output, sc_output, gn_output = model(input_image)
        # output, rot_output, gn_output = model(input_image)

        
        # gn_sigmoid = nn.Sigmoid()(gn_output.mean(dim=0))
        loss = criterion(output, target)
        region_loss = CE(region_output, region_label)
        flip_loss = CE(flip_output, flip_label)
        sc_loss = CE(sc_output, sc_label)
        # rot_output = model(input_image, rot=True)
        # if "rot" in args.method:
        #     rot_output = torch.argmax(rot_output, axis=1)
        #     rotloss = CE(rot_output.type(torch.float32), rotlabel.type(torch.float32))
        # if "fliplr" in args.method:
        #     flip_output = torch.argmax(flip_output, axis=1)
        #     fliploss = CE(
        #         flip_output.type(torch.float32), fliplabel.type(torch.float32)
        #     )
        # if "sc" in args.method:
        #     sc_output = torch.argmax(sc_output, axis=1)
        #     scloss = CE(sc_output.type(torch.float32), sclabel.type(torch.float32))

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input_image.size(0))
        top1.update(acc1[0], input_image.size(0))
        top5.update(acc5[0], input_image.size(0))

        

        if args.arch == 'Moe1':
            gn_softmax = nn.Softmax()(gn_output.mean(dim=0))
            loss = loss + args.r_ratio * (
                gn_softmax[0].item() * region_loss +
                #  gn_softmax[1].item() * fliploss
                + gn_softmax[1].item() * flip_loss
                + gn_softmax[2].item() * sc_loss
            )
        elif args.arch == 'Moe2':
            gn_softmax = nn.Softmax()(gn_output.mean(dim=0))
            loss = (
                gn_softmax[0].item() * loss
                + gn_softmax[1].item() * region_loss 
                + gn_softmax[2].item() * flip_loss
                + gn_softmax[3].item() * sc_loss
            )
        else:
            loss = loss + args.r_ratio * (
                region_loss
                + flip_loss
                + sc_loss
            )
            # loss = (
            #     gn_softmax[0].item() * loss
            #     + gn_softmax[1].item() * rotloss
            #     # + gn_softmax[2].item() * fliploss
            #     # + gn_softmax[3].item() * scloss
            # )
            # sigmoid
            # loss = (
            #     loss
            #     + gn_sigmoid[0] * rotloss 
            #     + gn_sigmoid[1] * fliploss
            #     + gn_sigmoid[2] * scloss
            # )

            # loss = (
            #     loss
            #     # + args.r_ratio * rotloss 
            #     + args.r_ratio * fliploss
            #     + args.r_ratio * scloss
            # )

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if b_i % args.print_freq == 0 or b_i == (len(train_loader)-1):
            output = (
                "Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                    epoch,
                    b_i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                    lr=optimizer.param_groups[-1]["lr"] * 0.1,
                )
            )  # TODO
            print('\r'+output, end='')
            log.write(output + "\n")
            log.flush()
    print()
    if args.arch == 'Moe1':
        output = f"Gated Network Weight Gate= Region:{gn_softmax[0].item():.2f}, Flip:{gn_softmax[1].item():.2f}, Sc:{gn_softmax[2].item():.2f}"
    else:
        if args.arch == 'Moe2':
            output = f"Gated Network Weight Gate= FC:{gn_softmax[0].item():.2f}, Region:{gn_softmax[1].item():.2f}, Flip:{gn_softmax[2].item():.2f}, Sc:{gn_softmax[3].item():.2f}"
        print(output)
    log.write(output + "\n")
    log.flush()

    tf_writer.add_scalar("loss/train", losses.avg, epoch)
    tf_writer.add_scalar("acc/train_top1", top1.avg, epoch)
    tf_writer.add_scalar("acc/train_top5", top5.avg, epoch)
    tf_writer.add_scalar("lr", optimizer.param_groups[-1]["lr"], epoch)


def validate(
    val_loader, model, criterion, epoch, args, log=None, tf_writer=None, flag="val"
):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    # switch to evaluate mode
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, (input_image, target) in enumerate(val_loader):
            if args.gpu is not None:
                input_image = input_image.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input_image)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input_image.size(0))
            top1.update(acc1[0], input_image.size(0))
            top5.update(acc5[0], input_image.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = (
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Prec@5 {top5.val:.3f} ({top5.avg:.3f})".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        top1=top1,
                        top5=top5,
                    )
                )
                print('\r'+output, end='')
        print()
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = "{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}".format(
            flag=flag, top1=top1, top5=top5, loss=losses
        )
        out_cls_acc = "%s Class Accuracy: %s" % (
            flag,
            (
                np.array2string(
                    cls_acc,
                    separator=",",
                    formatter={"float_kind": lambda x: "%.3f" % x},
                )
            ),
        )
        print(output)
        print(out_cls_acc)
        if log is not None:
            log.write(output + "\n")
            log.write(out_cls_acc + "\n")
            log.flush()

        tf_writer.add_scalar("loss/test_" + flag, losses.avg, epoch)
        tf_writer.add_scalar("acc/test_" + flag + "_top1", top1.avg, epoch)
        tf_writer.add_scalar("acc/test_" + flag + "_top5", top5.avg, epoch)
        tf_writer.add_scalars(
            "acc/test_" + flag + "_cls_acc",
            {str(i): x for i, x in enumerate(cls_acc)},
            epoch,
        )

    return top1.avg


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    epoch = epoch + 1
    if epoch <= 5:
        lr = args.lr * epoch / 5
    elif epoch > 180:
        lr = args.lr * 0.0001
    elif epoch > 160:
        lr = args.lr * 0.01
    else:
        lr = args.lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    main()
