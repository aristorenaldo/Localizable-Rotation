from torchvision import datasets

import itertools
import shutil
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from pathlib import Path
from typing import Union

def create_val_folder(data_set_path):
    """
    Used for Tiny-imagenet dataset
    Copied from https://github.com/soumendukrg/BME595_DeepLearning/blob/master/Homework-06/train.py
    This method is responsible for separating validation images into separate sub folders,
    so that test and val data can be read by the pytorch dataloaders
    """
    

    path = os.path.join(data_set_path, 'val/images')  # path where validation data is present now

    num_val_class = len([ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ])
    if num_val_class == 200:
        return

    filename = os.path.join(data_set_path,
                            'val/val_annotations.txt')  # file where image2class mapping is present
    fp = open(filename, "r")
    data = fp.readlines()

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        if os.path.exists(os.path.join(path, img)):
            os.rename(os.path.join(path, img), os.path.join(newpath, img))

def load_tinyimagenet_dataset(train_transform, val_transform, dataset_path='data/tiny-imagenet-200/'):
    create_val_folder(dataset_path)
    train_root = os.path.join(dataset_path,
                                'train')  # this is path to training images folder
    validation_root = os.path.join(dataset_path,
                                    'val/images')  # this is path to validation images folder
    train_dataset = datasets.ImageFolder(train_root, transform=train_transform)
    val_dataset = datasets.ImageFolder(validation_root, transform=val_transform)
    return train_dataset, val_dataset

def prepare_folders(args, use_argspars=True):
    folders_util = []
    if use_argspars:
        args.root_log = args.root_log
        folders_util = [
            args.root_log,
            args.root_model,
            os.path.join(args.root_log, args.store_name),
            os.path.join(args.root_model, args.store_name),
        ]
    else:
        folders_util = [
            args['root_log'],
            args['root_model'],
            os.path.join(args['root_log'], args['store_name']),
            os.path.join(args['root_model'], args['store_name']),
        ]

    for folder in folders_util:
        if not os.path.exists(folder):
            print("creating folder " + folder)
            mkdir_path = Path.cwd() / folder
            mkdir_path.mkdir(parents=True)

def save_checkpoint(args, state, is_best):

    filename = "%s/%s/ckpt.pth.tar" % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace("pth.tar", "best.pth.tar"))

class AverageMeter(object):
    def __init__(self, name, fmt=":f"):
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
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def shuffle_channel(img: torch.Tensor, index_shuffle: int) -> torch.Tensor:
    """Mengacak urutan dimensi RGB sebagai bentuk transformasi

    Parameters
    ----------
    img : torch.Tensor
        Pixel image RGB

    index_shuffle : int
        Index pengacakan berdasarkan kombinasi RGB
    Returns
    -------
    torch.Tensor
        Shuffled result image
    """
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img)

    list_to_permutations = list(itertools.permutations(range(3), 3))
    return img[list_to_permutations[index_shuffle], ...]

def flip_image(img: torch.Tensor, index_flip:int):
    if index_flip > 3 and index_flip < 0:
        raise ValueError('index_flip must be in range 0 to 3')

    if index_flip % 2 != 0: # check last digit in binary
        img = torch.fliplr(img)
    
    index_flip = index_flip >> 1 # shift to right one bit
    if index_flip % 2 != 0: # check last digit in binary
        img = torch.flipud(img)
    return img
