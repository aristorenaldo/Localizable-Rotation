from torchvision import datasets

import itertools
import shutil
import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

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

def abspath_dataset(path):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)),path)

def load_tinyimagenet_dataset(train_transform, val_transform, dataset_path=abspath_dataset('data/tiny-imagenet-200/')):
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

class SslTransform(object):
    '''
    Wrapper for SSL Transformation

    Parameter
    ---------
    arch : architecture of Gated-SSL to determine which transformations is used
        (Moe1, Lorot, Moe1Sc, Moe1Flip)
    
    Returns
    -------
    image : torch.Tensor
        Transformed image
    ssl_labels : int | tupple
        SSL label
    '''
    def __init__(self, arch) -> None:
        assert isinstance(arch,str)
        assert arch
        self.arch = arch

    
    def __transform_picker(self, image):
        idx = random.randint(0, 3) # select patch
        idx2 = random.randint(0, 3) # rotation
        r2 = image.size(1)
        r = r2 // 2
        
        if idx == 0:
            w1 = 0
            w2 = r
            h1 = 0
            h2 = r
        elif idx == 1:
            w1 = 0
            w2 = r
            h1 = r
            h2 = r2
        elif idx == 2:
            w1 = r
            w2 = r2
            h1 = 0
            h2 = r
        elif idx == 3:
            w1 = r
            w2 = r2
            h1 = r
            h2 = r2

        if self.arch == 'Moe1' or self.arch == 'Nomoe':
            flip_label = random.randint(0, 1)
            sc_label = random.randint(0, 5)
            if flip_label:
                image[:, w1:w2, h1:h2] = TF.hflip(image[:, w1:w2, h1:h2])
            # lorot
            image[:, w1:w2, h1:h2] = torch.rot90(image[:, w1:w2, h1:h2], idx2, [1,2])
            # shuffle channel
            image[:, w1:w2, h1:h2] = shuffle_channel(image[:, w1:w2, h1:h2], sc_label)

            rot_label = idx * 4 + idx2
            ssl_label = (rot_label, flip_label, sc_label)

            return image, ssl_label
        
        elif self.arch == 'Lorot':
            image[:, w1:w2, h1:h2] = torch.rot90(image[:, w1:w2, h1:h2], idx2, [1,2])
            rot_label = idx * 4 + idx2
            return image, rot_label
        
        elif self.arch == 'Moe1flip':
            flip_label = random.randint(0, 1)
            if flip_label:
                image[:, w1:w2, h1:h2] = TF.hflip(image[:, w1:w2, h1:h2])
            # lorot
            image[:, w1:w2, h1:h2] = torch.rot90(image[:, w1:w2, h1:h2], idx2, [1,2])
            rot_label = idx * 4 + idx2
            ssl_label = (rot_label, flip_label)
            return image, ssl_label
        
        elif self.arch == 'Moe1sc':
            sc_label = random.randint(0, 5)
             # lorot
            image[:, w1:w2, h1:h2] = torch.rot90(image[:, w1:w2, h1:h2], idx2, [1,2])
            # shuffle channel
            image[:, w1:w2, h1:h2] = shuffle_channel(image[:, w1:w2, h1:h2], sc_label)
            rot_label = idx * 4 + idx2
            ssl_label = (rot_label, sc_label)
            return image, ssl_label

        raise Exception('arch not implemented')
    
    def __call__(self, image: torch.Tensor):
        assert isinstance(image, torch.Tensor)
        return self.__transform_picker(image)

class TBLog:
    """
    Construc tensorboard writer (self.writer).
    The tensorboard is saved at os.path.join(tb_dir, file_name).
    """
    def __init__(self, tb_dir, file_name):
        self.tb_dir = tb_dir
        self.writer = SummaryWriter(os.path.join(self.tb_dir, file_name))
    
    def update(self, tb_dict, epoch, suffix=None):
        """
        Args
            tb_dict: contains scalar values for updating tensorboard
            epoch: contains information of epoch (int).
            suffix: If not None, the update key has the suffix.
        """
        if suffix is None:
            suffix = ''
        
        for key, value in tb_dict.items():
            if isinstance(value, dict):
                self.writer.add_scalars(suffix+key, value, epoch)
            else: 
                self.writer.add_scalar(suffix+key, value, epoch) 