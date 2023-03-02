import itertools
import shutil
import torch
import os
import string
import random
from pathlib import Path

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

def getRandomString(length=10):
    return ''.join(random.choices(string.ascii_letters+string.digits, k=length))

