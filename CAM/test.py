import torch
from utils import shuffle_channel 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image
import random

def main():
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]), download=True),
        batch_size=128, shuffle=True,
        num_workers=4, pin_memory=True)
    
class MyTransform(object):
    def __init__(self, transform_all:bool=True) -> None:
        assert isinstance(transform_all,bool)
        self.transform_all = transform_all

    def __transform_all(image):
        flip_label = random.randint(0, 1)
        sc_label = random.randint(0, 5)

        idx = random.randint(0, 3)
        idx2 = random.randint(0, 3)
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
        if flip_label:
            image[:, w1:w2, h1:h2] = torch.flip(image[:, w1:w2, h1:h2], [2])
        # lorot
        image[:, w1:w2, h1:h2] = torch.rot90(image[:, w1:w2, h1:h2], idx2, [1,2])
        # shuffle channel
        image[:, w1:w2, h1:h2] = shuffle_channel(image[:, w1:w2, h1:h2], sc_label)

        rot_label = idx * 4 + idx2
        ssl_labels = (rot_label, flip_label, sc_label)
        
        return image, ssl_labels
    
    def __transform_lorot(image):
        idx = random.randint(0, 3)
        idx2 = random.randint(0, 3)
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
        image[:, w1:w2, h1:h2] = torch.rot90(image[:, w1:w2, h1:h2], idx2, [1,2])
        rot_label = idx * 4 + idx2
        return image, rot_label
    
    def __call__(self, image: torch.Tensor):
        assert isinstance(image, torch.Tensor)
        if self.transform_all:
            return self.__transform_all(image)
        return self.__transform_lorot(image)
