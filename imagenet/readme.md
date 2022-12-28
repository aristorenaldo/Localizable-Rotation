### Training Tiny-Imagenet


train
```
python train_moe.py -a Moe1 -b resnet18 -j 8 --epochs 1 --gpu 0 -bs 256 --exp_str test
```

```
optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: Lorot | Moe1 | Moe1flip | Moe1sc |
                        vanilla (default: vanilla)
  -b Backbone, --backbone Backbone
                        model backbone: resnet18 | resnet34 | resnet50 |
                        resnet101 | resnet152 | resnext50_32x4d |
                        resnext101_32x8d | resnext101_64x4d | wide_resnet50_2
                        | wide_resnet101_2 (default: resnet18)
  --exp_str EXP_STR     number to indicate which experiment it is
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  --gpu GPU             GPU id to use.
  -bs N, --batch-size N
                        mini-batch size
  --lr LR, --learning-rate LR
...
  --root_log ROOT_LOG
  --root_model ROOT_MODEL
  --r_ratio R_RATIO     ratio
  --resume PATH         path to latest checkpoint (default: none)
```
download dataset tiny-imagenet [link](http://cs231n.stanford.edu/tiny-imagenet-200.zip)
```
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
```
### Acknowledgement

Implementations for Imbalanced Classification of LoRot is based on [LDAM-DRW](https://github.com/kaidic/LDAM-DRW)
