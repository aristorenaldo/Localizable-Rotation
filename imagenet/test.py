import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from utils import AverageMeter, accuracy

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x

def train(model, train_loader,  optimizer, epoch, tf_writer):
    model.train()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    for b_i, (input, target) in enumerate(train_loader):
        input = input.cuda(0, non_blocking=True)
        target = target.cuda(0, non_blocking=True)

        output = model(input)
        loss = F.cross_entropy(output, target)
        acc1, = accuracy(output, target)
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b_i % 10 == 0 or b_i == (len(train_loader)-1):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, b_i * len(input), len(train_loader.dataset),
                    100. * b_i / len(train_loader), loss.item()))

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

def validate(model, val_loader, epoch, tf_writer):
    model.eval()
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    with torch.no_grad():
        for b_i, (input, target) in enumerate(val_loader):
            input = input.cuda(0, non_blocking=True)
            target = target.cuda(0, non_blocking=True)

            output = model(input)
            loss = F.cross_entropy(output, target)

            acc1, = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
        
        output = ('Val Results: Acc@1 {top1.avg:.3f} Loss {loss.avg:.5f}'
                .format( top1=top1, loss=losses))
        print(output)

        tf_writer.add_scalar('loss/test_val', losses.avg, epoch)
        tf_writer.add_scalar('acc/test_val_top1', top1.avg, epoch)
    
    return top1.avg

def main():
    epochs = 100
    lr = 0.1

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = torchvision.datasets.MNIST(root='data/', train=True, download=True, transform=transform)
    val_data = torchvision.datasets.MNIST(root='data/', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False)

    tf_writer = SummaryWriter(log_dir='testlog/')

    model = Net()
    torch.cuda.set_device(0)
    model = model.cuda(0)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)
    
    best_acc1 = 0

    for epoch in range(epochs):
        train(model, train_loader, optimizer, epoch, tf_writer)
        acc1 = validate(model, val_loader, epoch, tf_writer)

        scheduler.step()

        best_acc1 = max(acc1, best_acc1)
        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)


if __name__ == '__main__':
    main()