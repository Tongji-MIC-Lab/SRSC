"""Serial restoration of shuffled clips."""
import os
import math
import argparse
import time
import random
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, distributed
from torchvision import transforms
import torch.optim as optim
import torch.distributed as dist
from tensorboardX import SummaryWriter

from datasets.ucf101 import UCF101SRSCDataset
from models.c3d import C3D
from models.r3d import R3DNet
from models.r21d import R2Plus1DNet
from models.srsc_ablation import SRSCAblation
import torchvision.transforms.functional as tF

class MyRandomCrop:
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        w, h = tF._get_image_size(sample)
        th, tw = self.output_size

        if h+1 < th or w+1 <tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
                )

        i = random.randint(0, h-th)
        j = random.randint(0, w-tw)

        return tF.crop(sample, i, j, th, tw)

def train(args, model, criterion, optimizer, device, train_dataloader, writer, epoch):
    torch.set_grad_enabled(True)
    model.train()

    running_loss = 0.0
    correct = 0
    for i, data in enumerate(train_dataloader, 1):
        # get inputs
        # tuple_clips: bs * seq_len * C * T * H * W
        # tuple_orders: bs * seq_len
        tuple_clips, tuple_orders = data
        inputs = tuple_clips.to(device)
        _, orders = torch.sort(tuple_orders) 
        targets = orders.to(device)
        # targets = tuple_orders.to(device)
        # forward
        optimizer.zero_grad()
        # forward and backward
        outputs, pts = model(inputs) # return logits here
        loss = criterion(outputs.contiguous().view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        # compute loss and acc
        running_loss += loss.item()
        # pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print statistics and write summary every N batch
        if i % args.pf == 0:
            avg_loss = running_loss / args.pf
            avg_acc = correct / (args.pf * args.bs * args.tl)
            print('[TRAIN] epoch-{}, batch-{}, loss: {:.4f}, acc: {:.4f}'.format(epoch, i, avg_loss, avg_acc))
            step = (epoch-1)*len(train_dataloader) + i
            writer.add_scalar('train/CrossEntropyLoss', avg_loss, step)
            writer.add_scalar('train/Accuracy', avg_acc, step)
            running_loss = 0.0
            correct = 0
    # summary params and grads per eopch
    for name, param in model.named_parameters():
        writer.add_histogram('params/{}'.format(name), param, epoch)
        # writer.add_histogram('grads/{}'.format(name), param.grad, epoch)


def validate(args, model, criterion, device, val_dataloader, writer, epoch):
    torch.set_grad_enabled(False)
    model.eval()
    total_loss = 0.0
    correct = 0
    for i, data in enumerate(val_dataloader):
        # get inputs
        tuple_clips, tuple_orders = data
        inputs = tuple_clips.to(device)
        _, orders = torch.sort(tuple_orders)
        targets = orders.to(device)
        # targets = tuple_orders.to(device)
        # forward
        outputs, pts = model(inputs) # return logits here
        loss = criterion(outputs.contiguous().view(-1, outputs.size(-1)), targets.view(-1))
        # compute loss and acc
        total_loss += loss.item()
        # pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    avg_loss = total_loss / len(val_dataloader)
    avg_acc = correct / (len(val_dataloader.dataset) * args.tl / args.ng) 
    writer.add_scalar('val/CrossEntropyLoss', avg_loss, epoch)
    writer.add_scalar('val/Accuracy', avg_acc, epoch)
    print('[VAL] loss: {:.4f}, acc: {:.4f}'.format(avg_loss, avg_acc))
    return avg_loss


def test(args, model, criterion, device, test_dataloader):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    for i, data in enumerate(test_dataloader, 1):
        # get inputs
        tuple_clips, tuple_orders = data
        inputs = tuple_clips.to(device)
        _, orders = torch.sort(tuple_orders)
        targets= orders.to(device)
        # forward
        outputs, pts = model(inputs)
        for i,j in zip(pts, targets):
            print(i, '   ', j)
        loss = criterion(outputs.contiguous().view(-1, outputs.size(-1)), targets.view(-1))
        # compute loss and acc
        total_loss += loss.item()
        # pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(targets == pts).item()
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
    avg_loss = total_loss / len(test_dataloader)
    avg_acc = correct / (len(test_dataloader.dataset) * args.tl / args.ng) 
    print('[TEST] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss


def inplace_convert(model):
    for m in model.modules():
        if hasattr(m, 'inplace'):
            m.inplace = True


def parse_args():
    parser = argparse.ArgumentParser(description='Serial Restoration of Shuffled Clips')
    parser.add_argument('--mode', type=str, default='train', help='train/test')
    parser.add_argument('--model', type=str, default='r21d', help='c3d/r3d/r21d/s3dg')
    parser.add_argument('--cl', type=int, default=16, help='clip length')
    parser.add_argument('--it', type=int, default=8, help='interval')
    parser.add_argument('--tl', type=int, default=3, help='tuple length')
    parser.add_argument('--gpu', type=int, default=1, help='GPU id')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', type=float, default=9e-1, help='momentum')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--log', type=str, help='log directory')
    parser.add_argument('--ckpt', type=str, help='checkpoint path')
    parser.add_argument('--desp', type=str, help='additional description')
    parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
    parser.add_argument('--start-epoch', type=int, default=1, help='manual epoch number (useful on restarts)')
    parser.add_argument('--bs', type=int, default=8, help='mini-batch size')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--pf', type=int, default=100, help='print frequency every batch')
    parser.add_argument('--seed', type=int, default=632, help='seed for initializing training.')
    parser.add_argument('--ablation', type=str, default=None, help='which part to ablate in the ablation study.')
    ## distributed training params
    parser.add_argument('--local_rank', type=int, default=-1, help='node rank in multi process training')
    parser.add_argument('--distributed-train', default=False, action='store_true', help='enable distributed training')
    parser.add_argument('--ng', type=int, default=1, help='number of gpus')
    parser.add_argument('--gpus', type=str, default=None, help='gpus')
    # parser.add_argument('--mask', default=False, action='store_true', help='whether to use mask in TaskNet')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    # Force the pytorch to create context on the specific device 
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    ########### model ##############
    if args.model == 'c3d':
        base = C3D(with_classifier=False)
    elif args.model == 'r3d':
        base = R3DNet(layer_sizes=(1,1,1,1), with_classifier=False)
    elif args.model == 'r21d':   
        base = R2Plus1DNet(layer_sizes=(1,1,1,1), with_classifier=False)
    srsc = SRSCAblation(base_network=base, feature_size=512, tuple_len=args.tl,
                hidden_dim=512, input_dim=512, p=0.5, ablation=args.ablation).to(device)

    inplace_convert(srsc)

    if args.mode == 'train':  ########### Train #############
        if args.ckpt:  # resume training
            srsc.load_state_dict({k:v for k, v in torch.load(args.ckpt).items() if 'conv1d' not in k}, strict=False)
            log_dir = os.path.dirname(args.ckpt)
        else:
            if args.desp:
                exp_name = '{}_cl{}_it{}_tl{}_{}_{}'.format(args.model, args.cl, args.it, args.tl, args.desp, time.strftime('%m%d%H%M'))
            else:
                exp_name = '{}_cl{}_it{}_tl{}_{}'.format(args.model, args.cl, args.it, args.tl, time.strftime('%m%d%H%M'))
            log_dir = os.path.join(args.log, exp_name)
            print(log_dir)
        writer = SummaryWriter(log_dir)
        inplace_convert(srsc)
        train_transforms = transforms.Compose([
            transforms.Resize((128, 171)),  # smaller edge to 128
            # transforms.RandomCrop(112),
            MyRandomCrop(112),
            transforms.ToTensor()
        ])
        train_dataset = UCF101SRSCDataset('data/ucf101', args.cl, args.it, args.tl, True, train_transforms)
        # split val for 800 videos
        train_dataset, val_dataset = random_split(train_dataset, (len(train_dataset)-800, 800))
        print('TRAIN video number: {}, VAL video number: {}.'.format(len(train_dataset), len(val_dataset)))
        train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True,
                                    num_workers=args.workers, pin_memory=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False,
                                    num_workers=args.workers, pin_memory=True)

        if args.ckpt:
            pass
        else:
            # save graph and clips_order samples
            for data in train_dataloader:
                tuple_clips, tuple_orders = data
                for i in range(args.tl):
                    writer.add_video('train/tuple_clips', tuple_clips[:, i, :, :, :, :].permute(0,2,1,3,4), i, fps=8)
                    writer.add_text('train/tuple_orders', str(tuple_orders[:, i].tolist()), i)
                tuple_clips = tuple_clips.to(device)
                break
            # save init params at step 0

        criterion = nn.CrossEntropyLoss()
        base_network_groups = {'params': [p for n,p in srsc.named_parameters() if 'base_network' in n], 'lr': 1e-3, 'weight_decay':5e-4} 
        rest_groups = {'params': [p for n,p in srsc.named_parameters() if 'base_network' not in n]}
        optimizer = optim.SGD([base_network_groups, rest_groups], 
            lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        # print(optimizer.state_dict())
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-5, patience=50, factor=0.1)
        prev_best_val_loss = float('inf')
        prev_best_model_path = None
        for epoch in range(args.start_epoch, args.start_epoch+args.epochs):
            time_start = time.time()
            train(args, srsc, criterion, optimizer, device, train_dataloader, writer, epoch)
            print('Epoch time: {:.2f} s.'.format(time.time() - time_start))
            val_loss = validate(args, srsc, criterion, device, val_dataloader, writer, epoch)
            # scheduler.step(val_loss)         
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
            # save model every 20 epoches
            if epoch % 20 == 0:
                torch.save(srsc.state_dict(), os.path.join(log_dir, 'model_{}.pt'.format(epoch)))
            # save model for the best val
            if val_loss < prev_best_val_loss:
                model_path = os.path.join(log_dir, 'best_model_{}.pt'.format(epoch))
                torch.save(srsc.state_dict(), model_path)
                prev_best_val_loss = val_loss
                if prev_best_model_path:
                    os.remove(prev_best_model_path)
                prev_best_model_path = model_path

    elif args.mode == 'test':  ########### Test #############
        srsc.load_state_dict(torch.load(args.ckpt))
        test_transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            # transforms.CenterCrop(112),
            MyRandomCrop(112),
            transforms.ToTensor()
        ])
        test_dataset = UCF101SRSCDataset('data/ucf101', args.cl, args.it, args.tl, False, test_transforms)
        test_dataloader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                num_workers=args.workers, pin_memory=True)
        print('TEST video number: {}.'.format(len(test_dataset)))
        criterion = nn.CrossEntropyLoss()
        test(args, srsc, criterion, device, test_dataloader)

