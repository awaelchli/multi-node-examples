"""
Adapted from PyTorch official examples at
https://github.com/pytorch/examples/blob/master/imagenet/main.py

Example:
Launching processes by hand on two nodes, two processes each:

# first process, first node
WORLD_SIZE=4 RANK=0 NODE_RANK=0 LOCAL_RANK=0 MASTER_ADDR=node01.cluster MASTER_PORT=1234 python train.py --num-gpus 2 --fake-data

# second process, first node
WORLD_SIZE=4 RANK=1 NODE_RANK=0 LOCAL_RANK=1 MASTER_ADDR=node01.cluster MASTER_PORT=1238 python train.py --num-gpus 2 --fake-data

# third process, second node
WORLD_SIZE=4 RANK=2 NODE_RANK=1 LOCAL_RANK=0 MASTER_ADDR=node01.cluster MASTER_PORT=1238 python train.py --num-gpus 2 --fake-data

# fourth process, second node
WORLD_SIZE=4 RANK=3 NODE_RANK=1 LOCAL_RANK=1 MASTER_ADDR=node01.cluster MASTER_PORT=1238 python train.py --num-gpus 2 --fake-data


Example:
Single node, two GPUs, launch with torch.distributed.launch:

python -m torch.distributed.launch --nnodes 1  --nproc_per_node 2 --master_addr 127.0.0.1 --master_port 1234  --use_env train.py --num-gpus 2 --fake-data

"""

import argparse
import os
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from imagenet.pytorch.utils import (
    AverageMeter,
    FakeImageNetDataset,
    ProgressMeter,
    accuracy,
    adjust_learning_rate,
    save_checkpoint,
)

parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
parser.add_argument("--data", default=".", metavar="DIR", help="path to dataset")
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
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
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
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
    "--num-gpus", required=True, type=int, help="number of gpus per node"
)
parser.add_argument(
    "--fake-data",
    default=False,
    action="store_true",
    help="simulate fake data instead of using ImageNet",
)


def main():
    args = parser.parse_args()

    random.seed(123)
    torch.manual_seed(123)
    cudnn.benchmark = True

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "23456")
    os.environ.setdefault("WORLD_SIZE", str(args.num_gpus))
    os.environ.setdefault("NODE_RANK", "0")

    args.world_size = int(os.environ["WORLD_SIZE"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.node_rank = int(os.environ["NODE_RANK"])
    args.rank = int(
        os.environ.get("RANK", args.node_rank * args.num_gpus + args.local_rank)
    )

    print("Initializing process group. Waiting for all processes to join ...")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
    )

    print(
        f"Using GPU {args.local_rank}, "
        f"GLOBAL RANK {args.rank}/{args.world_size}, "
        f"LOCAL RANK {args.local_rank}/{args.num_gpus}"
    )

    # create model
    model = models.resnet18(pretrained=args.pretrained)

    # Set the current device. Memory will only be allocated on the selected device.
    # DistributedDataParallel will use only this device.
    device = torch.device("cuda", args.local_rank)
    torch.cuda.set_device(device)
    model.to(device)

    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / args.num_gpus)
    args.workers = int((args.workers + args.num_gpus - 1) / args.num_gpus)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank]
    )

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Data loading code
    if args.fake_data:
        train_dataset = FakeImageNetDataset()
        val_dataset = FakeImageNetDataset()
    else:
        traindir = os.path.join(args.data, "train")
        valdir = os.path.join(args.data, "val")
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset = (
            datasets.ImageFolder(
                valdir,
                transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            ),
        )

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    if args.evaluate:
        validate(val_loader, model, criterion, device, args)
        return

    best_acc1 = 0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, device, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if args.rank == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
            )


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)


def validate(val_loader, model, criterion, device, args):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        print(f" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}")

    return top1.avg


if __name__ == "__main__":
    main()
