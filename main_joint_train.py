from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from loss import PSNR
from utils import MaskGenerator
from dataset import MiniImageNetDenoising, MiniImageNetDenoisingwithMask, FastMRI, SIDD, SimulatedNoise
from unet.unet import JointUNetMIM


parser = argparse.ArgumentParser(description='Joint Training')
parser.add_argument('root', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', default='imagenet', type=str,
                    choices=('imagenet', 'sidd', 'urban', 'cbsd', 'fastmri'))
parser.add_argument('--noise-mode', default='gaussian', type=str, metavar='NOISEMODE',
                    choices=('gaussian', 'salt', 'pepper', 'sp', 'poisson', 'gaussian+poisson'),
                    help='noise mode (default: \'gaussian\')')
parser.add_argument('--noise-var', default=0.005, type=float, metavar='NOISEVAR',
                    help='variance of Gaussian noise (default: 0.005)')
parser.add_argument('--noise-amount', default=0.05, type=float, metavar='NOISEAMOUNT',
                    help='amount of salt, pepper and s&p noise (default: 0.05)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='unet', choices=('unet'))
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=80, type=int,
                    metavar='N',
                    help='mini-batch size (default: 80), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--warmup-epochs', default=5, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--eta-min', default=1e-6, type=float, metavar='N',
                    help='eta min')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to pretrained checkpoint')
parser.add_argument('--mask-ratio', default=0.6, type=float)
parser.add_argument('--mask-patch-size', default=7, type=int)
parser.add_argument('--model-patch-size', default=1, type=int)
parser.add_argument('--balance', default=0.5, type=float, metavar='N')

best_psnr = 0


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "c" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_psnr
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'unet':
        model = JointUNetMIM(3, 3)
    else:
        raise Exception('no such model')

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith('module.'):
                    # remove prefix
                    state_dict[k[len("module."):]] = state_dict[k]

                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            for k in msg.missing_keys:
                print(k)

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    # infer learning rate before changing batch size
    # init_lr = args.lr * args.batch_size / 256
    init_lr = args.lr

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.L1Loss(reduction='none').cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.AdamW(parameters, init_lr,
                                  weight_decay=args.weight_decay)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=40)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    if args.noise_mode == 'sp':
        args.noise_mode = 's&p'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.dataset == 'imagenet':
        train_dataset = MiniImageNetDenoisingwithMask(
            args.root, mode='train', noise_mode=args.noise_mode, noise_var=args.noise_var, noise_amount=args.noise_amount,
            mask_generator=MaskGenerator(
                input_size=224,
                mask_patch_size=args.mask_patch_size,
                model_patch_size=args.model_patch_size,
                mask_ratio=args.mask_ratio
            ),
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]), residual_target=False)

        val_dataset = MiniImageNetDenoising(args.root, mode='val', noise_mode=args.noise_mode,
                                            noise_var=args.noise_var, noise_amount=args.noise_amount,
                                            transform=transforms.Compose([
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                normalize,
                                            ]), residual_target=False)
    elif args.dataset == 'sidd':
        train_transforms = transforms.Compose([
            transforms.RandomCrop(448),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        val_transforms = transforms.Compose([
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = SIDD(args.root, mode='train', transform=train_transforms, mask_generator=MaskGenerator(
            input_size=448,
            mask_patch_size=args.mask_patch_size,
            model_patch_size=args.model_patch_size,
            mask_ratio=args.mask_ratio
        ), length=6000)
        val_dataset = SIDD(args.root, mode='val', transform=val_transforms, length=6000)
    elif args.dataset == 'fastmri':
        train_transforms = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(3),
            transforms.ToTensor(),
        ])

        val_transforms = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Grayscale(3),
            transforms.ToTensor(),
        ])
        train_dataset = SimulatedNoise(os.path.join(args.root, 'singlecoil_train/gt'), mode='train', transform=train_transforms,
                                       mask_generator=MaskGenerator(
                                           input_size=224,
                                           mask_patch_size=args.mask_patch_size,
                                           model_patch_size=args.model_patch_size,
                                           mask_ratio=args.mask_ratio
                                       ), residual_target=False, channel=1, noise_var=args.noise_var, train_test=0.7)
        val_dataset = SimulatedNoise(os.path.join(args.root, 'singlecoil_val/gt'), mode='val', transform=val_transforms,
                                     residual_target=False, channel=1, noise_var=args.noise_var, train_test=0.7)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    metric = {'psnr': PSNR(denormalize=True).cuda() if args.dataset != 'fastmri' else PSNR(denormalize=False).cuda()}
    if args.evaluate:
        if args.distributed:
            val_sampler.set_epoch(0)
        validate(val_loader, model, criterion, metric, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        # instead of using PyTorch's lr scheduler, use a custom function to adjust the lr
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, optimizer, criterion, metric, epoch, args)

        # scheduler works here
        # scheduler.step()

        # evaluate on validation set
        psnr = validate(val_loader, model, criterion, metric, args)

        # remember best acc@1 and save checkpoint
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_psnr': best_psnr,
                'optimizer': optimizer.state_dict()
            }, is_best)


def train(train_loader, model, optimizer, criterion, metric, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_denoise = AverageMeter('Loss denoise', ':.4e')
    losses_recon = AverageMeter('Loss recon', ':.4e')
    psnr = AverageMeter('PSNR', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses_denoise, losses_recon, psnr],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (images, target, mask) in enumerate(train_loader):
        # measure data loading time
        optimizer.zero_grad()
        data_time.update(time.time() - end)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            mask = mask.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        output_main, output_recon = model(images, mask)

        loss_recon = criterion(output_recon, images)
        loss_recon = (loss_recon * mask.unsqueeze(1)).sum() / mask.sum() / 3
        loss_recon_stat = loss_recon.item()

        loss_denoise = criterion(output_main, images - target).mean()
        loss_denoise_stat = loss_denoise.item()

        loss = args.balance * loss_recon + (1-args.balance)*loss_denoise
        loss.backward()

        with torch.no_grad():
            psnr_tmp = metric['psnr'](images - output_main, target)
            psnr_tmp = psnr_tmp.detach().cpu().numpy()

        losses_recon.update(loss_recon_stat, images.size(0))
        losses_denoise.update(loss_denoise_stat, images.size(0))
        psnr.update(psnr_tmp, images.size(0))

        # compute gradient and do SGD step
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, metric, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    psnr = AverageMeter('PSNR', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, psnr],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target).mean()

            # measure accuracy and record loss

            psnr_tmp = metric['psnr'](images-output,  target)
            psnr_tmp = psnr_tmp.detach().cpu().numpy()

            losses.update(loss.item(), images.size(0))
            psnr.update(psnr_tmp, images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * PSNR {psnr.avg:.3f}'
              .format(psnr=psnr))

    return psnr.avg


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    if not os.path.exists('./mask_checkpoint/'):
        os.mkdir('./mask_checkpoint/')
    torch.save(state, './mask_checkpoint/' + 'simmim_' + args.arch + '_' + filename)
    if is_best:
        shutil.copyfile('./mask_checkpoint/' + 'simmim_' + args.arch + '_' + filename,
                        './mask_checkpoint/' + 'simmim_' + args.arch + '_model_best.pth.tar')


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    if epoch < args.warmup_epochs:
        cur_lr = init_lr * (epoch + 1) / args.warmup_epochs
    else:
        cur_lr = args.eta_min + (init_lr - args.eta_min) * 0.5 * (
                1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':
    main()
