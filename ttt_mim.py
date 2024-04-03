from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

import argparse
import builtins
import os
import random
import time
import warnings

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from dataset import SIDD, SimulatedNoise, FastMRI, MiniImageNetDenoisingwithMask, MiniImageNetDenoising, PolyU, \
    DenoisingTestMixFolder, DenoisingFolder, DND, CT
from loss import PSNR
from utils import MaskGenerator, Masker
from unet.unet import JointUNetMIM
import utils



parser = argparse.ArgumentParser(description='TTT-MIM')
parser.add_argument('root', metavar='DIR', help='path to dataset')
parser.add_argument('--dataset', default='imagenet', type=str,
                    choices=('imagenet', 'sidd', 'urban', 'cbsd', 'fastmri', 'polyu', 'fmdd', 'dnd', 'ct'))
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

parser.add_argument('--nepochs', default=20, type=int, help='maximum number of epoch for ttt')
parser.add_argument('--stopepoch', default=100, type=int)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size-val', default=64, type=int, metavar='N')
parser.add_argument('--batch-size-adapt', default=40, type=int)
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', default='', type=str,
                    help='path to joint pretrained checkpoint')
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
parser.add_argument('--mask-ratio', default=0.3, type=float)
parser.add_argument('--mask-patch-size', default=7, type=int)
parser.add_argument('--model-patch-size', default=1, type=int)
parser.add_argument('--fix-ssh', action='store_true')
parser.add_argument('--fix-proj', action='store_true')
parser.add_argument('--denoise-loss', default=None, choices=(None, 'pd', 'n2s', 'onlyn2s'))
parser.add_argument('--gn-train', action='store_true')
parser.add_argument('--supervise', action='store_true')
parser.add_argument('--alpha', default=0.01, type=float, metavar='N')

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

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                if k.startswith('module.'):
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
    # init_lr = args.lr * args.batch_size_adapt / 256
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
            args.batch_size_adapt = int(args.batch_size_adapt / ngpus_per_node)
            args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                              find_unused_parameters=True if args.denoise_loss is None or args.denoise_loss == 'onlyn2s' else False)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                              find_unused_parameters=True if args.denoise_loss is None or args.denoise_loss == 'onlyn2s' else False)
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
    parameters = []

    # GN
    if args.gn_train:
        if args.distributed:
            model.module.main_block.requires_grad_(False)
        else:
            model.main_block.requires_grad_(False)
        for name, m in model.named_modules():
            if 'main_block' not in name:
                if (args.fix_ssh and 'pred_block' in name) or (args.fix_proj and 'proj' in name):
                    continue
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias']:  # weight is scale, bias is shift
                            parameters.append(p)
            elif args.denoise_loss:
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.GroupNorm)):
                    for np, p in m.named_parameters():
                        if np in ['weight', 'bias']:  # weight is scale, bias is shift
                            parameters.append(p)

    # TTT
    else:
        if not args.denoise_loss:
            if args.distributed:
                model.module.main_block.requires_grad_(False)
            else:
                model.main_block.requires_grad_(False)
        if args.fix_proj:
            if args.distributed:
                model.module.proj.requires_grad_(False)
            else:
                model.proj.requires_grad_(False)

        for name, p in model.named_parameters():
            if 'main_block' not in name:
                if (args.fix_ssh and 'pred_block' in name) or (args.fix_proj and 'proj' in name):
                    continue
                parameters.append(p)
            elif args.denoise_loss:
                parameters.append(p)

    optimizer = torch.optim.AdamW(parameters, init_lr, weight_decay=args.weight_decay)

    # lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'max', factor=0.5, patience=3, cooldown=10,
                                                           threshold=1e-7, threshold_mode='rel', min_lr=1e-7,
                                                           verbose=True)

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
            best_psnr = checkpoint['best_psnr']
            if args.gpu is not None:
                best_psnr = best_psnr

            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    print("=> loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if args.dataset == 'imagenet':
        if args.noise_mode == 'sp':
            args.noise_mode = 's&p'
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = MiniImageNetDenoisingwithMask(
            args.root, mode='val' if not args.supervise else 'train', noise_mode=args.noise_mode,
            noise_var=args.noise_var, noise_amount=args.noise_amount,
            mask_generator=MaskGenerator(
                input_size=224,
                mask_patch_size=args.mask_patch_size,
                model_patch_size=args.model_patch_size,
                mask_ratio=args.mask_ratio
            ),
            transform=train_transforms, residual_target=False, length=200 if not args.supervise else None)

        val_dataset = MiniImageNetDenoising(args.root, mode='val', noise_mode=args.noise_mode,
                                            noise_var=args.noise_var, noise_amount=args.noise_amount,
                                            transform=val_transforms, residual_target=False)
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
        ))
        val_dataset = SIDD(args.root, mode='val', transform=val_transforms)

    elif args.dataset == 'polyu':
        t = transforms.RandomResizedCrop(512) if args.supervise else transforms.Compose([])
        train_transforms = transforms.Compose([
            t,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        val_transforms = transforms.Compose([
            # transforms.CenterCrop(256),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = PolyU(args.root, mode='train', transform=train_transforms, mask_generator=MaskGenerator(
            input_size=512,
            mask_patch_size=args.mask_patch_size,
            model_patch_size=args.model_patch_size,
            mask_ratio=args.mask_ratio
        ), length=70 if args.supervise else None)
        val_dataset = PolyU(args.root, mode='val', transform=val_transforms, length=70 if args.supervise else None)

    elif args.dataset == 'urban' or args.dataset == 'cbsd':
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = SimulatedNoise(args.root, mode='val' if not args.supervise else 'train', transform=train_transforms,
                                       mask_generator=MaskGenerator(
                                           input_size=224,
                                           mask_patch_size=args.mask_patch_size,
                                           model_patch_size=args.model_patch_size,
                                           mask_ratio=args.mask_ratio
                                       ), residual_target=False)
        val_dataset = SimulatedNoise(args.root, mode='val', transform=val_transforms, residual_target=False)

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
            transforms.ToTensor()
        ])
        train_dataset = SimulatedNoise(args.root, mode='val' if not args.supervise else 'train', transform=train_transforms, mask_generator=MaskGenerator(
            input_size=224,
            mask_patch_size=args.mask_patch_size,
            model_patch_size=args.model_patch_size,
            mask_ratio=args.mask_ratio
        ), channel=1, residual_target=False)

        val_dataset = SimulatedNoise(args.root, mode='val', transform=val_transforms, channel=1, residual_target=False)

    elif args.dataset == 'fmdd':
        train_transform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])
        val_transform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])

        target_train, target_val = train_transform, val_transform
        if not args.supervise:
            train_dataset = DenoisingTestMixFolder(args.root, utils.pil_loader, transform=train_transform,
                                                   target_transform=target_train, noise_levels=[1],
                                                   mask_generator=MaskGenerator(
                                                       input_size=256,
                                                       mask_patch_size=args.mask_patch_size,
                                                       model_patch_size=args.model_patch_size,
                                                       mask_ratio=args.mask_ratio
                                                   ))
        else:
            train_dataset = DenoisingFolder(args.root, transform=train_transform, loader=utils.pil_loader,
                                            target_transform=target_train, noise_levels=[1],
                                            mask_generator=MaskGenerator(
                                                input_size=256,
                                                mask_patch_size=args.mask_patch_size,
                                                model_patch_size=args.model_patch_size,
                                                mask_ratio=args.mask_ratio
                                            ), train=True)

        val_dataset = DenoisingTestMixFolder(args.root, utils.pil_loader, transform=val_transform,
                                             target_transform=target_val, noise_levels=[1])
    elif args.dataset == 'dnd':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train_dataset = DND(args.root, train_transform, mask_generator=MaskGenerator(
                                                input_size=512,
                                                mask_patch_size=args.mask_patch_size,
                                                model_patch_size=args.model_patch_size,
                                                mask_ratio=args.mask_ratio
                                            ))
        val_dataset = DND(args.root, val_transform, mask_generator=MaskGenerator(
            input_size=512,
            mask_patch_size=args.mask_patch_size,
            model_patch_size=args.model_patch_size,
            mask_ratio=args.mask_ratio
        ))

    elif args.dataset == 'ct':
        train_transforms = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(3),
            transforms.ToTensor(),
        ])

        val_transforms = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])
        train_dataset = CT(args.root, mode='val' if not args.supervise else 'train', transform=train_transforms, mask_generator=MaskGenerator(
            input_size=512,
            mask_patch_size=args.mask_patch_size,
            model_patch_size=args.model_patch_size,
            mask_ratio=args.mask_ratio
        ), residual_target=False)

        val_dataset = CT(args.root, mode='val', transform=val_transforms, residual_target=False)


    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size_adapt, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size_val, shuffle=(val_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    # ----------- Test Time Adaption ------------
    metric = {'psnr': PSNR().cuda() if args.dataset != 'fastmri' else PSNR(denormalize=False).cuda()}
    masker = Masker(width=4, mode='interpolate')
    if args.evaluate:
        if args.distributed:
            val_sampler.set_epoch(0)
        validate(val_loader, model, criterion, metric, args)
        return

    for epoch in range(args.start_epoch, args.nepochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
            val_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, masker, args)
        if args.dataset == 'dnd':
            if not os.path.exists('./checkpoint/dnd/'):
                os.mkdir('./checkpoint/dnd/')
            postfix = '_{}_epoch_{}'.format(args.dataset, epoch+1)
            state = {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_psnr': best_psnr,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            if args.gn_train:
                torch.save(state, './checkpoint/dnd/' + 'gnopt_' + args.arch + postfix + '.pth.tar')
            else:
                if args.denoise_loss == 'n2s':
                    torch.save(state, './checkpoint/dnd/' + 'ttt_n2s_' + args.arch + postfix + '.pth.tar')
                elif args.denoise_loss == 'pd':
                    torch.save(state, './checkpoint/dnd/' + 'ttt_pd_' + args.arch + postfix + '.pth.tar')
                else:
                    torch.save(state, './checkpoint/dnd/' + 'ttt_' + args.arch + postfix + '.pth.tar')
            continue

        # evaluate on validation set
        psnr = validate(val_loader, model, criterion, metric, args)

        # remember best psnr and save checkpoint
        is_best = psnr > best_psnr
        best_psnr = max(psnr, best_psnr)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint(args, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_psnr': best_psnr,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, is_best)

        # termination
        if epoch > args.stopepoch and is_best:
            print("Termination: {:.2f}".format(best_psnr))
            break

        # scheduler works here
        scheduler.step(psnr)


def train(train_loader, model, criterion, optimizer, epoch, masker, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    model.train()
    if not args.denoise_loss:
        if args.distributed:
            model.module.main_block.eval()
        else:
            model.main_block.eval()
    if args.fix_ssh:
        if args.distributed:
            model.module.pred_block.eval()
        else:
            model.pred_block.eval()

    end = time.time()
    for i, (images, target, mask) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        optimizer.zero_grad()

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            mask = mask.cuda(args.gpu, non_blocking=True)
            if args.supervise:
                target = target.cuda(args.gpu, non_blocking=True)

        if args.supervise:
            output = model(images)
            loss = criterion(output, images - target).mean()
            # output_main, output_recon = model(images, mask)
            # loss_recon = criterion(output_recon, images)
            # loss_recon = (loss_recon * mask.unsqueeze(1)).sum() / mask.sum() / 3
            # loss_denoise = criterion(output_main, target).mean()
            # loss = 0.5 * loss_recon + 0.5 * loss_denoise
        else:
            if args.denoise_loss == 'n2s' or args.denoise_loss == 'onlyn2s':
                interpolate_images, mask_n2s = masker.mask(images, i)  # n2s masking
                if args.denoise_loss == 'n2s':
                    cat_images = torch.cat([images, interpolate_images], dim=0)
                    output_denoise, output_recon = model(cat_images, mask, n2s=True)
                else:  # onlyn2s
                    output_denoise = model(images)
            elif args.denoise_loss == 'pd':
                output_denoise, output_recon = model(images, mask)
            else:
                output_recon = model(images, mask, only_recon=True)

            if args.denoise_loss != 'onlyn2s':
                loss = criterion(output_recon, images)
                loss = (loss * mask.unsqueeze(1)).sum() / mask.sum() / 3
            else:
                loss = criterion(output_denoise, interpolate_images.detach() - images.detach())
                loss = args.alpha * (loss * mask_n2s).sum() / mask_n2s.sum() / 3

            if args.denoise_loss == 'n2s':
                loss_denoise = criterion(output_denoise, interpolate_images.detach() - images.detach())
                loss_denoise = (loss_denoise * mask_n2s).sum() / mask_n2s.sum() / 3
                loss += args.alpha * loss_denoise
            elif args.denoise_loss == 'pd':
                loss_denoise = criterion(output_denoise, images.detach() - output_recon.detach())
                loss_denoise = (loss_denoise * mask.unsqueeze(1)).sum() / mask.sum() / 3
                loss += args.alpha * loss_denoise

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

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
            loss = criterion(images - output, target).mean()

            # measure accuracy and record loss
            psnr_tmp = metric['psnr'](images - output, target)

            losses.update(loss.item(), images.size(0))
            psnr.update(psnr_tmp.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

            # if i == 0:
            #     from utils import denormalize
            #     img = images.detach().cpu()
            #     target = target.detach().cpu()
            #     denoised_img = img - output.detach().cpu()
            #     denoised_img = denoised_img[0]
            #     t2pil = transforms.ToPILImage()
            #     denoised_img = torch.clamp(denoised_img, 0, 1)
            #     img = t2pil(img[0]).convert('RGB')
            #     denoised_img = t2pil(denoised_img).convert('RGB')
            #     target = t2pil(target[0]).convert('RGB')
            #     img.save('./img.png', 'png')
            #     denoised_img.save('./denoise.png', 'png')
            #     target.save('./gt.png', 'png')
            #     break
        # TODO: this should also be done with the ProgressMeter
        print(' * PSNR {psnr.avg:.3f}'
              .format(psnr=psnr))

    return psnr.avg


def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    if not os.path.exists('./checkpoint/'):
        os.mkdir('./checkpoint/')
    postfix = '_{}_'.format(args.dataset)

    if is_best:
        if args.supervise:
            torch.save(state, './checkpoint/' + 'finetune_' + args.arch + postfix + 'model_best.pth.tar')
        elif args.gn_train:
            torch.save(state, './checkpoint/' + 'gnopt_' + args.arch + postfix + 'model_best.pth.tar')
        else:
            if args.denoise_loss == 'n2s':
                torch.save(state, './checkpoint/' + 'ttt_n2s_' + args.arch + postfix + 'model_best.pth.tar')
            elif args.denoise_loss == 'pd':
                torch.save(state, './checkpoint/' + 'ttt_pd_' + args.arch + postfix + 'model_best.pth.tar')
            else:
                torch.save(state, './checkpoint/' + 'ttt_' + args.arch + postfix + 'model_best.pth.tar')


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
