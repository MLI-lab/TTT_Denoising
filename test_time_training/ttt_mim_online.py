from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import argparse
import builtins
import os
import random
import time
import warnings
import numpy
import scipy.io as sio

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from dataset import SIDD, SimulatedNoise, FastMRI, MiniImageNetDenoisingwithMask, PolyU, DenoisingTestMixFolder, DND, CT
from loss import PSNR
from utils import MaskGenerator, Masker
from utils import pil_loader, denormalize
from unet.unet import JointUNetMIM
from copy import deepcopy

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='TTT-MIM online adaptation')
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
parser.add_argument('-a', '--arch', metavar='ARCH', default='unet',
                    choices=('unet'))
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--niters', default=8, type=int, help='maximum number of epoch for ttt')
parser.add_argument('--batch-size', default=1, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
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
parser.add_argument('--ImageNum', default=None, type=int,
                    help='Image selection from 1 to 3.')
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
parser.add_argument('--alpha', default=0.01, type=float, metavar='N')

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
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True if args.denoise_loss is None or args.denoise_loss == 'onlyn2s' else False)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True if args.denoise_loss is None or args.denoise_loss == 'onlyn2s' else False)
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

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = MiniImageNetDenoisingwithMask(
            args.root, mode='val', noise_mode=args.noise_mode,
            noise_var=args.noise_var, noise_amount=args.noise_amount,
            mask_generator=MaskGenerator(
                input_size=224,
                mask_patch_size=args.mask_patch_size,
                model_patch_size=args.model_patch_size,
                mask_ratio=args.mask_ratio
            ),
            transform=transform, residual_target=False)

    elif args.dataset == 'sidd':
        transform = transforms.Compose([
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = SIDD(args.root, mode='val', transform=transform, mask_generator=MaskGenerator(
            input_size=448,
            mask_patch_size=args.mask_patch_size,
            model_patch_size=args.model_patch_size,
            mask_ratio=args.mask_ratio
        ))

    elif args.dataset == 'polyu':
        transform = transforms.Compose([
            # transforms.CenterCrop(256),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = PolyU(args.root, mode='val', transform=transform, mask_generator=MaskGenerator(
            input_size=512,
            mask_patch_size=args.mask_patch_size,
            model_patch_size=args.model_patch_size,
            mask_ratio=args.mask_ratio
        ))

    elif args.dataset == 'urban' or args.dataset == 'cbsd':
        if args.noise_mode == 'sp':
            args.noise_mode = 's&p'
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = SimulatedNoise(args.root, mode='val', transform=val_transforms, mask_generator=MaskGenerator(
            input_size=224,
            mask_patch_size=args.mask_patch_size,
            model_patch_size=args.model_patch_size,
            mask_ratio=args.mask_ratio
        ), residual_target=False,
            noise_mode=args.noise_mode,
            noise_var=args.noise_var, noise_amount=args.noise_amount)

    elif args.dataset == 'fastmri':
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])
        dataset = SimulatedNoise(args.root, mode='val', transform=transform, mask_generator=MaskGenerator(
            input_size=224,
            mask_patch_size=args.mask_patch_size,
            model_patch_size=args.model_patch_size,
            mask_ratio=args.mask_ratio
        ), channel=1, residual_target=False)

    elif args.dataset == 'fmdd':
        transform = transforms.Compose([
            transforms.CenterCrop(256),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])
        target_transform = transform
        # can not find utils for fmdd?
        '''
        dataset = DenoisingTestMixFolder(args.root, utils.pil_loader, transform=transform,
                                         target_transform=target_transform, noise_levels=[1],
                                         mask_generator=MaskGenerator(
                                             input_size=256,
                                             mask_patch_size=args.mask_patch_size,
                                             model_patch_size=args.model_patch_size,
                                             mask_ratio=args.mask_ratio
                                         ))
                                         '''
        dataset = DenoisingTestMixFolder(args.root, pil_loader, transform=transform,
                                         target_transform=target_transform, noise_levels=[1],
                                         mask_generator=MaskGenerator(
                                             input_size=256,
                                             mask_patch_size=args.mask_patch_size,
                                             model_patch_size=args.model_patch_size,
                                             mask_ratio=args.mask_ratio
                                         ))                                 
    elif args.dataset == 'dnd':
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        dataset = DND(args.root, transform, mask_generator=MaskGenerator(
            input_size=512,
            mask_patch_size=args.mask_patch_size,
            model_patch_size=args.model_patch_size,
            mask_ratio=args.mask_ratio
        ))

    elif args.dataset == 'ct':
        transform = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])
        dataset = CT(args.root, mode='val', transform=transform, mask_generator=MaskGenerator(
            input_size=512,
            mask_patch_size=args.mask_patch_size,
            model_patch_size=args.model_patch_size,
            mask_ratio=args.mask_ratio
        ), residual_target=False)

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=sampler)

    # ----------- Test Time Adaption ------------
    original_state = deepcopy(model.state_dict())
    original_opt_state = deepcopy(optimizer.state_dict())
    metric = {'psnr': PSNR().cuda() if args.dataset != 'fastmri' else PSNR(denormalize=False).cuda()}
    masker = Masker(width=4, mode='interpolate', channel=3)

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    psnr = AverageMeter('PSNR', ':6.3f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, losses, psnr])

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
    psnr_ttt_mim_list = []
    psnr_noisy_input_list = []
    for i, (images, target, mask) in enumerate(loader):        
        im_interest = args.ImageNum - 1 # apply test time training to single image in the directory   
        
        output_denoised_im_list = [] # append denoised outputs at each step, also includes clean and noisy image as first two elements
        output_reconstructed_im_list = [] # append reconstructed outputs at each step, also includes masked image as first element
        psnr_den_list_len = args.niters + 2 # # of total images: outputs after each iteration + clean image + noisy image
        psnr_den_list = [0] * psnr_den_list_len # initialize list to hold psnr value of noisy and denoised outputs
        psnr_rec_list_len = args.niters + 2 # 
        psnr_rec_list = [0] * psnr_rec_list_len # initialize list to hold psnr value of masked and reconstructed outputs           
        output_denoised_im_list.append(target)   
        
        psnr_images = metric['psnr'](images, target)
        psnr_den_list[0] = psnr_images.cpu().detach().numpy().astype('float32') # psnr value of noisy image    
        psnr_noisy_input_list.append(psnr_den_list[0])        
        output_denoised_im_list.append(images)  
        
        # masked input before adaptation
        expanded_mask = mask.unsqueeze(1).expand(-1, 3, -1, -1)
        image_masked = images * expanded_mask
        psnr_masked = metric['psnr'](image_masked, target)
        psnr_rec_list[0] = psnr_masked.cpu().detach().numpy().astype('float32')            
        output_reconstructed_im_list.append(image_masked)            
        # reinitialize the model
        model.load_state_dict(original_state)
        optimizer.load_state_dict(original_opt_state)
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            mask = mask.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        loss_tmp = 0                
        for j in range(args.niters):
            optimizer.zero_grad()
            if args.denoise_loss == 'n2s' or args.denoise_loss == 'onlyn2s':
                interpolate_images, mask_n2s = masker.mask(images, i)  # n2s masking
                cat_images = torch.cat([images, interpolate_images], dim=0)
                output_denoise, output_recon = model(cat_images, mask, n2s=True)
            elif args.denoise_loss == 'pd':                                            
                output_denoise, output_recon = model(images, mask)
                # find output image and psnr 
                output_denoised = images - output_denoise
                output_denoised_im_list.append(output_denoised)
                psnr_output_denoised = metric['psnr'](output_denoised, target)
                psnr_den_list[j+1] = psnr_output_denoised.cpu().detach().numpy().astype('float32')

                # find reconstructed image and psnr
                expanded_mask = mask.unsqueeze(1).expand(-1, 3, -1, -1)
                output_reconstructed = torch.where(expanded_mask == 1, images, output_recon)
                output_reconstructed_im_list.append(output_reconstructed)
                psnr_output_reconstructed = metric['psnr'](output_reconstructed, images)
                psnr_rec_list[j+1] = psnr_output_reconstructed.cpu().detach().numpy().astype('float32')

            else:
                output_recon = model(images, mask, only_recon=True)

            if args.denoise_loss != 'onlyn2s':
                loss = criterion(output_recon, images)
                loss = args.alpha * (loss * mask.unsqueeze(1)).sum() / mask.sum() / 3
            else:
                loss = criterion(output_denoise, interpolate_images.detach() - images.detach())
                loss = (loss * mask_n2s).sum() / mask_n2s.sum() / 3

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
            loss_tmp += loss.item() / args.niters
        
        losses.update(loss_tmp, images.size(0))
        # evaluate the final image
        output_denoise, output_recon = model(images, mask)
        # eval
        model.eval()
        with torch.no_grad():
            output = model(images)
        if args.dataset == 'dnd':
            denoised = (images - output).cpu()
            denoised = utils.denormalize(denoised)
            denoised = torch.clamp(denoised, 0., 1.)
            denoised = denoised[0, ...].cpu().numpy()
            denoised = numpy.transpose(denoised, [1, 2, 0])
            save_file = os.path.join('./dnd_submit/dnd', '%04d_%02d.mat' % (i // 20 + 1, i % 20 + 1))
            sio.savemat(save_file, {'Idenoised_crop': denoised})
            print('[%d/%d] done' % (i + 1, 1000))
            continue

        psnr_tmp = metric['psnr'](images - output, target)
        psnr.update(psnr_tmp.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)
                    
        # save the final output at n_iters index
        # find output image and psnr 
        # psnr_den_list: 0 noisy, 1 before adapt, 2 output of first iteraiton, ...
        # psnr_rec_list: 0 masked, 1 before adapt, 2 output of first iteraiton, ...
        
        output_denoised = images - output_denoise            
        output_denoised_im_list.append(output_denoised)
        psnr_output_denoised = metric['psnr'](output_denoised, target)
        psnr_den_list[args.niters+1] = psnr_output_denoised.cpu().detach().numpy().astype('float32')
        
        # find reconstructed image and psnr
        expanded_mask = mask.unsqueeze(1).expand(-1, 3, -1, -1)
        output_reconstructed = torch.where(expanded_mask == 1, images, output_recon)
        output_reconstructed_im_list.append(output_reconstructed)
        psnr_output_reconstructed = metric['psnr'](output_reconstructed, images)
        psnr_rec_list[args.niters+1] = psnr_output_reconstructed.cpu().detach().numpy().astype('float32')

        # append the TTT_MIM output psnr of current image to TTT_MIM psnr list
        psnr_ttt_mim_list.append(psnr_den_list[args.niters + 1])
        if i == im_interest:    
            # plot images: clean, before adaptation, TTT_MIM, GT
            image_indices_plotted = [1, 2, args.niters + 2, 0]
            plt.figure(figsize=(20, 20))
            for cnt, k in enumerate(image_indices_plotted):
              output_cpu = tensor2np(output_denoised_im_list[k], args.dataset)
              plt.subplot(1, len(image_indices_plotted), cnt+1)
              if k == 1:
                  title = f"Noisy \n PSNR: {psnr_den_list[0]:.2f}"                                     
              elif k == 2:              
                  title = f"Train on P test on Q \n PSNR: {psnr_den_list[1]:.2f}"
              elif k == args.niters + 2:
                  title = f"TTT-MIM \n PSNR: {psnr_den_list[args.niters + 1]:.2f}"    
              elif k == 0:
                  title = "Ground Truth \n Clean"              
              plt.title(title, fontsize=12)
              plt.imshow(output_cpu)
              plt.axis('off')
            plt.show();   
    avg_psnr_TTT_MIM = sum(psnr_ttt_mim_list)/len(psnr_ttt_mim_list)
    avg_psnr_input_noisy = sum(psnr_noisy_input_list)/len(psnr_noisy_input_list)    
    print(f"Input Noisy PSNR for {args.dataset}: {avg_psnr_input_noisy:.2f}")
    print(f"TTT-MIM Output PSNR for {args.dataset}: {avg_psnr_TTT_MIM:.2f}")           
    progress.display(len(loader))    

def save_checkpoint(args, state, is_best, filename='checkpoint.pth.tar'):
    if not os.path.exists('./checkpoint/'):
        os.mkdir('./checkpoint/')
    postfix = '_{}_'.format(args.dataset)

    if is_best:
        if args.gn_train:
            torch.save(state, './checkpoint/' + 'gnopt_' + args.arch + postfix + 'model_best.pth.tar')
        else:
            if args.denoise_loss == 'n2s':
                torch.save(state, './checkpoint/' + 'ttt_n2s_' + args.arch + postfix + 'model_best.pth.tar')
            elif args.denoise_loss == 'pd':
                torch.save(state, './checkpoint/' + 'ttt_pd_' + args.arch + postfix + 'model_best.pth.tar')
            else:
                torch.save(state, './checkpoint/' + 'ttt_' + args.arch + postfix + 'model_best.pth.tar')

# convert tensor image to a format that is given to imshow
def tensor2np(mytensor, dataset):  
  if dataset == 'ct' or dataset == 'fastmri' or dataset == 'fmdd':
    output = mytensor
  else:
    output = denormalize(mytensor)
  output_cpu = (output.cpu().detach().numpy() * 255).astype('uint8')
  output_cpu = numpy.transpose(output_cpu[0], (1, 2, 0))
  output_cpu = numpy.clip(output_cpu, 0, 255)    
  return output_cpu

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
