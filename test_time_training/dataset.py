import torch
from torch.utils.data import Dataset
from PIL import Image, ImageChops
import pandas as pd
import os
from skimage.util import random_noise
import numpy as np
import torchvision.transforms as transforms
import json
from utils import is_image_file, pil_loader
import scipy.io as sio

class MiniImageNetDenoising(Dataset):
    def __init__(self, root: str, mode: str = 'train', noise_mode='gaussian', noise_mean=0, noise_var=0.005,
                 noise_amount=0.05,
                 residual_target: bool = True, no_target: bool = False, transform=None, length=None):
        """
        root (str): root path of dataset
        mode (str): mode od dataset, 'train', 'val' or 'test' (default: 'train')
        noise_mean (int, float): mean of Gaussian noise (default: 0)
        noise_var (int, float): variance of Gaussian noise (default: 0.01)
        residual_target (bool): If True, noise as target, else, the original image (default: True)
        no_target (bool): If True, the output has no target, used for contrastive learning (default: False)
        transform: the transform function for image (default: None)
        """
        self.root = root
        self.mode = mode

        self.noise_mode = noise_mode
        assert noise_mode in ['gaussian', 'salt', 'pepper', 's&p', 'poisson', 'gaussian+poisson']
        self.mean = noise_mean
        self.var = noise_var
        self.amount = noise_amount

        if mode not in ['train', 'val', 'test']:
            raise ValueError('model can only be train, val or test')
        if not os.path.exists(os.path.join(root, mode + '.csv')):
            raise FileNotFoundError(os.path.join(root, mode + '.csv') + 'not found')
        df = pd.read_csv(os.path.join(root, mode + '.csv'))
        file_names = df['filename']
        self.file_paths = [os.path.join(root, name) for name in file_names]
        if length is not None:
            indices = np.random.permutation(len(self.file_paths))[:length]
            self.file_paths = np.array(self.file_paths)[indices]
        self.residual_target = residual_target
        self.transform = transform
        self.no_target = no_target
        self.pre_transform = None
        self.post_transform = None
        if transform is not None:
            if not self.no_target:
                self.pre_transform = []
                self.post_transform = []
                for t in transform.transforms:
                    if isinstance(t, (transforms.ToTensor, transforms.Normalize)):
                        self.post_transform.append(t)
                    else:
                        self.pre_transform.append(t)
                self.pre_transform = transforms.Compose(self.pre_transform)
                self.post_transform = transforms.Compose(self.post_transform)
            elif self.no_target and isinstance(transform, list):
                assert len(transform) == 2
                self.pre_transform = transform[0]
                self.post_transform = transform[1]

    def __getitem__(self, index):
        img = Image.open(self.file_paths[index])
        img.convert('RGB')
        if self.pre_transform is not None:
            img = self.pre_transform(img)

        if isinstance(self.var, (list, tuple)):
            var = np.random.rand(1) * (self.var[1] - self.var[0]) + self.var[0]
        else:
            var = self.var

        # add noise
        img_np = np.array(img)
        if self.mode == 'train':
            if self.noise_mode == 'gaussian':
                noisy_img = random_noise(img_np, mode='gaussian', mean=self.mean, var=var)
            elif self.noise_mode == 's&p' or self.noise_mode == 'salt' or self.noise_mode == 'pepper':
                noisy_img = random_noise(img_np, mode=self.noise_mode, amount=self.amount)
            elif self.noise_mode == 'poisson':
                noisy_img = random_noise(img_np, mode='poisson')
            elif self.noise_mode == 'gaussian+poisson':
                noisy_img = random_noise(img_np, mode='poisson')
                noisy_img = random_noise(noisy_img, mode='gaussian', mean=self.mean, var=var)

        else:
            # fixed noise for each image, convenient for validation
            if self.noise_mode == 'gaussian':
                noisy_img = random_noise(img_np, mode='gaussian', mean=self.mean, var=var, seed=1024 + index)
            elif self.noise_mode == 's&p' or self.noise_mode == 'salt' or self.noise_mode == 'pepper':
                noisy_img = random_noise(img_np, mode=self.noise_mode, amount=self.amount, seed=1024 + index)
            elif self.noise_mode == 'poisson':
                noisy_img = random_noise(img_np, mode='poisson', seed=1024 + index)
            elif self.noise_mode == 'gaussian+poisson':
                noisy_img = random_noise(img_np, mode='poisson', seed=1024 + index)
                noisy_img = random_noise(noisy_img, mode='gaussian', mean=self.mean, var=var, seed=1024 + index)
        noisy_img = Image.fromarray((255 * noisy_img).astype(np.uint8))

        if self.transform is not None:
            if self.no_target:
                noisy_img = self.transform(noisy_img) if self.post_transform is None else self.post_transform(noisy_img)
            else:
                noisy_img = self.post_transform(noisy_img)
                img = self.post_transform(img)

        if self.no_target:
            return noisy_img
        else:
            if self.residual_target:
                target = ImageChops.subtract(noisy_img, img) if isinstance(img, Image.Image) else noisy_img - img
            else:
                target = img
            return noisy_img, target

    def __len__(self):
        return len(self.file_paths)


class MiniImageNetDenoisingwithMask(MiniImageNetDenoising):
    def __init__(self, root: str, mode: str = 'train', noise_mode='gaussian', noise_mean=0, noise_var=0.005,
                 noise_amount=0.05, residual_target: bool = True, no_target: bool = False, transform=None,
                 mask_generator=None, length=None):

        super(MiniImageNetDenoisingwithMask, self).__init__(root, mode, noise_mode, noise_mean, noise_var,
                                                            noise_amount, residual_target, no_target, transform,
                                                            length=length)
        self.mask_generator = mask_generator

    def __getitem__(self, index):
        img = Image.open(self.file_paths[index])
        img.convert('RGB')
        if self.pre_transform is not None:
            img = self.pre_transform(img)

        # add noise
        if isinstance(self.var, (list, tuple)):
            var = np.random.rand(1) * (self.var[1] - self.var[0]) + self.var[0]
        else:
            var = self.var
        img_np = np.array(img)

        if self.mode == 'train':
            if self.noise_mode == 'gaussian':
                noisy_img = random_noise(img_np, mode='gaussian', mean=self.mean, var=var)
            elif self.noise_mode == 's&p' or self.noise_mode == 'salt' or self.noise_mode == 'pepper':
                noisy_img = random_noise(img_np, mode=self.noise_mode, amount=self.amount)
            elif self.noise_mode == 'poisson':
                noisy_img = random_noise(img_np, mode='poisson')
            elif self.noise_mode == 'gaussian+poisson':
                noisy_img = random_noise(img_np, mode='poisson')
                noisy_img = random_noise(noisy_img, mode='gaussian', mean=self.mean, var=var)

        else:
            # fixed noise for each image, convenient for validation
            if self.noise_mode == 'gaussian':
                noisy_img = random_noise(img_np, mode='gaussian', mean=self.mean, var=var, seed=1024 + index)
            elif self.noise_mode == 's&p' or self.noise_mode == 'salt' or self.noise_mode == 'pepper':
                noisy_img = random_noise(img_np, mode=self.noise_mode, amount=self.amount, seed=1024 + index)
            elif self.noise_mode == 'poisson':
                noisy_img = random_noise(img_np, mode='poisson', seed=1024 + index)
            elif self.noise_mode == 'gaussian+poisson':
                noisy_img = random_noise(img_np, mode='poisson', seed=1024 + index)
                noisy_img = random_noise(noisy_img, mode='gaussian', mean=self.mean, var=var, seed=1024 + index)

        noisy_img = Image.fromarray((255 * noisy_img).astype(np.uint8))

        if self.transform is not None:
            if self.no_target:
                noisy_img = self.transform(noisy_img) if self.post_transform is None else self.post_transform(noisy_img)
            else:
                noisy_img = self.post_transform(noisy_img)
                img = self.post_transform(img)

        if self.mask_generator is not None:
            mask = self.mask_generator()

        if self.no_target:
            if self.mask_generator is not None:
                return noisy_img, mask
            return noisy_img
        else:
            if self.residual_target:
                target = ImageChops.subtract(noisy_img, img) if isinstance(img, Image.Image) else noisy_img - img
            else:
                target = img
            if self.mask_generator is not None:
                return noisy_img, target, mask
            return noisy_img, target


class NaturalNoise(Dataset):
    def __init__(self, root, mode='train', transform=None, mask_generator=None):
        super(NaturalNoise, self).__init__()
        self.root = root
        self.transform = transform
        self.mask_generator = mask_generator
        self.mode = mode

        self.samples = []
        self.targets = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x = Image.open(self.samples[index])
        x.convert('RGB')
        y = Image.open(self.targets[index])
        y.convert('RGB')

        if self.mode == 'train':
            seed = np.random.randint(2147483647)
            torch.random.manual_seed(seed)

        if self.transform is not None:
            x = self.transform(x)
            if self.mode == 'train':
                torch.random.manual_seed(seed)
            y = self.transform(y)

        if self.mask_generator is not None:
            mask = self.mask_generator()
            return x, y, mask

        return x, y


class SIDD(NaturalNoise):
    def __init__(self, root, mode='train', transform=None, mask_generator=None, length=None):
        super(SIDD, self).__init__(root, mode, transform, mask_generator)
        file_list = os.listdir(self.root)

        for addr in file_list:
            if os.path.isdir(os.path.join(root, addr)):
                for file_name in os.listdir(os.path.join(root, addr)):
                    if os.path.isfile(os.path.join(root, addr, file_name)) and '.png' in file_name:
                        if 'NOISY' in file_name:
                            self.samples.append(os.path.join(root, addr, file_name))
                        elif 'GT' in file_name:
                            self.targets.append(os.path.join(root, addr, file_name))
        assert len(self.samples) == len(self.targets)
        if length is not None:
            if mode == 'train':
                self.samples = self.samples[:length]
                self.targets = self.targets[:length]
            else:
                self.samples = self.samples[length:]
                self.targets = self.targets[length:]


class FastMRI(NaturalNoise):
    def __init__(self, root, mode='train', transform=None, mask_generator=None):
        super(FastMRI, self).__init__(root, mode, transform, mask_generator)
        file_list = os.listdir(self.root)

        for addr in file_list:
            if os.path.isdir(os.path.join(root, addr)):
                for file_name in os.listdir(os.path.join(root, addr)):
                    if os.path.isfile(os.path.join(root, addr, file_name)) and '.png' in file_name:
                        if addr == 'noisy':
                            self.samples.append(os.path.join(root, addr, file_name))
                        elif addr == 'gt':
                            self.targets.append(os.path.join(root, addr, file_name))
        assert len(self.samples) == len(self.targets)


class PolyU(NaturalNoise):
    def __init__(self, root, mode='train', transform=None, mask_generator=None, length=None):
        super(PolyU, self).__init__(root, mode, transform, mask_generator)
        file_list = os.listdir(self.root)
        for addr in file_list:
            if addr.endswith('real.JPG') or addr.endswith('real.jpg'):
                self.samples.append(os.path.join(root, addr))
                addr = addr.replace('real', 'mean')
                if os.path.isfile(os.path.join(root, addr)):
                    self.targets.append(os.path.join(root, addr))
                else:
                    raise FileNotFoundError('The corresponding clean image of {} does\'t exist.'.format(addr))
        if length is not None:
            if mode == 'train':
                self.samples = self.samples[:30] + self.samples[-30:]
                self.targets = self.targets[:30] + self.targets[-30:]
            else:
                self.samples = self.samples[30:-30]
                self.targets = self.targets[30:-30]


class SimulatedNoise(Dataset):
    def __init__(self, root, mode='train', transform=None, mask_generator=None, residual_target=True,
                 noise_mode='gaussian', noise_mean=0, noise_var=0.005, noise_amount=0.05, channel=3, train_test=None):
        super(SimulatedNoise, self).__init__()
        self.root = root
        self.transform = transform
        self.mask_generator = mask_generator
        self.mode = mode
        self.residual_target = residual_target
        self.channel = channel
        self.pre_transform = None
        self.post_transform = None
        if transform is not None:
            self.pre_transform = []
            self.post_transform = []
            for t in transform.transforms:
                if isinstance(t, (transforms.Grayscale, transforms.ToTensor, transforms.Normalize)):
                    self.post_transform.append(t)
                else:
                    self.pre_transform.append(t)
            self.pre_transform = transforms.Compose(self.pre_transform)
            self.post_transform = transforms.Compose(self.post_transform)

        self.noise_mode = noise_mode
        assert noise_mode in ['gaussian', 'salt', 'pepper', 's&p', 'poisson', 'gaussian+poisson']
        self.mean = noise_mean
        self.var = noise_var
        self.amount = noise_amount

        self.file_path = []

        file_list = os.listdir(self.root)

        for file_name in file_list:
            if os.path.isfile(os.path.join(root, file_name)):
                self.file_path.append(os.path.join(root, file_name))
        if train_test is not None:
            length = int(train_test * (len(self.file_path)))
            self.file_path = self.file_path[:length] if mode == 'train' else self.file_path[length:]

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):
        img = Image.open(self.file_path[index])
        img.convert('RGB')
        if self.pre_transform is not None:
            img = self.pre_transform(img)

        img_np = np.array(img)

        if isinstance(self.var, (list, tuple)):
            var = np.random.rand(1) * (self.var[1] - self.var[0]) + self.var[0]
        else:
            var = self.var

        if self.mode == 'train':
            if self.noise_mode == 'gaussian':
                noisy_img = random_noise(img_np, mode='gaussian', mean=self.mean, var=var)
            elif self.noise_mode == 's&p' or self.noise_mode == 'salt' or self.noise_mode == 'pepper':
                noisy_img = random_noise(img_np, mode=self.noise_mode, amount=self.amount)
            elif self.noise_mode == 'poisson':
                noisy_img = random_noise(img_np, mode='poisson')
            elif self.noise_mode == 'gaussian+poisson':
                noisy_img = random_noise(img_np, mode='poisson')
                noisy_img = random_noise(noisy_img, mode='gaussian', mean=self.mean, var=var)

        else:
            # fixed noise for each image, convenient for validation
            if self.noise_mode == 'gaussian':
                noisy_img = random_noise(img_np, mode='gaussian', mean=self.mean, var=var, seed=1024 + index)
            elif self.noise_mode == 's&p' or self.noise_mode == 'salt' or self.noise_mode == 'pepper':
                noisy_img = random_noise(img_np, mode=self.noise_mode, amount=self.amount, seed=1024 + index)
            elif self.noise_mode == 'poisson':
                noisy_img = random_noise(img_np, mode='poisson', seed=1024 + index)
            elif self.noise_mode == 'gaussian+poisson':
                noisy_img = random_noise(img_np, mode='poisson', seed=1024 + index)
                noisy_img = random_noise(noisy_img, mode='gaussian', mean=self.mean, var=var, seed=1024 + index)

        if self.channel == 1:
            noise = (noisy_img - img_np)[:, :, 0]
            noise = noise[:, :, np.newaxis]
            noisy_img = img_np + noise

        noisy_img = Image.fromarray((255 * noisy_img).astype(np.uint8))

        if self.transform is not None:
            noisy_img = self.transform(noisy_img) if self.post_transform is None else self.post_transform(noisy_img)
            img = self.transform(img) if self.post_transform is None else self.post_transform(img)

        if self.residual_target:
            target = noisy_img - img
        else:
            target = img

        if self.mask_generator is not None:
            mask = self.mask_generator()
            return noisy_img, target, mask

        return noisy_img, target


class DenoisingTestMixFolder(Dataset):
    """Data loader for the denoising mixed test set.
        data_root/test_mix/noise_level/imgae.png
        type:           test_mix
        noise_level:    5 (+ 1: ground truth)
        captures.png:   48 images in each fov
    Args:
        noise_levels (seq): e.g. [1, 2, 4] select `raw`, `avg2`, `avg4` folders
    """

    def __init__(self, root, loader, noise_levels, transform, target_transform, mask_generator=None):
        super().__init__()
        all_noise_levels = [1, 2, 4, 8, 16]

        assert all([level in all_noise_levels for level in all_noise_levels])
        self.noise_levels = noise_levels

        self.root = root
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform
        self.samples = self._gather_files()
        self.mask_generator = mask_generator

        dataset_info = {'Dataset': 'test_mix',
                        'Noise levels': self.noise_levels,
                        '# samples': len(self.samples)
                        }
        print(json.dumps(dataset_info, indent=4))

    def _gather_files(self):
        samples = []
        root_dir = os.path.expanduser(self.root)
        test_mix_dir = os.path.join(root_dir, 'test_mix')
        gt_dir = os.path.join(test_mix_dir, 'gt')

        for noise_level in self.noise_levels:
            if noise_level == 1:
                noise_dir = os.path.join(test_mix_dir, 'raw')
            elif noise_level in [2, 4, 8, 16]:
                noise_dir = os.path.join(test_mix_dir, f'avg{noise_level}')

            for fname in sorted(os.listdir(noise_dir)):
                if is_image_file(fname):
                    noisy_file = os.path.join(noise_dir, fname)
                    clean_file = os.path.join(gt_dir, fname)
                    samples.append((noisy_file, clean_file))

        return samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (noisy, clean)
        """
        noisy_file, clean_file = self.samples[index]
        noisy, clean = self.loader(noisy_file), self.loader(clean_file)
        if self.transform is not None:
            noisy = self.transform(noisy)
        if self.target_transform is not None:
            clean = self.target_transform(clean)
        if self.mask_generator is not None:
            mask = self.mask_generator()
            return noisy, clean, mask
        return noisy, clean

    def __len__(self):
        return len(self.samples)


class DenoisingFolder(torch.utils.data.Dataset):
    """Class for the denoising dataset for both train and test, with
    file structure:
        data_root/type/noise_level/fov/capture.png
        type:           12
        noise_level:    5 (+ 1: ground truth)
        fov:          20 (the 19th fov is for testing)
        capture.png:    50 images in each fov --> use fewer samples

    Args:
        root (str): root directory to the dataset
        train (bool): Training set if True, else Test set
        noise_levels (seq): e.g. [1, 2, 4] select `raw`, `avg2`, `avg4` folders
        types (seq, optional): e.g. ['TwoPhoton_BPAE_B', 'Confocal_MICE`]
        test_fov (int, optional): default 19. 19th fov is test fov
        captures (int): select # images within one folder
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version. E.g, transforms.RandomCrop
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        loader (callable, optional): image loader
    """

    def __init__(self, root, train, noise_levels, types=None, test_fov=19,
                 captures=2, transform=None, target_transform=None, loader=pil_loader, mask_generator=None):
        super().__init__()
        all_noise_levels = [1, 2, 4, 8, 16]
        all_types = ['TwoPhoton_BPAE_R', 'TwoPhoton_BPAE_G', 'TwoPhoton_BPAE_B',
                     'TwoPhoton_MICE', 'Confocal_MICE', 'Confocal_BPAE_R',
                     'Confocal_BPAE_G', 'Confocal_BPAE_B', 'Confocal_FISH',
                     'WideField_BPAE_R', 'WideField_BPAE_G', 'WideField_BPAE_B']
        assert all([level in all_noise_levels for level in all_noise_levels])
        self.noise_levels = noise_levels
        if types is None:
            self.types = all_types
        else:
            assert all([img_type in all_types for img_type in types])
            self.types = types
        self.root = root
        if train:
            fovs = list(range(1, 20 + 1))
            fovs.remove(test_fov)
            self.fovs = fovs
        else:
            self.fovs = [test_fov]
        self.captures = captures
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.samples = self._gather_files()
        self.mask_generator = mask_generator

        dataset_info = {'Dataset': 'train' if train else 'test',
                        'Noise levels': self.noise_levels,
                        f'{len(self.types)} Types': self.types,
                        'Fovs': self.fovs,
                        '# samples': len(self.samples)
                        }
        print(json.dumps(dataset_info, indent=4))

    def _gather_files(self):
        samples = []
        root_dir = os.path.expanduser(self.root)
        # types: microscopy_cell
        subdirs = [os.path.join(root_dir, name) for name in os.listdir(root_dir)
                   if (os.path.isdir(os.path.join(root_dir, name)) and name in self.types)]

        for subdir in subdirs:
            gt_dir = os.path.join(subdir, 'gt')
            for noise_level in self.noise_levels:
                if noise_level == 1:
                    noise_dir = os.path.join(subdir, 'raw')
                elif noise_level in [2, 4, 8, 16]:
                    noise_dir = os.path.join(subdir, f'avg{noise_level}')
                for i_fov in self.fovs:
                    noisy_fov_dir = os.path.join(noise_dir, f'{i_fov}')
                    clean_file = os.path.join(gt_dir, f'{i_fov}', 'avg50.png')
                    for fname in sorted(os.listdir(noisy_fov_dir))[:self.captures]:
                        if is_image_file(fname):
                            noisy_file = os.path.join(noisy_fov_dir, fname)
                            samples.append((noisy_file, clean_file))
        return samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (noisy, clean)
        """
        noisy_file, clean_file = self.samples[index]
        noisy, clean = self.loader(noisy_file), self.loader(clean_file)
        if self.transform is not None:
            noisy = self.transform(noisy)
        if self.target_transform is not None:
            clean = self.target_transform(clean)
        if self.mask_generator is not None:
            mask = self.mask_generator()
            return noisy, clean, mask

        return noisy, clean

    def __len__(self):
        return len(self.samples)


class DND(NaturalNoise):
    def __init__(self, root, transform=None, mask_generator=None):
        super(DND, self).__init__(root, transform=transform, mask_generator=mask_generator)
        file_list = os.listdir(self.root)
        for name in sorted(file_list):
            self.samples.append(os.path.join(root, name))

    def __getitem__(self, idx):
        img = sio.loadmat(self.samples[idx])
        x = np.uint8(255 * np.array(img['Inoisy_crop']))
        x = Image.fromarray(x, mode='RGB')
        if self.transform is not None:
            x = self.transform(x)
        y = x

        if self.mask_generator is not None:
            mask = self.mask_generator()
            return x, y, mask

        return x, y


class CT(Dataset):
    def __init__(self, root, mode='train', transform=None, mask_generator=None, residual_target=True,
                 noise_mode='gaussian', noise_mean=0, noise_var=0.005, noise_amount=0.05):
        super().__init__()
        self.root = root
        self.transform = transform
        self.mask_generator = mask_generator
        self.mode = mode
        self.residual_target = residual_target
        self.pre_transform = None
        self.post_transform = None
        if transform is not None:
            self.pre_transform = []
            self.post_transform = []
            for t in transform.transforms:
                if isinstance(t, (transforms.Grayscale, transforms.ToTensor, transforms.Normalize)):
                    self.post_transform.append(t)
                else:
                    self.pre_transform.append(t)
            self.pre_transform = transforms.Compose(self.pre_transform)
            self.post_transform = transforms.Compose(self.post_transform)

        self.noise_mode = noise_mode
        assert noise_mode in ['gaussian', 'salt', 'pepper', 's&p', 'poisson', 'gaussian+poisson']
        self.mean = noise_mean
        self.var = noise_var
        self.amount = noise_amount

        self.file = np.load(self.root)

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        img = (self.file[index] - np.min(self.file[index])) / np.max(self.file[index])
        img = Image.fromarray((255 * img).astype(np.uint8))
        img.convert('RGB')
        if self.pre_transform is not None:
            img = self.pre_transform(img)

        img_np = np.array(img)

        if isinstance(self.var, (list, tuple)):
            var = np.random.rand(1) * (self.var[1] - self.var[0]) + self.var[0]
        else:
            var = self.var

        if self.mode == 'train':
            if self.noise_mode == 'gaussian':
                noisy_img = random_noise(img_np, mode='gaussian', mean=self.mean, var=var)
            elif self.noise_mode == 's&p' or self.noise_mode == 'salt' or self.noise_mode == 'pepper':
                noisy_img = random_noise(img_np, mode=self.noise_mode, amount=self.amount)
            elif self.noise_mode == 'poisson':
                noisy_img = random_noise(img_np, mode='poisson')
            elif self.noise_mode == 'gaussian+poisson':
                noisy_img = random_noise(img_np, mode='poisson')
                noisy_img = random_noise(noisy_img, mode='gaussian', mean=self.mean, var=var)

        else:
            # fixed noise for each image, convenient for validation
            if self.noise_mode == 'gaussian':
                noisy_img = random_noise(img_np, mode='gaussian', mean=self.mean, var=var, seed=1024 + index)
            elif self.noise_mode == 's&p' or self.noise_mode == 'salt' or self.noise_mode == 'pepper':
                noisy_img = random_noise(img_np, mode=self.noise_mode, amount=self.amount, seed=1024 + index)
            elif self.noise_mode == 'poisson':
                noisy_img = random_noise(img_np, mode='poisson', seed=1024 + index)
            elif self.noise_mode == 'gaussian+poisson':
                noisy_img = random_noise(img_np, mode='poisson', seed=1024 + index)
                noisy_img = random_noise(noisy_img, mode='gaussian', mean=self.mean, var=var, seed=1024 + index)


        noisy_img = Image.fromarray((255 * noisy_img).astype(np.uint8))

        if self.transform is not None:
            noisy_img = self.transform(noisy_img) if self.post_transform is None else self.post_transform(noisy_img)
            img = self.transform(img) if self.post_transform is None else self.post_transform(img)

        if self.residual_target:
            target = noisy_img - img
        else:
            target = img

        if self.mask_generator is not None:
            mask = self.mask_generator()
            return noisy_img, target, mask

        return noisy_img, target

