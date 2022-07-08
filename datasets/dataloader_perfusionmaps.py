import numpy as np
import logging
import glob
import torch
from torchvision import transforms
from einops import rearrange
from datasets.augmentation_perfusionmaps import RotateAngleA, HorizontalFlip
from torch.utils.data import DataLoader, Dataset
import SimpleITK as sitk
from utils.train_utils import seed_worker, read_file
import os
import random
import matplotlib.pyplot as plt
import time
import random
class PerfusionDataset(Dataset):
    def __init__(self, config, base_dir, validation=False, transform=None, fold=None):
        self.transform = transform
        self.validation = validation
        self.base_dir = base_dir
        self.fold_file = self.base_dir + fold + ".txt"
        self.CT_dir = self.base_dir + 'CT'
        self.CBF_dir = self.base_dir + 'CBF'
        self.CBV_dir = self.base_dir + 'CBV'
        self.Tmax_dir = self.base_dir + 'Tmax'
        self.MTT_dir = self.base_dir + 'MTT'
        self.mask_dir = self.base_dir + 'MASK'

        self.file_extension = config.file_extension
        self.img_size = config.img_size
        self.input_channels = config.input_channels
    def __len__(self):
        paths = sorted(glob.glob(self.CT_dir + '{}*{}'.format(os.sep, self.file_extension)))
        validation_cases = np.loadtxt(self.fold_file, delimiter=",")
        validation_cases = [str(int(x)).zfill(2) for x in validation_cases]
        if self.validation:
            paths = [x for x in paths if x.split(os.sep)[-1].split("_")[1] in validation_cases]
        else:
            paths = [x for x in paths if x.split(os.sep)[-1].split("_")[1] not in validation_cases]
        return len(paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Load the correct files
        validation_cases = np.loadtxt(self.fold_file, delimiter=",")
        validation_cases = [str(int(x)).zfill(2) for x in validation_cases]

        CT_name = sorted(glob.glob(self.CT_dir + '{}*{}'.format(os.sep, self.file_extension)))
        CBF_name = sorted(glob.glob(self.CBF_dir + '{}*{}'.format(os.sep, self.file_extension)))
        CBV_name = sorted(glob.glob(self.CBV_dir + '{}*{}'.format(os.sep, self.file_extension)))
        Tmax_name = sorted(glob.glob(self.Tmax_dir + '{}*{}'.format(os.sep, self.file_extension)))
        MTT_name = sorted(glob.glob(self.MTT_dir + '{}*{}'.format(os.sep, self.file_extension)))
        mask_name = sorted(glob.glob(self.mask_dir + '{}*{}'.format(os.sep, self.file_extension)))
        if self.validation:
            CT_name = [x for x in CT_name if x.split(os.sep)[-1].split("_")[1] in validation_cases][idx]
            CBF_name = [x for x in CBF_name if x.split(os.sep)[-1].split("_")[1] in validation_cases][idx]
            CBV_name = [x for x in CBV_name if x.split(os.sep)[-1].split("_")[1] in validation_cases][idx]
            Tmax_name = [x for x in Tmax_name if x.split(os.sep)[-1].split("_")[1] in validation_cases][idx]
            MTT_name = [x for x in MTT_name if x.split(os.sep)[-1].split("_")[1] in validation_cases][idx]
            mask_name = [x for x in mask_name if x.split(os.sep)[-1].split("_")[1] in validation_cases][idx]
        else:
            CT_name = [x for x in CT_name if x.split(os.sep)[-1].split("_")[1] not in validation_cases][idx]
            CBF_name = [x for x in CBF_name if x.split(os.sep)[-1].split("_")[1] not in validation_cases][idx]
            CBV_name = [x for x in CBV_name if x.split(os.sep)[-1].split("_")[1] not in validation_cases][idx]
            Tmax_name = [x for x in Tmax_name if x.split(os.sep)[-1].split("_")[1] not in validation_cases][idx]
            MTT_name = [x for x in MTT_name if x.split(os.sep)[-1].split("_")[1] not in validation_cases][idx]
            mask_name = [x for x in mask_name if x.split(os.sep)[-1].split("_")[1] not in validation_cases][idx]

        case_name_mask = mask_name.split(os.sep)[-1]
        case_names = [CT_name.split(os.sep)[-1], CBF_name.split(os.sep)[-1], CBV_name.split(os.sep)[-1],
                      Tmax_name.split(os.sep)[-1], MTT_name.split(os.sep)[-1]]
        for name in case_names:
            assert(name == case_name_mask)

        if self.file_extension == '.npy':
            ct = read_file(CT_name)
            cbf = read_file(CBF_name)
            cbv = read_file(CBV_name)
            tmax = read_file(Tmax_name)
            mtt = read_file(MTT_name)
            mask = read_file(mask_name)

            if self.input_channels == 1:
                ct, cbf, cbv, tmax, mtt = ct[:1], cbf[:1], cbv[:1], tmax[:1], mtt[:1]
            elif self.input_channels != 2:
                raise ValueError("Number of input channels not correct...")

            assert(cbf.shape[0] == self.input_channels)
        else:
           raise ValueError("SimpleITK version not implemented... Use .npy extension.")
        image = rearrange([ct, cbf, cbv, tmax, mtt], 'b c h w -> h w (b c)')
        sample = {'image': image, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        image = torch.Tensor(sample['image'])
        mask = torch.Tensor(sample['mask'])
        return case_name_mask, image, mask


class PerfusionDataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("PerfusionDataLoader")
        self.base_dir = self.config.data_folder
        self.fold = self.config.fold
        self.img_size = self.config.img_size
        self.clip_length = self.config.clip_length
        self.input_channels = self.config.input_channels
        if self.config.augmentations:
            print('Using augmentations...')
            self.train_transforms = transforms.Compose([
                    HorizontalFlip(self.config.hflip_prob),
                    RotateAngleA(self.config.rotation_prob),
                ])
        else:
            print('Not using augmentations...')
            self.train_transforms = transforms.Compose([
                ])
        self.val_transforms = transforms.Compose([
            ])
        if self.config.data_mode == "cv":
            self.logger.info("Loading DATA using cross-validation")
            self.logger.info("Validate on fold {}".format(self.fold))
            train_set = PerfusionDataset(config=self.config,
                                     base_dir=self.base_dir,
                                     validation=False,
                                     transform=self.train_transforms,
                                     fold=self.fold)

            valid_set = PerfusionDataset(config=self.config,
                                     base_dir=self.base_dir,
                                     validation=True,
                                     transform=self.val_transforms,
                                     fold=self.fold)

            print(len(train_set), len(valid_set))
            self.train_loader = DataLoader(train_set,
                                           batch_size=self.config.batch_size_train,
                                           shuffle=True, pin_memory=True, num_workers=0, worker_init_fn=seed_worker)
            self.valid_loader = DataLoader(valid_set,
                                           batch_size=self.config.batch_size_val,
                                           shuffle=False, pin_memory=True, num_workers=0, worker_init_fn=seed_worker)

            self.train_iterations = len(self.train_loader)
            self.valid_iterations = len(self.valid_loader)
        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

    def finalize(self):
        pass
