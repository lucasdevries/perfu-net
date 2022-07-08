import numpy as np
import logging
import glob
import torch
from torchvision import transforms
from einops import rearrange
from datasets.augmentation import RotateAngleA, HorizontalFlip, WindowSelection#, WindowSelectionRandom, Padding_to_32
from torch.utils.data import DataLoader, Dataset
import SimpleITK as sitk
from utils.train_utils import seed_worker
import os
import matplotlib.pyplot as plt
import time
import random
class CTPdataset(Dataset):
    def __init__(self, config, base_dir, validation=False, transform=None, fold=None):
        self.transform = transform
        self.validation = validation
        self.base_dir = base_dir
        self.fold_file = self.base_dir + fold + ".txt"
        self.CTP_dir = self.base_dir + 'CTP'
        self.mask_dir = self.base_dir + 'MASK'
        self.file_extension = config.file_extension
        self.img_size = config.img_size
        self.clip_length = config.clip_length
        self.input_channels = config.input_channels
    def __len__(self):
        paths = sorted(glob.glob(self.CTP_dir + '{}*{}'.format(os.sep, self.file_extension)))
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
        # print(torch.rand(2))

        CTP_name = sorted(glob.glob(self.CTP_dir + '{}*{}'.format(os.sep, self.file_extension)))
        mask_name = sorted(glob.glob(self.mask_dir + '{}*{}'.format(os.sep, self.file_extension)))
        if self.validation:
            CTP_name = [x for x in CTP_name if x.split(os.sep)[-1].split("_")[1] in validation_cases][idx]
            mask_name = [x for x in mask_name if x.split(os.sep)[-1].split("_")[1] in validation_cases][idx]
        else:
            CTP_name = [x for x in CTP_name if x.split(os.sep)[-1].split("_")[1] not in validation_cases][idx]
            mask_name = [x for x in mask_name if x.split(os.sep)[-1].split("_")[1] not in validation_cases][idx]
        case_name_ctp = CTP_name.split(os.sep)[-1]
        case_name_mask = mask_name.split(os.sep)[-1]
        assert(case_name_ctp == case_name_mask)
        if self.file_extension == '.npy':
            if self.input_channels == 1:
                with open(CTP_name, 'rb') as f:
                    ctp = np.load(f)[:, :1, :, :]
                ctp = rearrange(ctp, 't c h w -> h w t c', c=1)
            elif self.input_channels == 2:
                with open(CTP_name, 'rb') as f:
                    ctp = np.load(f)
                ctp = rearrange(ctp, 't c h w -> h w t c', c=2)
            else:
                raise ValueError("Number of input channels not correct...")
            mask = np.load(mask_name)
        else:
            if self.input_channels == 1:
                ctp = sitk.GetArrayFromImage(sitk.ReadImage(CTP_name))[:, :1, :, :]
                ctp = rearrange(ctp, 't c h w -> h w t c', c=1)
            elif self.input_channels == 2:
                ctp = sitk.GetArrayFromImage(sitk.ReadImage(CTP_name))
                ctp = rearrange(ctp, 't c h w -> h w t c', c=2)
            else:
                raise ValueError("Number of input channels not correct...")
            mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_name))
        # print('data loading no augment: ', time.time() - start)
        sample = {'ctp': ctp, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        ctp = torch.Tensor(sample['ctp'])
        mask = torch.Tensor(sample['mask'])
        # print(case_name_ctp)
        return case_name_ctp, ctp, mask


class CTPDataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("DataLoader")
        self.base_dir = self.config.data_folder
        self.fold = self.config.fold
        self.img_size = self.config.img_size
        self.clip_length = self.config.clip_length
        self.input_channels = self.config.input_channels
        if self.config.augmentations:
            print('Using augmentations...')
            self.train_transforms = transforms.Compose([
                    WindowSelection(self.clip_length),
                    HorizontalFlip(self.config.hflip_prob),
                    RotateAngleA(self.config.rotation_prob),
                ])
        else:
            print('Not using augmentations...')
            self.train_transforms = transforms.Compose([
                    WindowSelection(self.clip_length),
                ])
        self.val_transforms = transforms.Compose([
                WindowSelection(self.clip_length),
            ])
        self.val_transforms_window = transforms.Compose([
            ])
        if self.config.data_mode == "cv":
            self.logger.info("Loading DATA using cross-validation")
            self.logger.info("Validate on fold {}".format(self.fold))
            train_set = CTPdataset(config=self.config,
                                     base_dir=self.base_dir,
                                     validation=False,
                                     transform=self.train_transforms,
                                     fold=self.fold)

            valid_set = CTPdataset(config=self.config,
                                     base_dir=self.base_dir,
                                     validation=True,
                                     transform=self.val_transforms,
                                     fold=self.fold)
            # valid_set_window = CTPdataset(config=self.config,
            #                          base_dir=self.base_dir,
            #                          validation=True,
            #                          transform=self.val_transforms_window,
            #                          fold=self.fold)

            print(len(train_set), len(valid_set))
            self.train_loader = DataLoader(train_set,
                                           batch_size=self.config.batch_size_train,
                                           shuffle=True, pin_memory=True, num_workers=0, worker_init_fn=seed_worker)
            self.valid_loader = DataLoader(valid_set,
                                           batch_size=self.config.batch_size_val,
                                           shuffle=False, pin_memory=True, num_workers=0, worker_init_fn=seed_worker)
            # self.valid_loader_window = DataLoader(valid_set_window,
            #                                batch_size=self.config.batch_size_val,
            #                                shuffle=False, pin_memory=True, num_workers=0, worker_init_fn=seed_worker)
            self.train_iterations = len(self.train_loader)
            self.valid_iterations = len(self.valid_loader)
        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

    def finalize(self):
        pass
