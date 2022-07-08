import numpy as np
import logging
import glob
import torch
from torchvision import transforms
from einops import rearrange
from datasets.augmentation_testset import WindowSelection
from torch.utils.data import DataLoader, Dataset
import SimpleITK as sitk
from utils.train_utils import seed_worker
import os

class CTPdataset(Dataset):
    def __init__(self, config, base_dir, validation=False, transform=None):
        self.transform = transform
        self.validation = validation
        self.base_dir = base_dir
        self.CTP_dir = self.base_dir + 'CTPsmoothnotpadded'
        self.file_extension = config.file_extension
        self.img_size = config.img_size
        self.clip_length = config.clip_length
        self.input_channels = config.input_channels
    def __len__(self):
        paths = sorted(glob.glob(self.CTP_dir + '{}*{}'.format(os.sep, self.file_extension)))
        return len(paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        CTP_name = sorted(glob.glob(self.CTP_dir + '{}*{}'.format(os.sep, self.file_extension)))[idx]
        case_name_ctp = CTP_name.split(os.sep)[-1]

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
        else:
            if self.input_channels == 1:
                ctp = sitk.GetArrayFromImage(sitk.ReadImage(CTP_name))[:, :1, :, :]
                ctp = rearrange(ctp, 't c h w -> h w t c', c=1)
            elif self.input_channels == 2:
                ctp = sitk.GetArrayFromImage(sitk.ReadImage(CTP_name))
                ctp = rearrange(ctp, 't c h w -> h w t c', c=2)
            else:
                raise ValueError("Number of input channels not correct...")
        # print('data loading no augment: ', time.time() - start)
        sample = {'ctp': ctp}
        if self.transform:
            sample = self.transform(sample)
        ctp = torch.Tensor(sample['ctp'])
        return case_name_ctp, ctp


class CTPDataLoader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("DataLoader")
        self.base_dir = self.config.data_folder
        self.img_size = self.config.img_size
        self.clip_length = self.config.clip_length
        self.input_channels = self.config.input_channels
        self.val_transforms = transforms.Compose([
                WindowSelection(self.clip_length),
            ])
        if self.config.data_mode == "test":
            self.logger.info("Validate on test-data {}")
            test_set = CTPdataset(config=self.config,
                                     base_dir=self.base_dir,
                                     validation=True,
                                     transform=self.val_transforms)
            print(len(test_set))
            self.valid_loader = DataLoader(test_set,
                                           batch_size=self.config.batch_size_val,
                                           shuffle=False, pin_memory=True, num_workers=0, worker_init_fn=seed_worker)
            self.valid_iterations = len(self.valid_loader)
        else:
            raise Exception("Please specify in the json a specified mode in data_mode")

    def finalize(self):
        pass
