"""
Main Agent for PerfU-Net
"""
import numpy as np
from tqdm import tqdm
import shutil, os
import torch
import logging
from einops import rearrange
from graphs.losses.losses_w import loss_module

from graphs.models.PerfUNetVariant import PerfUNetVariant
from graphs.models.PerfUNet import PerfUNet
from graphs.models.RBNet import RBNet

from utils.metrics import DiceMetric
from utils.train_utils import adjust_learning_rate, adjust_alpha

from torchsummary import summary
from datasets.dataloader import CTPDataLoader
from utils.metrics import AverageMeter
from utils.valid_utils import SaveProbMap, getMetrics
import wandb

logging.basicConfig(level=logging.DEBUG)
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)


class Agent:
    def __init__(self):
        self.PID = os.getpid()
        self.config = wandb.config
        self.logger = logging.getLogger(str(self.PID))
        # Check is cuda is available or not
        self.is_cuda = torch.cuda.is_available()
        # Construct the flag and make sure that cuda is available
        self.cuda = self.is_cuda & self.config.cuda
        if self.cuda:
            print('Available devices ', torch.cuda.device_count())
            print('Current cuda device ', torch.cuda.current_device())
            self.device = torch.device("cuda:{}".format(self.config.gpu_device))
            torch.cuda.set_device("cuda:{}".format(self.config.gpu_device))
            self.logger.info("Operation will be on *****GPU-CUDA{}***** ".format(self.config.gpu_device))
            print(self.device)

        else:
            self.device = torch.device("cpu")
            self.logger.info("Operation will be on *****CPU***** ")
        # Create an instance from the Model
        if self.config.modelname == 'PerfUNetVariant':
            self.model = PerfUNetVariant(self.config)
        if self.config.modelname == 'PerfUNet':
            self.model = PerfUNet(self.config)
        if self.config.modelname == 'RBNet':
            self.model = RBNet(self.config)

        if self.config.input_channels == 2:
            print(summary(self.model, input_size=(2, 16, 256, 256), device='cpu'))
        else:
            print(summary(self.model, input_size=(1, 16, 256, 256), device='cpu'))

        self.model = self.model.float().to(self.device)

        self.nonlin = torch.nn.Softmax(dim=1)
        # Create an instance from the data loader
        self.data_loader = CTPDataLoader(self.config)
        # loss function
        self.criterion = loss_module(self.config)
        self.loss_alpha = self.config.loss_alpha

        # optimizer
        self.learning_rate = self.config.learning_rate
        self.logger.info("Initial learning rate is {}".format(self.learning_rate))
        if self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=self.learning_rate)
        elif self.config.optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                               lr=self.learning_rate,
                                               weight_decay=0.01)
        elif self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=self.learning_rate,
                                             momentum=float(self.config.momentum),
                                             weight_decay=0.0,
                                             nesterov=True)

        # initialize my counters
        self.current_epoch = 0
        self.current_iteration = 0
        self.current_iteration_val = 0
        self.best_valid_dice = -999
        self.best_valid_dice_full_scan = -999
        # Set evaluation metric
        self.dice = DiceMetric(apply_nonlin=torch.nn.Softmax(dim=1))

    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=0, save_wandb=False, is_best_full_scan=False):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + filename)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            if not save_wandb:
                shutil.copyfile(self.config.checkpoint_dir + filename,
                                self.config.checkpoint_dir + 'model_best.pth.tar')
            else:
                shutil.copyfile(self.config.checkpoint_dir + filename,
                                self.config.checkpoint_dir + 'model_best.pth.tar')
                shutil.copyfile(self.config.checkpoint_dir + filename,
                                os.path.join(wandb.run.dir, 'model_best.pth.tar'))
        # If it is the best based on whole scan copy it to another file 'model_best_full_scan.pth.tar'
        if is_best_full_scan:
            if not save_wandb:
                shutil.copyfile(self.config.checkpoint_dir + filename,
                                self.config.checkpoint_dir + 'model_best_full_scan.pth.tar')
            else:
                shutil.copyfile(self.config.checkpoint_dir + filename,
                                self.config.checkpoint_dir + 'model_best_full_scan.pth.tar')
                shutil.copyfile(self.config.checkpoint_dir + filename,
                                os.path.join(wandb.run.dir, 'model_best_full_scan.pth.tar'))

    def load_checkpoint(self, filename):
        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def run(self):
        """
        This function will the operator
        :return:
        """
        try:
            if self.config.mode == 'test':
                self.validate()
            else:
                print('Process ID: {}'.format(self.PID))
                self.train()
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def full_validator(self):
        valid_metrics = self.validate_3d()
        is_best_full_scan = valid_metrics['3d_dice'] > self.best_valid_dice_full_scan
        if is_best_full_scan:
            self.logger.info("############# New best Full Scan #############")
            self.best_valid_dice_full_scan = valid_metrics['3d_dice']
            self.save_checkpoint(is_best_full_scan=is_best_full_scan, save_wandb=False)

    def train(self):
        """
        Main training function, with per-epoch model saving
        """
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()
            # self.scheduler.step()
            if self.current_epoch % 1 == 0:
                valid_loss, valid_acc = self.validate()
                is_best = valid_acc > self.best_valid_dice
                if is_best:
                    self.logger.info("############# New best #############")
                    self.best_valid_dice = valid_acc
                    self.full_validator()
                    self.save_checkpoint(is_best=is_best, save_wandb=False)

            if ((self.current_epoch % 5 == 0) & (self.current_epoch > 0)) | (
                    self.current_epoch == self.config.max_epoch - 1):
                self.full_validator()

    def train_one_epoch(self):
        """
        One epoch training function
        """
        # Initialize tqdm
        tqdm_batch = tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
                          desc="Epoch number -{}-".format(self.current_epoch))
        # Set the model to be in training mode
        self.model.train()
        # Initialize your average meters
        epoch_loss = AverageMeter()
        dice_score_total = AverageMeter()

        current_batch = 0
        for case_name, inputs, masks in tqdm_batch:
            self.optimizer.zero_grad()

            lr = adjust_learning_rate(self.optimizer, self.current_epoch, self.config, batch=current_batch,
                                      nBatch=self.data_loader.train_iterations, method=self.config.lr_type)
            lr = self.optimizer.param_groups[0]['lr']

            inputs = rearrange(inputs, 'b h w t c -> b c t h w').to(self.device)
            masks = masks.to(self.device)
            output = self.model(inputs.float())

            if self.config.criterion in ['wCE', 'wFocal', 'gDicewCE']:
                # TODO make function to determine weights
                values, counts = torch.unique(masks, return_counts=True)
                weights = torch.zeros((2))
                for v, c in zip(values, counts):
                    weights[int(v)] = c
                weights = 1 / ((weights + 1) / (torch.sum(weights) + self.config.num_classes))

                if self.config.criterion == 'gDicewCE' and self.config.loss_schedule:
                    self.current_loss_alpha = adjust_alpha(start_alpha=self.loss_alpha, epoch=self.current_epoch,
                                                      start_epoch=30, config=self.config)
                    cur_loss = self.criterion(output, masks.long(), weights, alpha=self.current_loss_alpha)
                else:
                    cur_loss = self.criterion(output, masks.long(), weights)
            elif self.config.criterion == 'gDice':
                cur_loss = self.criterion(output.float(), masks.long())
            else:
                raise ValueError('Criterion is not defined...')

            dice_score = self.dice(output, masks)

            dice_score_total.update(dice_score.item(), n=len(inputs))
            epoch_loss.update(cur_loss.item(), len(inputs))

            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')
            # optimizer
            cur_loss.backward()
            self.optimizer.step()

            self.current_iteration += 1
            current_batch += 1

        metrics = {
            "training_loss": epoch_loss.avg,
            "learning_rate_epoch": self.optimizer.param_groups[0]['lr'],
            "training_dice": dice_score_total.avg,
        }
        if self.config.loss_schedule:
            metrics['loss_alpha'] = self.current_loss_alpha

        wandb.log(metrics, step=self.current_epoch)
        tqdm_batch.close()

        self.logger.info("Training at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            metrics['training_loss']) + "  |  dice: " + str(metrics['training_dice']))

    def validate(self):
        """
        One epoch validation
        :return:
        """
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                          desc="Validation at -{}-".format(self.current_epoch))
        # set the model in training mode
        self.model.eval()

        epoch_loss = AverageMeter()
        dice_score_total = AverageMeter()

        for case_name, inputs, masks in tqdm_batch:
            inputs = rearrange(inputs, 'b h w t c -> b c t h w').to(self.device)
            masks = masks.to(self.device)
            with torch.no_grad():
                output = self.model(inputs.float())

            if self.config.criterion in ['wCE', 'wFocal', 'gDicewCE']:
                cur_loss = self.criterion(output, masks.long(), None)
            elif self.config.criterion == 'gDice':
                cur_loss = self.criterion(output.float(), masks.long())
            else:
                raise ValueError('Criterion is not defined...')

            # cur_loss = self.criterion(output.float(), masks.long())
            dice_score = self.dice(output, masks)

            epoch_loss.update(cur_loss.item(), len(inputs))
            dice_score_total.update(dice_score.item(), n=len(inputs))

            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during validation...')
            self.current_iteration_val += 1

        metrics = {"validation_loss": epoch_loss.avg,
                   "validation_dice": dice_score_total.avg,
                   }
        wandb.log(metrics, step=self.current_epoch)
        self.logger.info(
            "Validation at epoch-" + str(self.current_epoch) + " | " + "dice:  " + str(metrics['validation_dice']))
        tqdm_batch.close()

        return metrics['validation_loss'], metrics['validation_dice']

    def validate_3d(self, final=False):
        """
        Final validation: calculate evaluation metrics on validation set,
        generate some images and save model graph to tensorboard.
        :return:
        """
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                          desc="3D Validation at -{}-".format(self.current_epoch))

        # set the model in training mode
        self.model.eval()

        for name, inputs, masks in tqdm_batch:
            inputs = rearrange(inputs, 'b h w t c -> b c t h w').to(self.device)
            with torch.no_grad():
                output = self.model(inputs)
                output = self.nonlin(output)
            SaveProbMap(output[0, 1, ...], wandb.run.dir, name[0], self.config.file_extension)
        # detrmine the metric scores and log to wandb

        metrics = getMetrics(self.config, wandb.run.dir,
                             best_up_to_now=self.best_valid_dice_full_scan,
                             epoch=self.current_epoch)

        self.logger.info(
            "3D Validation at epoch-" + str(self.current_epoch) + " | " + "dice:  " + str(metrics['3d_dice']))
        if final:
            wandb.log(metrics, step=self.config.max_epoch)
        else:
            wandb.log(metrics, step=self.current_epoch)
        return metrics
    def validate_3d_window(self, start=0):
        """
        Final validation: calculate evaluation metrics on validation set,
        generate some images and save model graph to tensorboard.
        :return:
        """
        tqdm_batch = tqdm(self.data_loader.valid_loader_window, total=self.data_loader.valid_iterations,
                          desc="3D Validation at -{}-".format(self.current_epoch))

        # set the model in training mode
        self.model.eval()
        print(start)
        for name, inputs, masks in tqdm_batch:
            inputs = rearrange(inputs, 'b h w t c -> b c t h w').to(self.device)
            print(inputs.shape)
            with torch.no_grad():
                output = self.model(inputs[:,:,start:start+self.config.clip_length, ...])
                output = self.nonlin(output)
            SaveProbMap(output[0, 1, ...], wandb.run.dir, name[0], self.config.file_extension)
        # detrmine the metric scores and log to wandb
        metrics = getMetrics(self.config, wandb.run.dir,
                             best_up_to_now=self.best_valid_dice_full_scan,
                             epoch=self.current_epoch)
        return metrics
    def final_validate(self):
        """
        Final validation: calculate evaluation metrics on validation set,
        generate some images and save model graph to tensorboard.
        :return:
        """
        metrics = self.validate_3d(final=True)
        final_metrics = {
            'final_3d_dice': metrics['3d_dice'],
            'final_3d_recall': metrics['3d_recall'],
            'final_3d_precision': metrics['3d_precision'],
            'final_3d_volume': metrics['3d_volume'],
            'final_3d_surface_dice': metrics['3d_surface_dice'],
            'final_3d_hd95': metrics['3d_hd95'],
            'final_3d_hd100': metrics['3d_hd100'],
            'final_3d_abs_volume': metrics['3d_abs_volume'],
        }
        wandb.log(final_metrics)
    def final_validate_sliding_window(self):
        """
        Final validation: calculate evaluation metrics on validation set,
        generate some images and save model graph to tensorboard.
        :return:
        """
        for start in [0, 4, 8, 12]:
            print(f'Validation starting at {start}')
            metrics = self.validate_3d_window(start=start)
            final_metrics = {
                f'final_3d_dice_{start}': metrics['3d_dice'],
                f'final_3d_recall_{start}': metrics['3d_recall'],
                f'final_3d_precision_{start}': metrics['3d_precision'],
                f'final_3d_volume_{start}': metrics['3d_volume'],
                f'final_3d_surface_dice_{start}': metrics['3d_surface_dice'],
                f'final_3d_hd95_{start}': metrics['3d_hd95'],
                f'final_3d_hd100_{start}': metrics['3d_hd100'],
                f'final_3d_abs_volume_{start}': metrics['3d_abs_volume'],
            }
            wandb.log(final_metrics)
    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint(save_wandb=False)

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint('model_best_full_scan.pth.tar')

        # set the model in training mode
        self.model.eval()

        # perform final validation
        self.logger.info("Inference with best found model.. Thank you")
        self.final_validate()
        # self.final_validate_sliding_window()
        # save best model in wandb folder
        torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, 'model_best_inference.pth.tar'))
        # save best model on full scann in wandb folder
        self.load_checkpoint('model_best_full_scan.pth.tar')
        torch.save(self.model.state_dict(), os.path.join(wandb.run.dir, 'model_best_full_scan_inference.pth.tar'))

        # remove models form experiments folder
        os.remove(os.path.join(self.config.checkpoint_dir, 'model_best.pth.tar'))
        os.remove(os.path.join(self.config.checkpoint_dir, 'model_best_full_scan.pth.tar'))
        os.remove(os.path.join(self.config.checkpoint_dir, 'checkpoint.pth.tar'))

        self.data_loader.finalize()
