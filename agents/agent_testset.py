"""
Main Agent for SiT
"""
from tqdm import tqdm
import shutil, os
import torch
import logging
from einops import rearrange
from graphs.models.PerfUNet import PerfUNet
from torchsummary import summary
from datasets.dataloader_testset import CTPDataLoader
from utils.valid_utils import SaveProbMap
import time
import wandb
logging.basicConfig(level=logging.DEBUG)
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)
class AgentTestSet:
    def __init__(self):
        self.PID = os.getpid()
        self.config = wandb.config
        self.logger = logging.getLogger(str(self.PID))
        # Check is cuda is available or not
        self.is_cuda = torch.cuda.is_available()
        # Construct the flag and make sure that cuda is available
        self.cuda = self.is_cuda & self.config.cuda
        if self.cuda:
            self.device = torch.device("cuda:{}".format(self.config.gpu_device))
            torch.cuda.set_device("cuda:{}".format(self.config.gpu_device))
            self.logger.info("Operation will be on *****GPU-CUDA{}***** ".format(self.config.gpu_device))
            print(self.device)

        else:
            self.device = torch.device("cpu")
            self.logger.info("Operation will be on *****CPU***** ")
        # Create an instance from the Model
        if self.config.modelname == 'PerfUNet':
            self.model = PerfUNet(self.config)

        if self.config.input_channels == 2:
            print(summary(self.model, input_size=(2, 16, 256, 256), device='cpu'))
        else:
            print(summary(self.model, input_size=(1, 16, 256, 256), device='cpu'))

        self.model = self.model.float().to(self.device)
        self.model_path = self.config.model_path
        self.nonlin = torch.nn.Softmax(dim=1)
        # Create an instance from the data loader
        self.data_loader = CTPDataLoader(self.config)

    def load_checkpoint(self):
        filename = self.model_path
        filepath = r"test_models\{}\model_best_full_scan_inference.pth.tar".format(filename)
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filepath, map_location='cuda:0')
            # for ix, param in enumerate(self.model.parameters()):
            #     if ix == 0:
            #         print(param)
            self.model.load_state_dict(checkpoint)
            # for ix, param in enumerate(self.model.parameters()):
            #     if ix == 0:
            #         print(param)
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
                self.inference()
            else:
                raise ValueError('Set config to test mode..')
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")


    def inference(self):
        """
        Final validation: calculate evaluation metrics on validation set,
        generate some images and save model graph to tensorboard.
        :return:
        """
        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                          desc="Inference at -{}-".format('test set.'))

        # set the model in training mode
        self.load_checkpoint()
        self.model.eval()
        t0 = time.time()
        for name, inputs in tqdm_batch:
            inputs = rearrange(inputs, 'b h w t c -> b c t h w').to(self.device)
            with torch.no_grad():
                output = self.model(inputs)
                output = self.nonlin(output)
            SaveProbMap(output[0, 1, ...], wandb.run.dir, name[0], self.config.file_extension)
        t1 = time.time() - t0
        self.logger.info("Elapsed time {}".format(t1))
        self.logger.info("Elapsed time {}".format(t1/len(tqdm_batch)))


        # detrmine the metric scores and log to wandb


    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.data_loader.finalize()
