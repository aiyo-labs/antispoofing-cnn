"""
Antispoofing Main agent, as mentioned in the tutorial
"""
import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from agents.base import BaseAgent

from graphs.models.antispoofing import AntiSpoofing
from graphs.losses.L1_loss import L1_loss
from datasets.siw import SiwDataLoader

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics


cudnn.benchmark = True


class AntiSpoofingAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # define models
        self.model = AntiSpoofing()

        # define data_loader
        self.data_loader = SiwDataLoader(config=config)

        # define loss
        self.loss = L1_loss()

        # define optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        self.model = self.model.to(self.device)
        self.loss = self.loss.to(self.device)

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        # Summary Writer
        self.summary_writer = None

    def load_checkpoint(self, file_name):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        filename = self.config.checkpoint_dir + file_name
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                  .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + filename)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        """
        The main operator
        :return:
        """
        assert self.config.mode in ['train', 'test', 'val']
        try:
            if self.config.mode == 'test':
                self.test()
            elif self.config.mode == 'val':
                self.validate()
            else:
                self.train()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """

        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.validate()
            self.save_checkpoint()


    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        # Initialize tqdm
        tqdm_batch = tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
                          desc="Epoch-{}-".format(self.current_epoch))
        self.model.train()

        epoch_loss = AverageMeter()

        for batch_idx, (data, target) in enumerate(tqdm_batch):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = L1_loss(output,target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.config.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch, batch_idx * len(data), len(self.data_loader.train_loader.dataset),
                           100. * batch_idx / len(self.data_loader.train_loader), loss.item()))
            self.current_iteration += 1

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        # Initialize tqdm
        tqdm_batch = tqdm(self.data_loader.val_loader, total=self.data_loader.val_iterations,
                          desc="Epoch-{}-".format(self.current_epoch))

        self.model.eval()
        val_loss = 0
        #correct = 0
        with torch.no_grad():
            for data, target in self.data_loader.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += L1_loss(output, target, reduction='sum').item()  # sum up batch loss
                #pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                #correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(self.data_loader.val_loader.dataset)
        # self.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, len(self.data_loader.test_loader.dataset),
        #     100. * correct / len(self.data_loader.test_loader.dataset)))
        self.logger.info('\nValidation set: Average loss: {:.4f}\n'.format(    
            val_loss))


    def test(self):
        """
        One cycle of model test
        :return:
        """
        # Initialize tqdm
        tqdm_batch = tqdm(self.data_loader.test_loader, total=self.data_loader.test_iterations,
                          desc="Epoch-{}-".format(self.current_epoch))

        self.model.eval()
        test_loss = 0
        #correct = 0
        with torch.no_grad():
            for data, target in self.data_loader.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += L1_loss(output, target, reduction='sum').item()  # sum up batch loss
                #pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                #correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.data_loader.test_loader.dataset)
        # self.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        #     test_loss, correct, len(self.data_loader.test_loader.dataset),
        #     100. * correct / len(self.data_loader.test_loader.dataset)))
        self.logger.info('\Test set: Average loss: {:.4f}\n'.format(    
            test_loss))

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        print("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
        self.data_loader.finalize()
