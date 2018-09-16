"""
SIW Data loader, as given in Mnist tutorial
"""

import json
import imageio as io
import matplotlib.pyplot as plt
import torch
import torchvision.utils as v_utils
from torchvision import datasets, transforms
import os

from torch.utils.data import DataLoader, TensorDataset, Dataset


def imshow(image,depth):

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    image = plt.imshow(image)
    ax.set_title('Image')
    ax = fig.add_subplot(1, 2, 1)
    #image = plt.imshow(depth_image)
    plt.tight_layout()
    ax.set_title("Depth Image")
    ax.axis('off')
    plt.show()


class SiwDataset(Dataset):

    def __init__(self,dataset_type,json_path,transform=None):

        self.dataset_type = dataset_type
        self.json_path = json_path
        self.transform = transform
        print(os.getcwd())
        with open(json_path, 'r') as f:
            self.data_json = json.load(f)
        
        
    def __len__(self):
        return(len(self.data_json.keys()))

    def __getitem__(self, idx):

        img_name = list(self.data_json)[idx]
        img_depth_name = self.data_json[img_name]

        image = io.imread(img_name)
        #depth = io.imread(img_depth_name)

        if self.transform:
            image = self.transform(image)
            #depth = self.transform(depth)

        sample = {'image': image, 'depth': None}


        return sample


class SiwDataLoader:
    def __init__(self, config):
        """
        :param config:
        """
        self.config = config

        self.transform_to_tensor = transforms.Compose([transforms.ToTensor()])

        self.train_dataset = SiwDataset(dataset_type="train",json_path="data/train.json",transform=self.transform_to_tensor)
        self.val_dataset = SiwDataset(dataset_type="val",json_path="data/val.json",transform=self.transform_to_tensor)
        self.test_dataset = SiwDataset(dataset_type="test",json_path="data/test.json",transform=self.transform_to_tensor)

        if config.data_mode == "json":
            self.train_loader = DataLoader(self.train_dataset,
                                           batch_size=self.config.batch_size, 
                                           shuffle=True, 
                                           num_workers=self.config.data_loader_workers)

            train_len = len(self.train_dataset)
            self.train_iterations = (train_len + self.config.batch_size - 1) // self.config.batch_size
                                    
            self.val_loader = DataLoader(self.val_dataset,
                                           batch_size=self.config.batch_size, 
                                           shuffle=True, 
                                           num_workers=self.config.data_loader_workers)

            val_len = len(self.val_dataset)
            self.val_iterations = (val_len + self.config.batch_size - 1) // self.config.batch_size

            self.test_loader = DataLoader(self.test_dataset,
                                           batch_size=self.config.batch_size, 
                                           shuffle=True, 
                                           num_workers=self.config.data_loader_workers)

            test_len = len(self.test_dataset)
            self.test_iterations = (test_len + self.config.batch_size - 1) // self.config.batch_size

                                        
    def plot_samples_per_epoch(self, batch, epoch):
        """
        Plotting the batch images
        :param batch: Tensor of shape (B,C,H,W)
        :param epoch: the number of current epoch
        :return: img_epoch: which will contain the image of this epoch
        """
        img_epoch = '{}samples_epoch_{:d}.png'.format(self.config.out_dir, epoch)
        v_utils.save_image(batch,
                           img_epoch,
                           nrow=4,
                           padding=2,
                           normalize=True)
        return io.imread(img_epoch)

    def make_gif(self, epochs):
        """
        Make a gif from a multiple images of epochs
        :param epochs: num_epochs till now
        :return:
        """
        gen_image_plots = []
        for epoch in range(epochs + 1):
            img_epoch = '{}samples_epoch_{:d}.png'.format(self.config.out_dir, epoch)
            try:
                gen_image_plots.append(io.imread(img_epoch))
            except OSError as e:
                pass

        io.mimsave(self.config.out_dir + 'animation_epochs_{:d}.gif'.format(epochs), gen_image_plots, fps=2)

    def finalize(self):
        pass
