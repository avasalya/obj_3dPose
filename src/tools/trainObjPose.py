#!/usr/bin/env python

'''
Neural network to obtain the rotation from the 2D image of a tool.

Attributes:
    use_gpu (TYPE): Description
'''

from __future__ import print_function, division

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset, DataLoader
import ImageDataConvert as idc

from torchvision import transforms, utils

import numpy as np
from sys import argv
from os.path import isfile
from torchvision import models
import matplotlib.pyplot as plt
import time
import pandas as pd
from skimage import io, transform
import os
plt.ion()   # interactive mode
#use_gpu =  False
use_gpu =  torch.cuda.is_available()



# DATA
#  1.transform
#  2.dataset
#  3.datasetloader


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, rotation = sample['image'], sample['rotation']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        image = (image.astype(np.float32) / 128) - 1


        return {'image': torch.from_numpy(image).float(), 'rotation': torch.from_numpy(rotation.squeeze()).float()}

class ObjectPoseDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None, target_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.rotation = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.rotation)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.rotation.iloc[idx, 0])
        #image = io.imread(img_name)

        image = cv2.imread(img_name)
        image = cv2.resize(image,(227,227))

        #image = (image.astype(np.float32) / 128) - 1
        image = idc.cv2torch(image)
        image = image.type(torch.FloatTensor)
        image = (image / 128) - 1


        rotation = self.rotation.iloc[idx, 1:].as_matrix()
        rotation = rotation.astype('float').reshape(1, 9)
        rotation = torch.from_numpy(rotation.squeeze()).float()

        #image = torch.from_numpy(image).float()

        return image, rotation

def load_data(dataset_path):
    """
    Function that loads and preprocesses the data in
    the dataset file.
    Returns 2 torch dataloaders.

    Args:
        dataset_path (str): Path to the dataset

    Returns:
        dict: Dictionary containing the training and
              validation pytorch dataloaders.
    """

    """
    x_train, x_test, y_train, y_test = np.load(dataset_path)


    # Preprocess
    x_train = (x_train.astype(np.float32)/128)-1
    x_test = (x_test.astype(np.float32)/128)-1

    y_train = y_train.astype(np.float32).squeeze()
    y_test = y_test.astype(np.float32).squeeze()

    # Load Tensors
    x_trainT = torch.Tensor(x_train)
    y_trainT = torch.Tensor(y_train)
    x_testT = torch.Tensor(x_test)
    y_testT = torch.Tensor(y_test)

    trainset = TensorDataset(x_trainT, y_trainT)
    validset = TensorDataset(x_testT, y_testT)
    """
    # TRAIN FUNCTION

    label_train_file = dataset_path +'/train/label.csv'
    label_val_file = dataset_path + '/val/label.csv'

    path_train = dataset_path + '/train/'
    path_val = dataset_path + '/val/'

    #transform = ToTensor()

    trainset_transformed = ObjectPoseDataset(csv_file= label_train_file, root_dir = path_train)#, transform= transform)
    validset_transformed = ObjectPoseDataset(csv_file= label_val_file, root_dir = path_val)#, transform= transform)
    trainloader = torch.utils.data.DataLoader(trainset_transformed, batch_size=10, shuffle=True, num_workers=2 )
    validloader = torch.utils.data.DataLoader(validset_transformed, batch_size=10, shuffle=False, num_workers=2)
    return {'train': trainloader, 'val': validloader}



# Net

def alexnet(numPred=1000):

    alexnet = models.alexnet(pretrained=True)
    in_feats = alexnet.classifier._modules['6'].in_features
    alexnet.classifier._modules['6'] = nn.Linear(in_feats, numPred)
    alexnet.train(False)

    if use_gpu:
        alexnet = alexnet.cuda()

    return alexnet

def resnet18(numPred=1000):

    resnet18 = models.resnet18(pretrained=True)


    in_feats = resnet18.fc.in_features

    #in_feats = resnet18.classifier._modules['6'].in_features
    resnet18.fc = nn.Linear(in_feats, numPred)
    print(resnet18)

    #resnet18.train(False)

    if use_gpu:
        resnet18 = resnet18.cuda()

    return resnet18



def train(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    """
    Function to train a model.

    Args:
        model (torch.nn.Module): Torch Neural Network model.
        dataloaders (dict): Dictionary containing the training and
              validation pytorch dataloaders.
        criterion (torch.nn.Module): Loss Criterion (e.g. MSELoss)
        optimizer (torch.nn.Module): Optimizer algorithm (e.g. SGD, Adam)
        scheduler (torch.nn.Module): Scheduler for decreasing Learnig Rate
        num_epochs (int, optional): Number of training epochs (default=25)

    Returns:
        TYPE: Description
    """
    since = time.time()

    best_model_wts = model.state_dict()
    best_loss = 10000.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            itr = 0
            for data in dataloaders[phase]:
                itr += 1
                # get the inputs
                inputs, labels = data


                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
            epoch_loss = running_loss / itr  # dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



# main

def main(dataset_path,save_path):
    """
    Function that loads an Alexnet model edits the last layer and trains it.

    Args:
        dataset_path (str): Path to the dataset npy file.
    """
    dataloaders = load_data(dataset_path)

    #net = resnet18(9)
    net = alexnet(9)

    criterion = nn.MSELoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 20 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # TRAIN
    model = train(net, dataloaders, criterion, optimizer, exp_lr_scheduler, num_epochs=50)

    torch.save(model.state_dict(), save_path + ".dat")
    print('Model saved as:' + save_path)


if __name__ == '__main__':

    dataset_path = argv[1]
    save_path = argv[2]
    #assert isfile(dataset_path)
    main(dataset_path, save_path)

