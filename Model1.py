import torch
import os
import glob
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from Common import *
# CREATE THE MODEL
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_layer_1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv_layer_2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv_layer_3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv_layer_4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv_layer_5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv_layer_6 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, kernel_size=3, padding='same'),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )


        self.fc_layer_1 = torch.nn.Sequential(
            torch.nn.Linear(3*3*256, 128), #torch.nn.Linear(7*7*128, 128),
            torch.nn.Dropout(),
            torch.nn.ReLU()
        )
        self.fc_layer_2 = torch.nn.Linear(128, len(TYPESDICT))
    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_4(x)
        x = self.conv_layer_5(x)
        #x = self.conv_layer_6(x)
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc_layer_1(x)
        x = self.fc_layer_2(x)
        return x