#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:52:25 2020

@author: darp_lord
"""

import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np


class AudioNet(nn.Module):
    
    def __init__(self, n_classes=1251):
        super(AudioNet, self).__init__()
        self.n_classes=n_classes
        self.features=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv1d(in_channels=512, out_channels=2048, kernel_size=(7), stride=(2), padding=1)),
            ('bn1', nn.BatchNorm1d(2048, momentum=0.5)),
            ('relu1', nn.ReLU()),
            ('mpool1', nn.MaxPool1d(kernel_size=(3), stride=(2))),
            ('conv2', nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=(5), stride=(2), padding=1)),
            ('bn2', nn.BatchNorm1d(2048, momentum=0.5)),
            ('relu2', nn.ReLU()),
            ('mpool2', nn.MaxPool1d(kernel_size=(3), stride=(2))),
            ('conv3', nn.Conv1d(in_channels=2048, out_channels=4096, kernel_size=(3), stride=(1), padding=1)),
            ('bn3', nn.BatchNorm1d(4096, momentum=0.5)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv1d(in_channels=4096, out_channels=2048, kernel_size=(3), stride=(1), padding=1)),
            ('bn4', nn.BatchNorm1d(2048, momentum=0.5)),
            ('relu4', nn.ReLU()),
            ('conv5', nn.Conv1d(in_channels=2048, out_channels=2048, kernel_size=(3), stride=(1), padding=1)),
            ('bn5', nn.BatchNorm1d(2048, momentum=0.5)),
            ('relu5', nn.ReLU()),
            ('mpool5', nn.MaxPool1d(kernel_size=(3), stride=(2))),
            ('fc6', nn.Conv1d(in_channels=2048, out_channels=4096, kernel_size=(1), stride=(1,1))),
            ('bn6', nn.BatchNorm1d(4096, momentum=0.5)),
            ('relu6', nn.ReLU()),
            ('apool6', nn.AdaptiveAvgPool1d((1))),
            ('flatten', nn.Flatten())]))
            
        self.classifier=nn.Sequential(OrderedDict([
            ('fc7', nn.Linear(4096, 1024)),
            #('drop1', nn.Dropout()),
            ('relu7', nn.ReLU()),
            ('fc8', nn.Linear(1024, n_classes))]))
    
    def forward(self, inp):
        inp = inp.squeeze()
        inp=self.features(inp)
        #inp=inp.view(inp.size()[0],-1)
        inp=self.classifier(inp)
        return inp

if __name__=="__main__":
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=AudioNet(1251)
    model.to(device)
    print(summary(model, (1,512,300)))
