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
        self.block1=nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=96, kernel_size=(7,7), stride=(2,2), padding=1)),
            ('bn1', nn.BatchNorm2d(96, momentum=0.5)),
            ('relu1', nn.ReLU())]))
        
        self.extract1 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(1,1), stride=(1,1), padding=0)
            
        self.block2=nn.Sequential(OrderedDict([           
            ('mpool1', nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))),
            ('conv2', nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5,5), stride=(2,2), padding=1)),
            ('bn2', nn.BatchNorm2d(256, momentum=0.5)),
            ('relu2', nn.ReLU())]))
            
        self.extract2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=0)
          
        self.block3=nn.Sequential(OrderedDict([              
            ('mpool2', nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))),
            ('conv3', nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3,3), stride=(1,1), padding=1)),
            ('bn3', nn.BatchNorm2d(384, momentum=0.5)),
            ('relu3', nn.ReLU()),
            ('conv4', nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)),
            ('bn4', nn.BatchNorm2d(256, momentum=0.5)),
            ('relu4', nn.ReLU()),
            ('conv5', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=1)),
            ('bn5', nn.BatchNorm2d(256, momentum=0.5)),
            ('relu5', nn.ReLU())]))
        
        self.extract3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=(1,1), padding=0)        
         
        self.block4=nn.Sequential(OrderedDict([         
            ('mpool5', nn.MaxPool2d(kernel_size=(5,3), stride=(3,2))),
            ('fc6', nn.Conv2d(in_channels=256, out_channels=4096, kernel_size=(9,1), stride=(1,1))),
            ('bn6', nn.BatchNorm2d(4096, momentum=0.5)),
            ('relu6', nn.ReLU())]))  
         
        self.block5=nn.Sequential(OrderedDict([       
            ('apool6', nn.AdaptiveAvgPool2d((1,1))),
            ('flatten', nn.Flatten())]))
            
        self.classifier=nn.Sequential(OrderedDict([
            ('fc7', nn.Linear(4096, 1024)),
            #('drop1', nn.Dropout()),
            ('relu7', nn.ReLU()),
            ('fc8', nn.Linear(1024, n_classes))]))
    
    def forward(self, inp):
        inp=self.block1(inp)
        feat1 = self.extract1(inp).relu().mean(-1)
        feat1 = torch.flatten(feat1, start_dim=1)
        
        inp=self.block2(inp)
        feat2 = self.extract2(inp).relu().mean(-1)
        feat2 = torch.flatten(feat2, start_dim=1)

        
        inp=self.block3(inp)
        feat3 = self.extract3(inp).relu().mean(-1)
        feat3 = torch.flatten(feat3, start_dim=1)
        
        
        inp=self.block4(inp)
        inp=self.block5(inp)
        
        inp = torch.cat([inp, feat1, feat2, feat3], 1)
        print(inp.shape)
        
        inp=self.classifier(inp)
        return inp

if __name__=="__main__":
    from torchsummary import summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=AudioNet(1251)
    model.to(device)
    print(summary(model, (1,512,300)))
