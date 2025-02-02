#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:12:22 2020

@author: darp_lord
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler, SGD, Adam
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm
import signal_utils as sig
from scipy.io import wavfile
from audionet import AudioNet
import argparse
import random



LR=0.01
B_SIZE=100
N_EPOCHS=150
N_CLASSES=1251
transformers=transforms.ToTensor()
LOCAL_DATA_DIR="data/"
MODEL_DIR="models/"

class AudioDataset(Dataset):
    def __init__(self, csv_file, data_dir, croplen=48320, is_train=True):
        if isinstance(csv_file, str):
            csv_file=pd.read_csv(csv_file)
        assert isinstance(csv_file, pd.DataFrame), "Invalid csv path or dataframe"
        self.X=csv_file['Path'].values
        self.y=(csv_file['Label'].values-10001).astype(int)
        self.data_dir=data_dir
        self.is_train=is_train
        self.croplen=croplen

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        label=self.y[idx]
        sr, audio=wavfile.read(os.path.join(self.data_dir,self.X[idx]))
        if(self.is_train):
            start=np.random.randint(0,audio.shape[0]-self.croplen+1)
            audio=audio[start:start+self.croplen]
        audio=sig.preprocess(audio).astype(np.float32)
        audio=np.expand_dims(audio, 2)
        return transformers(audio), label

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul(100.0 / batch_size)).item())
        return res

def test(model, Dataloaders):
    corr1=0
    corr5=0
    counter=0
    top1=0
    top5=0
    for Dataloader in Dataloaders:
        sub_counter=0
        sub_top1=0
        sub_top5=0
        for audio, labels in Dataloader:
            audio = audio.to(device)
            labels = labels.to(device)
            outputs = model(audio)
            corr1, corr5=accuracy(outputs, labels, topk=(1,5))
            #Cumulative values
            top1+=corr1
            top5+=corr5
            counter+=1
            #Subset Values
            sub_top1+=corr1
            sub_top5+=corr5
            sub_counter+=1
        print("Subset Val:\tTop-1 accuracy: %.5f\tTop-5 accuracy: %.5f"%(sub_top1/sub_counter, sub_top5/sub_counter))
    print("Cumulative Val:\nTop-1 accuracy: %.5f\nTop-5 accuracy: %.5f"%(top1/counter, top5/counter))
    return top1/counter, top5/counter

def ppdf(df_F):
    df_F['Label']=df_F['Path'].str.split("/", n=1, expand=True)[0].str.replace("id","")
    df_F['Label']=df_F['Label'].astype(dtype=float)
    # print(df_F.head(20))
    df_F['Path']="wav/"+df_F['Path']
    return df_F



if __name__=="__main__":
    parser=argparse.ArgumentParser(
        description="Train and evaluate VGGVox on complete voxceleb1 for identification")
    parser.add_argument("--dir","-d",help="Directory with wav and csv files", default="./Data/")
    args=parser.parse_args()
    DATA_DIR=args.dir
    df_meta=pd.read_csv(LOCAL_DATA_DIR+"vox1_meta.csv",sep="\t")
    df_F=pd.read_csv(LOCAL_DATA_DIR+"iden_split.txt", sep=" ", names=["Set","Path"] )
    val_F=pd.read_pickle(LOCAL_DATA_DIR+"val.pkl")
    df_F=ppdf(df_F)
    val_F=ppdf(val_F)

    Datasets={
        "train":AudioDataset(df_F[df_F['Set']==1], DATA_DIR),
        "val":[AudioDataset(val_F[val_F['lengths']==i], DATA_DIR, is_train=False) for i in range(300,1100,100)],
        "test":AudioDataset(df_F[df_F['Set']==3], DATA_DIR, is_train=False)}
    batch_sizes={
            "train":B_SIZE,
            "val":1,
            "test":1}
    Dataloaders={}
    Dataloaders['train']=DataLoader(Datasets['train'], batch_size=batch_sizes['train'], shuffle=True, num_workers=4)
    Dataloaders['val']=[DataLoader(i, batch_size=batch_sizes['train'], shuffle=False, num_workers=2) for i in Datasets['val']]
    Dataloaders['test']=[DataLoader(Datasets['test'], batch_size=batch_sizes['test'], shuffle=False)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model=AudioNet(1251)
    model.to(device)
    loss_func=nn.CrossEntropyLoss()
    optimizer=SGD(model.parameters(), lr=0.01, momentum=0.99, weight_decay=5e-4)
    scheduler=lr_scheduler.StepLR(optimizer, step_size=5, gamma=1/1.17)
    #Save models after accuracy crosses 75
    best_acc=75
    update_grad=1
    best_epoch=0
    print("Start Training")
    for epoch in range(N_EPOCHS):
        model.train()
        running_loss=0.0
        corr1=0
        corr5=0
        top1=0
        top5=0
        random_subset=None
        loop=tqdm(Dataloaders['train'])
        loop.set_description(f'Epoch [{epoch+1}/{N_EPOCHS}]')
        for counter, (audio, labels) in enumerate(loop, start=1):
            optimizer.zero_grad()
            audio = audio.to(device)
            labels = labels.to(device)
            augmented = audio.clone()
            for _ in range(5):
                start = random.randint(0,300)
                length = random.randint(0, min(300 - start, 20))
                augmented[..., start:start+length]  = 0
            for _ in range(10):
                start = random.randint(0,512)
                length = random.randint(0, min(512 - start, 20))
                augmented[..., start:start+length, :]  = 0
            
            
            
            
            if counter==32:
                random_subset=audio
            outputs = model(audio)
            aug_out = model(augmented)
            loss = loss_func(outputs, labels) + loss_func(aug_out, labels)
            running_loss+=loss
            corr1, corr5=accuracy(outputs, labels, topk=(1,5))
            top1+=corr1
            top5+=corr5
            if(counter%update_grad==0):
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=(running_loss.item()/(counter)), top1_acc=top1/(counter), top5_acc=top5/counter)

        model(random_subset)
        model.eval()
        with torch.no_grad():
            acc1, _=test(model, Dataloaders['val'])
            if acc1>best_acc:
                best_acc=acc1
                best_model=model.state_dict()
                best_epoch=epoch
                torch.save(best_model, os.path.join(MODEL_DIR,"VGGMVAL_BEST_%d_%.2f.pth"%(best_epoch, best_acc)))
        scheduler.step()


    print('Finished Training..')
    PATH = os.path.join(MODEL_DIR,"VGGM_F.pth")
    torch.save(model.state_dict(), PATH)
    model.eval()
    acc1=test(model, Dataloaders['test'])

