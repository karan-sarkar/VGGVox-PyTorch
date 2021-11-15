import os
import sys
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import argparse
import torchmetrics

import torch
from easyfsl.data_tools import EasySet, TaskSampler
from easyfsl.utils import plot_images, sliding_average
from torch import nn, optim
from torch.optim import lr_scheduler, SGD, Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm
from audiodataset import AudioDataset
from audionet import AudioNet
# from relation_net_util import compute_backbone_output_shape
# if 'easyfsl.utils.compute_backbone_output_shape' in sys.modules:
#     del sys.modules['easyfsl.utils']
sys.modules['easyfsl.utils'].compute_backbone_output_shape = __import__('relation_net_util').compute_backbone_output_shape
from easyfsl.methods import RelationNetworks

LR=0.01
B_SIZE=100
N_EPOCHS=150
N_CLASSES=1251
LOCAL_DATA_DIR=os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR=os.path.join(os.path.dirname(__file__), "models")
N_WAY = 5 # Number of classes in a task
N_SHOT = 5 # Number of images per class in the support set
N_QUERY = 10 # Number of images per class in the query set
N_TRAINING_TASKS = 100
N_VALIDATION_TASKS = 100
N_EVALUATION_TASKS = 100

accuracy = torchmetrics.Accuracy()

# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append((correct_k.mul(100.0 / batch_size)).item())
#         return res

def evaluate_on_one_task(
    model,
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
) -> Tuple[int, int]:
    """
    Returns the number of correct predictions of query labels, and the total number of predictions.
    """
    output = model(support_images, support_labels, query_images).detach().data
    return (
        torch.max(
            output,
            1,
        )[1]
        == query_labels
    ).sum().item(), len(query_labels)


def evaluate(model, data_loader: DataLoader):
    # We'll count everything and compute the ratio at the end
    total_predictions = 0
    correct_predictions = 0

    # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
    # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
    model.eval()
    accuracy.reset()
    with torch.no_grad():
        loop = tqdm(enumerate(data_loader), total=len(data_loader))
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
        ) in loop:

            output = model(support_images, support_labels, query_images)
            episode_acc = accuracy(output, query_labels)

            loop.set_postfix(acc=episode_acc)

    return accuracy.compute()
    
def ppdf(df_F):
    df_F['Label']=df_F['Path'].str.split("/", n=1, expand=True)[0].str.replace("id","")
    df_F['Label']=df_F['Label'].astype(dtype=float)
    # print(df_F.head(20))
    df_F['Path']="wav/"+df_F['Path']
    return df_F

def train(model, Dataloaders: Dict, device):
    loss_func=nn.CrossEntropyLoss()
    optimizer=SGD(model.parameters(), lr=LR, momentum=0.99, weight_decay=5e-4)
    scheduler=lr_scheduler.StepLR(optimizer, step_size=5, gamma=1/1.17)
    #Save models after accuracy crosses 75
    best_acc=75
    update_grad=1
    best_epoch=0
    print("Start Training")
    for epoch in range(N_EPOCHS):
        model.train()
        running_loss=0.0
        # random_subset=None
        loop=tqdm(Dataloaders['train'])
        loop.set_description(f'Epoch [{epoch+1}/{N_EPOCHS}]')
        accuracy.reset()
        for episode_index, (
            support_images,
            support_labels,
            query_images,
            query_labels,
            class_ids,
        ) in enumerate(loop, start=1):
            optimizer.zero_grad()
            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)
            
            outputs = model(support_images, support_labels, query_images)
            
            episode_acc = accuracy(outputs, query_labels)
            
            loss = loss_func(outputs, query_labels)
            running_loss+=loss
            
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=running_loss.item(), episode_acc=episode_acc, avg_acc=accuracy.compute())
            
        model.eval()
        with torch.no_grad():
            val_acc = evaluate(model, Dataloaders['val'])
            print('Validation accuracy: %.2f'%val_acc)
            if val_acc>best_acc:
                best_acc=val_acc
                best_model=model.state_dict()
                best_epoch=epoch
                torch.save(best_model, os.path.join(MODEL_DIR,"FSL_BEST_%d_%.2f.pth"%(best_epoch, best_acc)))
        scheduler.step()
        
        
def dataloaders(data_dir):
    df_meta=pd.read_csv(os.path.join(LOCAL_DATA_DIR, "vox1_meta.csv"),sep="\t")
    df_F=pd.read_csv(os.path.join(LOCAL_DATA_DIR, "iden_split.txt"), sep=" ", names=["Set","Path"] )
    val_F=pd.read_pickle(os.path.join(LOCAL_DATA_DIR, "val.pkl"))
    df_F=ppdf(df_F)
    val_F=ppdf(val_F)
    wav_folders = os.listdir(os.path.join(data_dir, 'wav'))
    wav_folders = [float(f.split("/")[0].replace("id","")) for f in wav_folders if os.path.isdir(os.path.join(data_dir, 'wav', f))]
    df_F = df_F[df_F.Label.map(lambda l: l in wav_folders)]
    val_F = val_F[val_F.Label.map(lambda l: l in wav_folders)]
    Datasets={
        "train":AudioDataset(df_F[df_F['Set']==1], data_dir),
        "val":[AudioDataset(val_F[val_F['lengths']==i], data_dir, is_train=False) for i in range(300,1100,100)],
        "test":AudioDataset(df_F[df_F['Set']==3], data_dir, is_train=False)
    }
    samplers ={
        "train":TaskSampler(Datasets['train'], n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_TRAINING_TASKS),
        "val":[TaskSampler(i, n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_VALIDATION_TASKS) for i in Datasets['val']],
        "test":TaskSampler(Datasets['test'], n_way=N_WAY, n_shot=N_SHOT, n_query=N_QUERY, n_tasks=N_EVALUATION_TASKS),
    }
    
    Dataloaders={}
    Dataloaders['train']=DataLoader(Datasets['train'],num_workers=4, batch_sampler=samplers['train'], collate_fn=samplers['train'].episodic_collate_fn)
    Dataloaders['val']=[DataLoader(i, num_workers=2, batch_sampler=j, collate_fn=j.episodic_collate_fn) for i, j in zip(Datasets['val'], samplers['val'])]
    Dataloaders['test']=[DataLoader(Datasets['test'], batch_sampler=samplers['test'], collate_fn=samplers['test'].episodic_collate_fn)]
    return Dataloaders

if __name__=="__main__":
    parser=argparse.ArgumentParser(
        description="Train and evaluate VGGVox on complete voxceleb1 for identification")
    parser.add_argument("--dir","-d",help="Directory with wav and csv files", default="./Data/")
    args=parser.parse_args()
    
    Dataloaders = dataloaders(args.dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    backbone=AudioNet(N_WAY)
    model = RelationNetworks(backbone=backbone)
    model.to(device)
    
    train(model, Dataloaders, device)
        


    print('Finished Training..')
    PATH = os.path.join(MODEL_DIR,"VGGM_F.pth")
    torch.save(model.state_dict(), PATH)
    model.eval()
    acc1=evaluate(model, Dataloaders['test'])
