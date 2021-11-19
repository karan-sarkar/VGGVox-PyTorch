import os
import sys
from typing import Dict, List, Tuple
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
from tqdm.autonotebook import tqdm
from audiodataset import AudioDataset
from audionet import AudioNet
# from relation_net_util import compute_backbone_output_shape
# if 'easyfsl.utils.compute_backbone_output_shape' in sys.modules:
#     del sys.modules['easyfsl.utils']
sys.modules['easyfsl.utils'].compute_backbone_output_shape = __import__('relation_net_util').compute_backbone_output_shape
from easyfsl.methods import RelationNetworks, AbstractMetaLearner
from augmentations.spec_augment import SpecAugment


class Experiment(object):
    

    def __init__(self, 
                batch_size: int = 128,
                num_workers = 2,
                num_epochs = 10000,
                num_way = 10,
                num_shot = 1,
                num_query = 3,
                num_train_tasks = 100,
                num_eval_tasks = 100,
                num_val_tasks = 100,
                device='cuda',
        ) -> None:
        super().__init__()
        self.LOCAL_DATA_DIR=os.path.join(os.path.dirname(__file__), "data")
        self.MODEL_DIR=os.path.join(os.path.dirname(__file__), "models")
        
        self.LR=0.001
        self.B_SIZE=batch_size
        self.N_EPOCHS=num_epochs
        
        self.N_WAY = num_way                # Number of classes in a task
        self.N_SHOT = num_shot              # Number of images per class in the support set
        self.N_QUERY = num_query            # Number of images per class in the query set
        self.N_TRAINING_TASKS = num_train_tasks
        self.N_VALIDATION_TASKS = num_val_tasks
        self.N_EVALUATION_TASKS = num_eval_tasks
        self.NUM_WORKERS = num_workers
        
        self.device = device
        

    def evaluate(self, model: AbstractMetaLearner, data_loaders: List[DataLoader]):
        # We'll count everything and compute the ratio at the end
        total_predictions = 0
        correct_predictions = 0

        # eval mode affects the behaviour of some layers (such as batch normalization or dropout)
        # no_grad() tells torch not to keep in memory the whole computational graph (it's more lightweight this way)
        model.eval()
        accuracy = torchmetrics.Accuracy().to(self.device)
        accuracy.reset()
        for data_loader in data_loaders:
            with torch.no_grad():
                loop = tqdm(enumerate(data_loader), total=len(data_loader))
                for episode_index, (
                    support_images,
                    support_labels,
                    query_images,
                    query_labels,
                    class_ids,
                ) in loop:
                    #print(support_images.shape, query_images.shape)
                    #print(support_labels.shape, query_labels.shape)
                    support_images = support_images.to(self.device)
                    query_images = query_images.to(self.device)
                    support_labels = support_labels.to(self.device)
                    query_labels = query_labels.to(self.device)

                    model.process_support_set(support_images, support_labels)
                    output = model(query_images)
                    episode_acc = accuracy(output, query_labels)

                    loop.set_postfix(acc=episode_acc.item())

        return accuracy.compute()
        
    def ppdf(self, df_F):
        df_F['Label']=df_F['Path'].str.split("/", n=1, expand=True)[0].str.replace("id","")
        df_F['Label']=df_F['Label'].astype(dtype=float)
        # print(df_F.head(20))
        df_F['Path']="wav/"+df_F['Path']
        return df_F

    def train(self, model: AbstractMetaLearner, Dataloaders: Dict):
        accuracy = torchmetrics.Accuracy().to(self.device)
        optimizer=SGD(model.parameters(), lr=self.LR, momentum=0.99, weight_decay=5e-4)
        scheduler=lr_scheduler.StepLR(optimizer, step_size=5, gamma=1/1.17)
        #Save models after accuracy crosses 75
        best_acc=75
        update_grad=1
        best_epoch=0
        print("Start Training")
        for epoch in range(self.N_EPOCHS):
            model.train()
            running_loss=[]
            # random_subset=None
            loop=tqdm(Dataloaders['train'])
            loop.set_description(f'Epoch [{epoch+1}/{self.N_EPOCHS}]')
            accuracy.reset()
            for episode_index, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                class_ids,
            ) in enumerate(loop, start=1):
                optimizer.zero_grad()
                support_images = support_images.to(self.device)
                support_labels = support_labels.to(self.device)
                query_images = query_images.to(self.device)
                query_labels = query_labels.to(self.device)
                
                model.process_support_set(support_images, support_labels)
                
                outputs = model(query_images)
                
                episode_acc = accuracy(outputs, query_labels)
                
                loss = model.compute_loss(outputs, query_labels)
                running_loss.append(loss.item())
                
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=sum(running_loss)/len(running_loss), episode_acc=episode_acc.item(), avg_acc=accuracy.compute().item())
                
            model.eval()
            with torch.no_grad():
                val_acc = self.evaluate(model, Dataloaders['test'])
                print('Validation accuracy: %.2f'%val_acc)
                if val_acc>best_acc:
                    best_acc=val_acc
                    best_model=model.state_dict()
                    best_epoch=epoch
                    torch.save(best_model, os.path.join(self.MODEL_DIR,"FSL_BEST_%d_%.2f.pth"%(best_epoch, best_acc)))
            scheduler.step()
            
            
    def dataloaders(self, data_dir):
        df_meta=pd.read_csv(os.path.join(self.LOCAL_DATA_DIR, "vox1_meta.csv"),sep="\t")
        df_F=pd.read_csv(os.path.join(self.LOCAL_DATA_DIR, "iden_split.txt"), sep=" ", names=["Set","Path"] )
        val_F=pd.read_pickle(os.path.join(self.LOCAL_DATA_DIR, "val.pkl"))
        df_F=self.ppdf(df_F)
        val_F=self.ppdf(val_F)
        wav_folders = os.listdir(os.path.join(data_dir, 'wav'))
        wav_folders = [float(f.split("/")[0].replace("id","")) for f in wav_folders if os.path.isdir(os.path.join(data_dir, 'wav', f))]
        df_F = df_F[df_F.Label.map(lambda l: l in wav_folders)]
        val_F = val_F[val_F.Label.map(lambda l: l in wav_folders)]
        Datasets={
            "train":AudioDataset(df_F[df_F['Set']==1], data_dir, data_transforms=[
                SpecAugment(W=50, F=30, T=40, freq_masks=2, time_masks=2, freq_zero=False, time_zero=False, to_mel=True),
            ]),
            "val":[AudioDataset(val_F[val_F['lengths']==i], data_dir, is_train=False) for i in range(300,1100,100)],
            "test":AudioDataset(df_F[df_F['Set']==3], data_dir, is_train=False)
        }
        samplers ={
            "train":TaskSampler(Datasets['train'], n_way=self.N_WAY, n_shot=self.N_SHOT, n_query=self.N_QUERY, n_tasks=self.N_TRAINING_TASKS),
            "val":[TaskSampler(i, n_way=self.N_WAY, n_shot=self.N_SHOT, n_query=self.N_QUERY, n_tasks=self.N_VALIDATION_TASKS) for i in Datasets['val']],
            "test":TaskSampler(Datasets['test'], n_way=self.N_WAY, n_shot=self.N_SHOT, n_query=self.N_QUERY, n_tasks=self.N_EVALUATION_TASKS),
        }
        print(samplers['test'].items_per_label)
 
        Dataloaders={}
        Dataloaders['train']=DataLoader(Datasets['train'],num_workers=self.NUM_WORKERS, batch_sampler=samplers['train'], collate_fn=samplers['train'].episodic_collate_fn)
        Dataloaders['val']=[DataLoader(i, num_workers=self.NUM_WORKERS, batch_sampler = j, shuffle=False, collate_fn=j.episodic_collate_fn) for i, j in zip(Datasets['val'], samplers['val'])]
        Dataloaders['test']=[DataLoader(Datasets['test'], num_workers=self.NUM_WORKERS, batch_sampler=samplers['test'], shuffle=False, collate_fn=samplers['test'].episodic_collate_fn)]
        return Dataloaders

def get_model(
        fsl_arch='relation-net',
        backbone_arch='resnet18',
        pretrained: bool =False,
        num_ways = 5
    ):
    if backbone_arch == 'resnet18':
        backbone = resnet18(pretrained)
        # replace the first conv1 layer and remove the avgpool and fc layers
        backbone = nn.Sequential(
            # Audio dataset has only one channel
            nn.Conv2d(1, backbone.inplanes, kernel_size=7, stride=2, padding=3,bias=False),
            *list(backbone.children())[1:-2],
        )
        
    elif backbone_arch == 'audionet':
        backbone = AudioNet(num_ways, mode='fe')
    
    if fsl_arch == 'relation-net':
        # hack to get relationnet to work with audiodataset
        model = RelationNetworks(backbone)
    else:
        raise ValueError('Invalid FSL arch type')
    return model

if __name__=="__main__":
    parser=argparse.ArgumentParser(
        description="Train and evaluate VGGVox on complete voxceleb1 for identification")
    parser.add_argument("--dir","-d",help="Directory with wav and csv files", default="./Data/")
    parser.add_argument("--batch-size","-bs",help="Batch Size", default=1, type=int)
    parser.add_argument("--num_workers","-nw",help="Number of workers to use in the Dataloader", default=2, type=int)
    parser.add_argument("--fsl-arch",help="The Few-shot architecture to use", default="relation-net", choices=['relation-net'])
    parser.add_argument("--backbone-arch",help="The Backbone architecture to use", default="resnet18", choices=['resnet18', 'audionet'])
    args=parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    experiment = Experiment(batch_size=args.batch_size, num_workers=args.num_workers, device=device)
    Dataloaders = experiment.dataloaders(args.dir)
    
    
    model = get_model(pretrained=False, backbone_arch='audionet')
    model = model.to(device)
    
    experiment.train(model, Dataloaders)

    print('Finished Training..')
    PATH = os.path.join(experiment.MODEL_DIR,"VGGM_F.pth")
    torch.save(model.state_dict(), PATH)
    model.eval()
    print('Running test...')
    acc1=experiment.evaluate(model, Dataloaders['test'])
