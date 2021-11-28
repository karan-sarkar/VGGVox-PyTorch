import random
import argparse
import os
import sys
from time import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from easyfsl.data_tools import EasySet, TaskSampler
from easyfsl.utils import plot_images, sliding_average
from torch import functional, nn, optim
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.models import resnet18, resnet34
from tqdm.autonotebook import tqdm

from audiodataset import AudioDataset
from audionet import AudioNet

sys.modules['easyfsl.utils'].compute_backbone_output_shape = __import__('relation_net_util').compute_backbone_output_shape
import pytorch_lightning as pl
from easyfsl.methods import (AbstractMetaLearner, PrototypicalNetworks,
                             RelationNetworks)
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from augmentations.mixup import mixup_batch
from augmentations.spec_augment import SpecAugment


class FSLModel(pl.LightningModule):
    
    def __init__(self, model: AbstractMetaLearner, 
                lr = 0.001,
                momentum=0.99,
                weight_decay=5e-4,
                lr_step_size=5,
                lr_gamma=1/1.17,
                apply_mixup: bool = False, 
                mixup_alpha: float = 0.3, 
                transforms = [],
                num_way = 20,
                num_shot = 1,
                num_query = 3,
                num_train_tasks = 100,
                num_eval_tasks = 100,
                num_val_tasks = 100,
                augmentation='none',
                backbone_arch='resnet18',
                fsl_arch='relation-net',
                **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.model = model
        self.accuracy = Accuracy()
        self.apply_mixup = apply_mixup
        self.mixup_alpha = mixup_alpha
        self.transforms = nn.Sequential(*transforms)
    
    def configure_optimizers(self):
        optimizer=SGD(self.model.parameters(), lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        scheduler=lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_step_size, gamma=self.hparams.lr_gamma)
        return [optimizer], [scheduler]
        
    def training_step(self, batch, batch_idx):
        support_images, support_labels, query_images, query_labels, class_ids = batch
        self.model.process_support_set(support_images, support_labels)
        QUERY_BATCH_SIZE = query_images.shape[0]
        query_labels_hot = F.one_hot(query_labels)
        # Create interpolated data for mixup
        if self.apply_mixup:
            query_perm = torch.randperm(query_images.size()[0])
            
            query_images2 = query_images[query_perm]
            query_labels2 = query_labels[query_perm]
            
            query_labels2_hot = F.one_hot(query_labels2)
            
            query_images2, query_labels2_hot = mixup_batch((query_images, query_labels_hot), (query_images2, query_labels2_hot), alpha = self.mixup_alpha)
            # concatenate the original data with mixup data. This doubles the batch size
            query_images = torch.cat((query_images, query_images2), dim=0)
            query_labels_hot = torch.cat((query_labels_hot, query_labels2_hot), dim=0)
        if self.transforms is not None:
            query_images = self.transforms(query_images)
        outputs = self.model(query_images)
        
        self.accuracy(outputs[:QUERY_BATCH_SIZE], query_labels)
        self.log('train_acc', self.accuracy, prog_bar=True, batch_size=QUERY_BATCH_SIZE)
        
        loss = self.model.compute_loss(outputs, query_labels_hot.to(torch.float))
        self.log('train_loss', loss, batch_size=QUERY_BATCH_SIZE)
        return loss
    
    def evaluate(self, support_images, support_labels, query_images, query_labels):
        self.model.process_support_set(support_images, support_labels)
        return self.model(query_images)
        
    def validation_step(self, batch, batch_idx):
        support_images, support_labels, query_images, query_labels, class_ids = batch
        outputs = self.evaluate(support_images, support_labels, query_images, query_labels)
        self.accuracy(outputs, query_labels)
        self.log('val_acc', self.accuracy, prog_bar=True, batch_size=query_images.shape[0])
        return self.model.compute_loss(outputs, query_labels)
        
    def test_step(self, batch, batch_idx):
        support_images, support_labels, query_images, query_labels, class_ids = batch
        outputs = self.evaluate(support_images, support_labels, query_images, query_labels)
        self.accuracy(outputs, query_labels)
        self.log('test_acc', self.accuracy, prog_bar=True, batch_size=query_images.shape[0])
        

class ExpandChannels(nn.Module):
    def __init__(self, out_channels=3) -> None:
        """Expands a single channel input to the specified number of channels. Shares memory across all channels.

        Args:
            out_channels (int, optional): The number of output channels. Defaults to 3.
        """
        super().__init__()
        self.out_channels = out_channels
        
    def forward(self, x):
        shape = list(x.shape)
        shape[1] = self.out_channels
        return x.expand(*shape)

class MySampler(TaskSampler):
    def __init__(
        self, dataset, n_way: int=1, n_shot: int=1, n_query: int=1, n_tasks: int=1, batch_size = 1,drop_last=False
    ):
        super().__init__(dataset, n_way, n_shot, n_query, n_tasks)
        self.batch_size = batch_size
        self.drop_last = drop_last
        

class RelationNetworksMixup(RelationNetworks):
    
    # We need to override the compute loss method to accept one hot encoded tensors as well as normal tensors.
    def compute_loss(
        self, classification_scores: torch.Tensor, query_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Overrides the method compute_loss of AbstractMetaLearner because Relation Networks
        use the Mean Square Error (MSE) loss. MSE is a regression loss, so it requires the ground
        truth to be of the same shape as the predictions. In our case, this means that labels
        must be provided in a one hot fashion.

        Note that we need to enforce the number of classes by using the last computed prototypes,
        in case query_labels doesn't contain all possible labels.

        Args:
            classification_scores: predicted classification scores of shape (n_query, n_classes)
            query_labels: one hot ground truth labels of shape (n_query, n_classes) or (n_query,)

        Returns:
            MSE loss between the prediction and the ground truth
        """
        if query_labels.shape != classification_scores.shape:
            # The query_labels are not in one hot shape. Transform them
            query_labels = nn.functional.one_hot(
                query_labels, num_classes=self.prototypes.shape[0]
            ).float()
        return self.loss_function(
            classification_scores, query_labels,
        )
    
class Experiment(object):
    

    def __init__(self, 
                batch_size: int = 32,
                num_workers = 2,
                num_epochs = 10,
                num_way = 20,
                num_shot = 1,
                num_query = 3,
                lr = 0.001,
                num_train_tasks = 100,
                num_eval_tasks = 100,
                num_val_tasks = 100,
                device='cuda',
                augmentation='none',
                backbone_arch='resnet18',
                fsl_arch='relation-net',
                alpha=0.2,
                dir='./data',
                is_dev_run=False,
        ) -> None:
        super().__init__()
        self.LOCAL_DATA_DIR=os.path.join(os.path.dirname(__file__), "data")
        self.MODEL_DIR=os.path.join(os.path.dirname(__file__), "models")
        
        self.LR=lr
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
        self.augmentation = augmentation
        self.backbone_arch = backbone_arch
        self.fsl_arch = fsl_arch
        self.alpha = alpha
        self.dir = dir
        self.is_dev_run = is_dev_run
        
        self.model = self.get_model(pretrained=False, backbone_arch=self.backbone_arch, fsl_arch = self.fsl_arch, transforms=self._get_transforms())
        self.Dataloaders = self.dataloaders(self.dir)

    def evaluate(self):
        
        trainer = pl.Trainer(
            gpus = -1 if str(self.device) != 'cpu' else 0,
            max_epochs = self.N_EPOCHS,
            fast_dev_run=self.is_dev_run,
            progress_bar_refresh_rate= 3
        )
        # trainer.logger._default_hp_metric = None
        trainer.test(model = self.model, dataloaders=self.Dataloaders['test'])

    def train(self):
        
        trainer = pl.Trainer(
            gpus = -1 if str(self.device) != 'cpu' else 0,
            max_epochs = self.N_EPOCHS,
            callbacks=[
                ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                LearningRateMonitor("epoch"),
            ],
            fast_dev_run=self.is_dev_run,
            replace_sampler_ddp=False,
            progress_bar_refresh_rate= 3
        )
        
        # trainer.logger._default_hp_metric = None
        trainer.fit(model=self.model, train_dataloaders=self.Dataloaders['train'], val_dataloaders=self.Dataloaders['val'])
        print(f'The best model is stored at {trainer.checkpoint_callback.best_model_path}')
    
    def _get_transforms(self):
        ts =[]
        if self.augmentation == 'none':
            pass
        elif self.augmentation == 'spec':
            ts.append(SpecAugment(W=50, F=30, T=40, freq_masks=2, time_masks=2, freq_zero=False, time_zero=False, to_mel=False),)
        elif self.augmentation == 'mixup':
            pass
        print('ts', ts)
        return ts
    
    def ppdf(self, df_F):
        df_F['Label']=df_F['Path'].str.split("/", n=1, expand=True)[0].str.replace("id","")
        df_F['Label']=df_F['Label'].astype(dtype=float)
        # print(df_F.head(20))
        df_F['Path']="wav/"+df_F['Path']
        return df_F
    
    def dataloaders(self, data_dir):
        # df_meta=pd.read_csv(os.path.join(self.LOCAL_DATA_DIR, "vox1_meta.csv"),sep="\t")
        # use a constant seed for reproducible splits
        SPLIT_SEED = 42
        
        df_F=pd.read_csv(os.path.join(self.LOCAL_DATA_DIR, "iden_split.txt"), sep=" ", names=["Set","Path"] )
        df_F=self.ppdf(df_F)
        # filter out unavailable classes . This occurs when we are working with a test subset
        wav_folders = os.listdir(os.path.join(data_dir, 'wav'))
        wav_folders = [float(f.split("/")[0].replace("id","")) for f in wav_folders if os.path.isdir(os.path.join(data_dir, 'wav', f))]
        df_F = df_F[df_F.Label.map(lambda l: l in wav_folders)]
        
        zlist = pd.Series(pd.unique(df_F.Label)).sample(frac = 1, random_state=SPLIT_SEED).reset_index(drop=True)
        
        ds_len = len(zlist)
        
        split_lens = [int(ds_len * 0.7), int(ds_len * 0.2)]
        
        train_classes = zlist.iloc[:split_lens[0]]
        val_classes = zlist.iloc[split_lens[0]:split_lens[0]+split_lens[1]]
        test_classes = zlist.iloc[split_lens[0]+split_lens[1]:]
        
        train_F = df_F[df_F.Label.isin(train_classes)]
        val_F = df_F[df_F.Label.isin(val_classes)]
        test_F = df_F[df_F.Label.isin(test_classes)]
        
        Datasets={
            "train":AudioDataset(train_F, data_dir),
            "val":[AudioDataset(val_F, data_dir, is_train=False)],
            "test":AudioDataset(test_F, data_dir, is_train=False)
        }
        samplers ={
            "train":MySampler(Datasets['train'], n_way=self.N_WAY, n_shot=self.N_SHOT, n_query=self.N_QUERY, n_tasks=self.N_TRAINING_TASKS),
            "val":[MySampler(i, n_way=self.N_WAY, n_shot=self.N_SHOT, n_query=self.N_QUERY, n_tasks=self.N_VALIDATION_TASKS) for i in Datasets['val']],
            "test":MySampler(Datasets['test'], n_way=self.N_WAY, n_shot=self.N_SHOT, n_query=self.N_QUERY, n_tasks=self.N_EVALUATION_TASKS),
        }
        # print(samplers['test'].items_per_label)
 
        Dataloaders={}
        Dataloaders['train']=DataLoader(Datasets['train'],num_workers=self.NUM_WORKERS, batch_sampler=samplers['train'], collate_fn=samplers['train'].episodic_collate_fn)
        Dataloaders['val']=[DataLoader(i, num_workers=self.NUM_WORKERS, batch_sampler = j, shuffle=False, collate_fn=j.episodic_collate_fn) for i, j in zip(Datasets['val'], samplers['val'])]
        Dataloaders['test']=[DataLoader(Datasets['test'], num_workers=self.NUM_WORKERS, batch_sampler=samplers['test'], shuffle=False, collate_fn=samplers['test'].episodic_collate_fn)]
        return Dataloaders
    
    def get_model(
        self,
        fsl_arch='relation-net',
        backbone_arch='resnet18',
        pretrained: bool =False,
        num_ways = 10,
        transforms = None
    ) -> AbstractMetaLearner:
        if backbone_arch == 'resnet18': 
            backbone = resnet18(pretrained)
            backbone = nn.Sequential(
                # Audio dataset has only one channel, expand to 3
                ExpandChannels(3),
                # Remove the avgpool and fc layers
                *list(backbone.children())[:-2],
            )
        elif backbone_arch == 'resnet34':
            backbone = resnet34(pretrained)
            backbone = nn.Sequential(
                # Audio dataset has only one channel, expand to 3
                ExpandChannels(3),
                # Remove the avgpool and fc layers
                *list(backbone.children())[:-2],
            ) 
        elif backbone_arch == 'audionet':
            backbone = AudioNet(512, mode='fe')
        print(backbone)
        if fsl_arch == 'relation-net':
            
            model = RelationNetworksMixup(backbone)
            
        elif fsl_arch == 'proto-net':
            # print('Proto-Net')
            backbone = nn.Sequential(
                *list(backbone.children())[:-2],
                nn.Flatten(),
            )
            model = PrototypicalNetworks(backbone)
        else:
            raise ValueError('Invalid FSL arch type')
        model = FSLModel(model=model, apply_mixup=self.augmentation=='mixup', mixup_alpha=self.alpha, transforms=transforms,
                num_way=self.N_WAY,
                num_shot=self.N_SHOT,
                num_query=self.N_QUERY,
                num_train_tasks=self.N_TRAINING_TASKS,
                num_eval_tasks=self.N_EVALUATION_TASKS,
                num_val_tasks=self.N_VALIDATION_TASKS,
                augmentation=self.augmentation,
                backbone_arch=self.backbone_arch,
                fsl_arch=self.fsl_arch,
        )
        return model

if __name__=="__main__":
    
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!


    parser=argparse.ArgumentParser(
        description="Train and evaluate VGGVox on complete voxceleb1 for identification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dir","-d",help="Directory with wav and csv files", default="./data/")
    parser.add_argument("--batch-size","-bs",help="Batch Size", default=1, type=int)
    parser.add_argument("--learning-rate","-lr",help="Learning rate", default=0.001, type=float)
    parser.add_argument("--num_workers","-nw",help="Number of workers to use in the Dataloader", default=2, type=int)
    parser.add_argument("--fsl-arch",help="The Few-shot architecture to use", default="relation-net", choices=['relation-net', 'proto-net'])
    parser.add_argument("--backbone-arch",help="The Backbone architecture to use", default="resnet18", choices=['resnet18', 'resnet34', 'audionet'])
    parser.add_argument("--num_way","-w",help="Number of ways to classify the model", default=5, type=int)
    parser.add_argument("--num_shot","-s",help="Number of support images per class", default=3, type=int)
    parser.add_argument("--num_query","-q",help="Number of query images for each class in one task", default=2, type=int)
    parser.add_argument("--num_train_tasks","-tt",help="Number of tasks to sample for training", default=20, type=int)
    parser.add_argument("--num_val_tasks","-vt",help="Number of tasks to sample for validation", default=5, type=int)
    parser.add_argument("--num_eval_tasks","-et",help="Number of tasks to sample for evaluation/test", default=5, type=int)
    parser.add_argument("--num_epochs","-e",help="Number of epochs", default=10, type=int)
    parser.add_argument("--augmentation", "-a", help="The data augmentation to use", default="none", choices=['spec', 'mixup', 'none'])
    parser.add_argument("--is_dev_run", help="Use this during development to train only for one batch", action='store_true')
    
    args=parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    experiment = Experiment(
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        device=device,
        num_way=args.num_way,
        num_shot=args.num_shot,
        num_query=args.num_query,
        num_train_tasks=args.num_train_tasks,
        num_val_tasks=args.num_val_tasks,
        num_eval_tasks=args.num_eval_tasks,
        num_epochs=args.num_epochs,
        augmentation=args.augmentation,
        backbone_arch=args.backbone_arch,
        fsl_arch=args.fsl_arch,
        dir=args.dir,
        is_dev_run=args.is_dev_run,
    )

    # Dataloaders = experiment.dataloaders(args.dir)
    
    
    # model = get_model(pretrained=False, backbone_arch=args.backbone_arch, fsl_arch = args.fsl_arch)
    # model = model.to(device)
    
    experiment.train()

    print('Finished Training..')
    # PATH = os.path.join(experiment.MODEL_DIR,"VGGM_F.pth")
    # torch.save(model.state_dict(), PATH)
    # model.eval()
    print('Running test...')
    experiment.evaluate()
