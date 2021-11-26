import torch
from typing import List
import pandas as pd
from scipy.io import wavfile
import os
import numpy as np
import signal_utils as sig
from torchvision import transforms
from torch.utils.data import Subset, Dataset
from torchvision import transforms


# transformers=transforms.ToTensor()


class AudioDataset(Dataset):
    # renamed the target variable name to `labels` as per easyfsl conventions
    def __init__(
        self,
        csv_file,
        data_dir,
        croplen=48320,
        is_train=True,
        data_transforms: List = None,
    ):
        if isinstance(csv_file, str):
            csv_file = pd.read_csv(csv_file)
        assert isinstance(csv_file, pd.DataFrame), "Invalid csv path or dataframe"
        self.X = csv_file["Path"].values
        self.labels = (csv_file["Label"].values - 10001).astype(int)
        self.data_dir = data_dir
        self.is_train = is_train
        self.croplen = croplen
        if data_transforms is None:
            data_transforms = []
            
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                *data_transforms,
            ]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        sr, audio = wavfile.read(os.path.join(self.data_dir, self.X[idx]))
        start = np.random.randint(0, audio.shape[0] - self.croplen + 1)
        audio = audio[start : start + self.croplen]
        audio = sig.preprocess(audio).astype(np.float32)
        audio = np.expand_dims(audio, 2)
        return self.transforms(audio), label
