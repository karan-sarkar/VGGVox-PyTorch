
# Useful Papers
SpecAugment: A New Data Augmentation Method for Automatic Speech Recognition
https://arxiv.org/pdf/1904.08779.pdf 
https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html


Classical Approaches to Speaker Recognition
Speaker Recognition: A Tutorial
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=628714


Deep Speaker Recognition

Speaker Recognition Based on Deep Learning: An Overview
https://arxiv.org/pdf/2012.00931.pdf

VoxCeleb2: Deep Speaker Recognition
https://arxiv.org/pdf/1806.05622.pdf

Deep Speaker: an End-to-End Neural Speaker Embedding Syste
https://arxiv.org/pdf/1705.02304.pdf
http://research.baidu.com/Blog/index-view?id=98

Few-Shot Learning

Learning to Compare: Relation Network for Few-Shot Learni
https://openaccess.thecvf.com/content_cvpr_2018/papers/Sung_Learning_to_Compare_CVPR_2018_paper.pdf

Prototypical Networks for Few-shot Learning
https://arxiv.org/pdf/1703.05175.pdf


# VGGVox-PyTorch
Implementing VGGVox for VoxCeleb1 dataset in PyTorch.

## Train

```
pip install -r requirements.txt
python3 train.py --dir ./data/
```

###### Specify data dir with --dir

## Notes
- 81.79% Top-1 & 93.17 Top-5 Test-set accuracy, pretty satisfactory. Find details in [results.txt](results.txt).
- Training on the V100 takes 4 mins per epoch.

## Model
- Run `python3 vggm.py` for model architecture.
- Best model weights uploaded [VGGM300_BEST_140_81.99.pth](models/VGGM300_BEST_140_81.99.pth)

#### What i've done so far:
 - [x] **All the data preprocessed exactly as author's matlab code.** Checked and verified online on matlab
 - [x] **Random 3s cropped segments for training.**
 - [x] **Copy all hyperparameter**... LR, optimizer params, batch size from the author's net.
 - [x] **Stabilize PyTorch's BatchNorm and test variants.** Improved results by a small percentage.
 - [x] **Try onesided spectrogram input as mentioned on the author's github.**
 - [ ] ~~**Port the authors network from matlab and train.** The matlab model has 1300 outputs dimension, will test it later.~~
 - [ ] ~~**Copy weights from the matlab network and test.**~~

# References and Citations:

 - [VGGVox](https://github.com/a-nagrani/VGGVox)
 - linhdvu14's [vggvox-speaker-identification](https://github.com/linhdvu14/vggvox-speaker-identification)
 - jameslyons's [python_speech_features](https://github.com/jameslyons/python_speech_features)
 - Easy few shot learning [link](https://github.com/sicara/easy-few-shot-learning)
 
 ```bibtex
@InProceedings{Nagrani17,
  author       = "Nagrani, A. and Chung, J.~S. and Zisserman, A.",
  title        = "VoxCeleb: a large-scale speaker identification dataset",
  booktitle    = "INTERSPEECH",
  year         = "2017",
}


@InProceedings{Nagrani17,
  author       = "Chung, J.~S. and Nagrani, A. and Zisserman, A.",
  title        = "VoxCeleb2: Deep Speaker Recognition",
  booktitle    = "INTERSPEECH",
  year         = "2018",
}

@software{Bennequin_easyfsl,
author = {Bennequin, Etienne},
license = {MIT},
title = {{easyfsl}},
url = {https://github.com/sicara/easy-few-shot-learning}
}
```

