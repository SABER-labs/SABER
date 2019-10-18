![alt text](icons/character+fat+game+hero+inkcontober+movie+icon-1320183878106104615_24.png) SABER - Semi-Supervised Audio Baseline for Easy Reproduction
=====
A PyTorch implementation for the paper #TODO which provides easily reproducible baselines for automatic speech recognition using semi-supervised learning.

## Overview
SABER consists of the following components

* An Mixnet based modification of quartznet
* RAdam optimizer to offset warmup used by SpecAugment
* SpecAugment and SpecSparkle a cutout inspired variant as Data Augmentations whose parameters linearly increase in a curriculum based approach
* Aggregated Cross Entropy loss instead of CTC loss for easier training
* Unsupervised Data Augmentation as means for Semi-Supervised Learning

## Requirements

* ariar2c
* python3.x
* libraries in requirements.txt

## Download Dataset

Librispeech dataset using download scripts, change dir parameter as per your configuration
```
sh download_scripts/download_librispeech.sh
```

## Training
Modify `utils/config.py` as per your configuration and run
```
CUDA_VISIBLE_DEVICES="0,1,2" python3 train.py
```


References
==========

## Codebases

[DeepSpeech2](https://github.com/PaddlePaddle/DeepSpeech)

[NVIDIA Neural Modules: NeMo](https://github.com/NVIDIA/NeMo)

[RAdam](https://github.com/LiyuanLucasLiu/RAdam)

[LR-Finder](https://github.com/davidtvs/pytorch-lr-finder)

[Cyclical Learning Rate Scheduler With Decay in Pytorch](https://github.com/bluesky314/Cyclical_LR_Scheduler_With_Decay_Pytorch)

[MixNet](https://github.com/romulus0914/MixNet-Pytorch)

## Papers

[Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/abs/1512.02595)

[Jasper: An End-to-End Convolutional Neural Acoustic Model](https://arxiv.org/abs/1904.03288)

[SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/pdf/1904.08779.pdf)

[Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/abs/1708.04552)

[On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265)

[Aggregation Cross-Entropy for Sequence Recognition](https://arxiv.org/abs/1904.08364)

[MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249)

[MixConv: Mixed Depthwise Convolutional Kernels](https://arxiv.org/abs/1907.09595)

[Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848)

[Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/pdf/1506.01186.pdf)

[Cycle-consistency training for end-to-end speech recognition](https://arxiv.org/abs/1811.01690)

[RandAugment: Practical data augmentation with no separate search](https://arxiv.org/abs/1909.13719)