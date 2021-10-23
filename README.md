# Introduction
Contrastive Self-supervised Sequential Recommendation with Robust Augmentation

Source code for paper: [Contrastive Self-supervised Sequential Recommendation with Robust Augmentation](https://arxiv.org/pdf/2108.06479.pdf)

Model architecture:

<img src="./img/framework.png" width="800">

Data Augmentations:

<img src="./img/augmentation.png" width="600">

# Reference

Please cite our paper if you use this code.

```
@article{liu2021contrastive,
  title={Contrastive self-supervised sequential recommendation with robust augmentation},
  author={Liu, Zhiwei and Chen, Yongjun and Li, Jia and Yu, Philip S and McAuley, Julian and Xiong, Caiming},
  journal={arXiv preprint arXiv:2108.06479},
  year={2021}
}
```

# Implementation
## Requirements

Python >= 3.7  
Pytorch >= 1.2.0  
tqdm == 4.26.0

## Datasets

Four prepared datasets are included in `data` folder.

## Train Model

To train our model on `Sports_and_Outdoors` dataset, change to the `src` folder and run following command: 

```
python main.py --data_name Sports_and_Outdoors
```

# Acknowledgement
 - Transformer and training pipeline are implemented based on [S3-Rec](https://github.com/RUCAIBox/CIKM2020-S3Rec). Thanks them for providing efficient implementation.

