# Contrastive Self-supervised Sequential Recommendation with Robust Augmentation

Source code for the paper: [Contrastive Self-supervised Sequential Recommendation with Robust Augmentation](https://arxiv.org/pdf/2108.06479.pdf)

<img src="./img/framework.png" width="450">

<img src="./img/augmentation.png" width="250">

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

## Citation

Please cite our paper if it is helpful for your research.

```
@article{liu2021contrastive,
  title={Contrastive self-supervised sequential recommendation with robust augmentation},
  author={Liu, Zhiwei and Chen, Yongjun and Li, Jia and Yu, Philip S and McAuley, Julian and Xiong, Caiming},
  journal={arXiv preprint arXiv:2108.06479},
  year={2021}
}
```

## Acknowledge
 - Transformer and training pipeline are implemented based on [S3-Rec](https://github.com/RUCAIBox/CIKM2020-S3Rec). Thanks them for providing efficient implementation.

