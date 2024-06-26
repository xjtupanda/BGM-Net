# BGM-Net
PyTorch implementation for the paper "Exploiting Instance-level Relationships in Weakly Supervised Text-to-Video Retrieval", accepted by ACM TOMM 2024 [[Link]](https://dl.acm.org/doi/abs/10.1145/3663571).
![model_overview](./assets/arch.png)


## Environments

OS: Ubuntu 20.04.4 LTS 

Python: 3.8

Pytorch: 1.13.1

CUDA: 11.6, cudnn: 8.4.0

GPU: NVIDIA GeForce RTX 3090 Ti

## Getting started
1. Clone this repository

```shell
$ git clone git@github.com:xjtupanda/BGM-Net.git
$ cd BGM-Net
```
2. Prepare environment

```shell
$ conda create -n env_name python=3.8
$ conda activate env_name
$ pip install -r requirements.txt
```

3. Download features

The features provided by [ms-sl](https://github.com/HuiGuanLab/ms-sl) can be downloaded from [Baidu pan](https://pan.baidu.com/s/1UNu67hXCbA6ZRnFVPVyJOA?pwd=8bh4).
Refer to [here](https://github.com/HuiGuanLab/ms-sl/tree/main/dataset) for detailed description of the datasets.
Extract the feature file to the `ROOTPATH` specified by yourself. This directory is used for storing datasets, logs, checkpoints and so on:
```shell
tar -xvf {DATASET_NAME}.tar -C ROOTPATH
```
4. Training and Inference

Simply run scripts `do_tvr.sh`, `do_activitynet.sh`, `do_charades.sh` for training and evaluation on TVR, Activitynet and Charades-STA, respectively. For example:
```shell
$ bash do_tvr.sh
```
for training and evaluation on TVR dataset.
Set `device_ids` in the script to assign which GPU to use.
`bsz` and `lr` are batch-size and learning rate, respectively.
Check `method/config.py` for all more description of parameters.
The best checkpoint and the log should be saved in `ROOTPATH/DATASET_NAME/results`.


***
To reproduce the results using pre-trained checkpoints, please download [here](https://pan.baidu.com/s/1JJhkXX2Zi9e9KkThxYK_AA?pwd=b6di).

Extract and put the model file in the right place, set `MODEL_DIR` in `do_test.sh` and run:
```
$ bash do_test.sh
```