# BGM-Net
PyTorch implementation for the paper "Bipartite Graph Matching Network for Partially Relevant Video Retrieval): [ACM MM version (Coming soon)](),  [arXiv version]().

> The code is modified from [ms-sl
](https://github.com/HuiGuanLab/ms-sl).

## Results
We compare our method against others on three benchmark datasets, i.e., TVR, Activitynet and Charades-STA:

|              | R@1  | R@5  | R@10 | R@100 | SumR  |
|--------------|------|------|------|-------|-------|
| TVR          | 13.5 | 32.1 | 43.4 | 83.4  | 172.3 |
| Activitynet  | 7.1  | 22.5 | 34.7 | 75.8  | 140.1 |
| Charades-STA | 1.8  | 7.1  | 11.8 | 47.7  | 68.4  |

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
Refer to [here](https://github.com/HuiGuanLab/ms-sl/tree/main/dataset) for detailed descriptions of the datasets.

4. Training and Inference

Set `SUB_LIST`, 
`OUTPUT` (dir for saving ckpts, log and results)
and `DATASET` ( ["samm" | "cas(me)^2"] )  in [pipeline.sh](https://github.com/xjtupanda/AUW-GCN/blob/main/pipeline.sh), then run:
```shell
$ bash pipeline.sh
```

**We also provide ckpts, logs, etc.** to reproduce the results in the paper, please download [ckpt.tar.gz](https://pan.baidu.com/s/1U-LEYH_fGOwgeToJ2Abhlw?pwd=5kan).
## Citation
If you feel this project helpful to your research, please cite our work. (To be updated when published on ICME.)
```
@inproceedings{Yin2023AUawareGC,
  title={AU-aware graph convolutional network for Macro- and Micro-expression spotting},
  author={Shukang Yin and Shiwei Wu and Tong Xu and Shifeng Liu and Sirui Zhao and Enhong Chen},
  year={2023}
}
```

##### Please email me at xjtupanda@mail.ustc.edu.cn if you have any inquiries or issues.
