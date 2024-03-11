# CUCL: Codebook for Unsupervised Continual Learning

Cheng Chen, Jingkuan Song, Xiaosu Zhu, Junchen Zhu, LianLi Gao, Hnegtao Shen
ACM MM 2023.

<img src="./image/architecture snapshot.png">

## Abstract
The focus of this study is on Unsupervised Continual Learning (UCL), as it presents an alternative to Supervised Continual Learning which needs high-quality manual labeled data. The experiments under UCL paradigm indicate a phenomenon where the results on the first few tasks are suboptimal. This phenomenon can render the model inappropriate for practical applications. To address this issue, after analyzing the phenomenon and identifying the lack of diversity as a vital factor, we propose a method named Codebook for Unsupervised Continual Learning (CUCL) which promotes the model to learn discriminative features to complete the class boundary. Specifically, we first introduce a Product Quantization to inject diversity into the representation and apply a cross quantized contrastive loss between the original representation and the quantized one to capture discriminative information. Then, based on the quantizer, we propose a effective Codebook Rehearsal to address catastrophic forgetting. This study involves conducting extensive experiments on CIFAR100, TinyImageNet, and MiniImageNet benchmark datasets. Our method significantly boosts the performances of supervised and unsupervised methods. For instance, on TinyImageNet, our method led to a relative improvement of 12.76% and 7% when compared with Simsiam and BYOL, respectively

## Installation
1. Clone this repository and navigate to CUCL folder
``` 
git clone git@github.com:zackschen/CUCL.git
cd CUCL 
```
2. Install Package
```
conda create -n cucl python=3.8 -y
conda activate cucl
pip install -r requirements.txt
```

## Experiments
This repository currently contains experiments reported in the paper for CIFAR-100, TinyImageNet, MiniImageNet. 
For simsiam on  CIFAR-100 dataset, experiments can be run using the following command:

```python
bash scripts/unsupervised/cifar100/simsiam_CUCL.sh $GPU_ID $MEMORY_SIZE
```
For various experiments, you should know the role of each argument.

- `Task`：The number of all training task, such as [5,10,20].
- `CUCL`: Whether useing CUCL.
- `buffer_size`: The buffer size for rehearsal samples.
- `N_books`: The size of codebooks.
- `N_words`: The size of codewords.
- `L_word`: The dimension of the representation.

For other unsupervised methods or datasets, you could train other scripts in the scripts directory.

## Citation
```
@inproceedings{cheng2023cucl,
  title={CUCL: Codebook for Unsupervised Continual Learning},
  author={Cheng, Chen and Song, Jingkuan and Zhu, Xiaosu and Zhu, Junchen and Gao, Lianli and Shen, Hengtao},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={1729--1737},
  year={2023}
}
```
## Acknowledgement
[solo-learn](https://github.com/vturrisi/solo-learn): the codebase we built upon.