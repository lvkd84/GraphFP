# GraphFP: Fragment-based Pretraining and Finetuning on Molecular Graphs

**NeurIPS 2023**

Authors: Kha-Dinh Luong, Ambuj Singh

[[Paper](https://openreview.net/forum?id=77Nq1KjmLl)]
[[ArXiv](https://arxiv.org/abs/2310.03274)]

In this repository, we provide the code used in the NeurIPS 2023 paper **Fragment-based Pretraining and Finetuning on Molecular Graphs**. We also include the model weights to reproduce the results from the paper.

<p align="center">
  <img src="figs/contrastive_main.png"/> 
</p>

<p align="center">
  <img src="figs/predictive_main.png"/> 
</p>

## Environment

The source code is organized into 2 directories:
* `mol`: experiments on small molecule prediction tasks, corresponding to `Table 2` from the paper.
* `long-range`: experiments on long-range datasets, corresponding to `Table 3` from the paper.

The folders contain different chemical featurizations and GNN implementations. To ensure fair comparison with previous works, the `mol` folder contains codes compatible with those from Hu et al. (2019). On the other hand, `long-range` folder contains newer codes, with featurizations and GNN implementations provided in Open Graph Benchmark. As a result, the environment requirements for both scenarios are also different.

The package requirements for `mol` and `long-range` are in `mol/requirements.txt` and `long-range/requirements.txt`, respectively.

## Data Preparation

## Pretraining

## Finetuning


