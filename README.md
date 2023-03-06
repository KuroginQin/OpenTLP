# OpenTLP

This repository is the open-source project of a survey paper entitled "Temporal Link Prediction: A Unified Framework, Taxonomy, and Review" (https://arxiv.org/abs/2210.08765). It refactors or implements some representative techniques of temporal link prediction (TLP), a classic inference task on dynamic graphs, based on a unified encoder-decoder framework and terminologies (e.g., task settings, taxonomy, etc.) introduced in the survey. In addition, this repository also summarizes some other open-source projects regarding TLP.

Note that this repository is not the official implementation of related methods. Some of the implemented TLP approaches also need further parameter tuning to achieve the best performance on different datasets. We will keep updating this repository to include some other (SOTA or classic) TLP methods, task settings, dynamic graph datasets, etc.

### Citing
If you find this project useful for research, please cite our survey paper.
```
@article{qin2022temporal,
  title={Temporal Link Prediction: A Unified Framework, Taxonomy, and Review}, 
  author={Meng Qin and Dit-Yan Yeung},
  journal={arXiv preprint arXiv:2210.08765},
  year={2022}
}

```
If you have any questions regarding this repository, you can contact the author via [mengqin_az@foxmail.com].

### Usage

We implement some representative TLP methods using MATLAB and PyTorch, with the corresponding code and data put under the direcories './Matlab' and './Python'.

For each method implemented by MATLAB, [method_name].m provides the functions that give the definitions of encoder and decoder, where the loss function is derived during the optimization of encoder. [method_name]_demo.m demonstrates how to use these functions. To run the demonstration code, please first unzip the data.zip in ./Matlab/data.

For each method implemented by PyTorch, a set of classes are used to define the encoder, decoder, and loss function, which are put under the directory './Python/[method_name]'. Furthermore, [method_name].py demonstrates how to use these classes. To run the demonstration code, please first unzip the data.zip in ./Python/data. In particular, these methods (implemented by PyTorch) can be speeded up via GPUs.

Details of the implemented TLP methods are summarized as follows.

| Methods | Data Models | Paradigms | Level | Attributes | Weighted TLP | Language |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| CRJMF [1] | Evenly-Spaced Snapshot | OTI | 1 | Static | Yes | MATLAB |
| GrNMF [2] | Evenly-Spaced Snapshot | OTI | 1 | N/A | Yes | MATLAB |
| DeepEye [3] | Evenly-Spaced Snapshot | OTI | 1 | N/A | Yes | MATLAB |
| AM-NMF [4] | Evenly-Spaced Snapshot | OTI | 1 | N/A | Yes | MATLAB |
| TMF [5] | Evenly-Spaced Snapshot | OTI | 1 | N/A | Yes | Python |
| LIST [6] | Evenly-Spaced Snapshot | OTI | 1 | N/A | Yes | Python |
| Dyngraph2vec [7] | Evenly-Spaced Snapshot | OTOG | 1 | N/A | Yes | Python |
| DDNE [8] | Evenly-Spaced Snapshot | OTOG | 1 | N/A | Yes | Python |
| E-LSTM-D [9] | Evenly-Spaced Snapshot | OTOG | 1 | N/A | Yes | Python |
| GCN-GAN [10] | Evenly-Spaced Snapshot | OTOG | 1 | N/A | Yes | Python |
| NetworkGAN [11] | Evenly-Spaced Snapshot | OTOG | 1 | N/A | Yes | Python |
| STGSN [12] | Evenly-Spaced Snapshot | OTOG | 2 | N/A | Yes | Python |

### Other Open-Source Projects & Sources

| Methods | Data Models | Paradigms | Level | Attributes | Weighted TLP|
| ---- | ---- | ---- | ---- | ---- | ---- |
| [TLSI](https://github.com/linhongseba/Temporal-Network-Embedding) [13] | Evenly-Spaced Snapshot | OTI | 1 | N/A | Yes |
| [MLjFE](https://github.com/xkmaxidian/MLjFE) [14] | Evenly-Spaced Snapshot | OTI | 1 | N/A | Yes |
| [EvolveGCN](https://github.com/IBM/EvolveGCN) [15] | Evenly-Spaced Snapshot | OTOG | 2 | Dynamic | D/L-Dep |
| [CTDNE](https://github.com/LogicJake/CTDNE) [16] | Unevenly-Spaced Edge Seq | OTOG | 1 | N/A | No |
| [M2DNE](https://github.com/rootlu/MMDNE) [17] | Unevenly-Spaced Edge Seq | OTOG | 1 | N/A | No |
| [DyRep](https://github.com/uoguelph-mlrg/LDG/blob/master/dyrep.py) [18] | Unevenly-Spaced Edge Seq | OTOG | 2 | Dynamic | No |
| [TGAT](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs) [19] | Unevenly-Spaced Edge Seq | OTOG | 2 | Dynamic | No |
| [CAW](http://snap.stanford.edu/caw/) [20] | Unevenly-Spaced Edge Seq | OTOG | 2 | Dynamic | No |
| [DySAT](https://github.com/aravindsankar28/DySAT) [21] | Evenly-Spaced Snapshot| OTOG | 2 | Dynamic | No |
| [TREND](https://github.com/WenZhihao666/TREND) [22] | Unevenly-Spaced Edge Seq | OTOG | 2 | Static | No |
| [DyGNN](https://github.com/alge24/dygnn) [23] | Unevenly-Spaced Edge Seq | OTOG | 1 | No | No |
| [IDEA](https://github.com/KuroginQin/IDEA) [24] | Evenly-Spaced Snapshot | OTOG | 2 | Static | Yes |
| [GSNOP](https://github.com/RManLuo/GSNOP) [25] | Unevenly-Spaced Edge Seq | OTOG | 2 | Dynamic | No |


### References
[1] Gao, Sheng, Ludovic Denoyer, and Patrick Gallinari. Temporal Link Prediction by Integrating Content and Structure Information. ACM CIKM, 2011.

[2] Ma, Xiaoke, Penggang Sun, and Yu Wang. Graph Regularized Nonnegative Matrix Factorization for Temporal Link Prediction in Dynamic Networks. Physica A: Statistical Mechanics & Its Applications 496 (2018): 121-136.

[3] Ahmed, Nahla Mohamed, et al. DeepEye: Link Prediction in Dynamic Networks Based on Non-Negative Matrix Factorization. Big Data Mining & Analytics 1.1 (2018): 19-33.

[4] Qin, Meng, et al. Adaptive Multiple Non-Negative Matrix Factorization for Temporal Link Prediction in Dynamic Networks. ACM SIGCOMM Workshop on Network Meets AI & ML, 2018.

[5] Yu, Wenchao, Charu C. Aggarwal, and Wei Wang. Temporally Factorized Network Modeling for Evolutionary Network Analysis. ACM WSDM, 2017.

[6] Yu, Wenchao, et al. Link Prediction with Spatial and Temporal Consistency in Dynamic Networks. IJCAI, 2017.

[7] Goyal, Palash, Sujit Rokka Chhetri, and Arquimedes Canedo. Dyngraph2vec: Capturing Network Dynamics Using Dynamic Graph Representation Learning. Knowledge-Based Systems 187 (2020): 104816.

[8] Li, Taisong, et al. Deep Dynamic Network Embedding for Link Prediction. IEEE Access 6 (2018): 29219-29230. 

[9] Chen, Jinyin, et al. E-LSTM-D: A Deep Learning Framework for Dynamic Network Link Prediction. IEEE Transactions on Systems, Man, & Cybernetics: Systems 51.6 (2019): 3699-3712.

[10] Qin, Meng et al. GCN-GAN: A Non-linear Temporal Link Prediction Model for Weighted Dynamic Networks. IEEE INFOCOM, 2019.

[11] Yang, Min, et al. An Advanced Deep Generative Framework for Temporal Link Prediction in Dynamic Networks. IEEE Transactions on Cybernetics 50.12 (2019): 4946-4957.

[12] Min, Shengjie, et al. STGSN—A Spatial–Temporal Graph Neural Network Framework for Time-Evolving Social Networks. Knowledge-Based Systems 214 (2021): 106746.

[13] Zhu, Linhong, et al. Scalable Temporal Latent Space Inference for Link Prediction in Dynamic Social Networks. IEEE Transactions on Knowledge & Data Engineering (TKDE) 28.10 (2016): 2765-2777.

[14] Ma, Xiaoke, et al. Joint Multi-Label Learning and Feature Extraction for Temporal Link Prediction. Pattern Recognition 121 (2022): 108216.

[15] Pareja, Aldo, et al. EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs. AAAI, 2020.

[16] Nguyen, Giang Hoang, et al. Continuous-Time Dynamic Network Embeddings. ACM Companion Proceedings of WWW, 2018.

[17] Lu, Yuanfu, et al. Temporal Network Embedding with Micro-and Macro-Dynamics. ACM CIKM, 2019.

[18] Trivedi, Rakshit, et al. DyRep: Learning Representations over Dynamic Graphs. ICLR, 2019.

[19] Xu, Da, et al. Inductive Representation Learning on Temporal Graphs. ICLR, 2020.

[20] Wang, Yanbang, et al. Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks. ICLR, 2021.

[21] Sankar, Aravind, et al. DySAT: Deep Neural Representation Learning on Dynamic Graphs via Self-Attention Networks. ACM WSDM, 2020.

[22] Wen, Zhihao, and Yuan Fang. TREND: TempoRal Event and Node Dynamics for Graph Representation Learning. ACM WWW, 2022.

[23] Ma, Yao, et al. Streaming Graph Neural Networks. ACM SIGIR, 2020.

[24] Qin, Meng, et al. High-Quality Temporal Link Prediction for Weighted Dynamic Graphs Via Inductive Embedding Aggregation. IEEE TKDE, 2023.

[25] Luo, Linhao, Gholamreza Haffari, and Shirui Pan. Graph Sequential Neural ODE Process for Link Prediction on Dynamic and Sparse Graphs. ACM WSDM, 2023.
