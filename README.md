# OpenTLP

This repository provides reference implementations for some representative technqiues regarding temporal link prediction (TLP), a classic inference task on dynamic graphs. All the TLP methods are implemented via a unified encoder-decoder framework. In addition, this repository also summarizes some other open-source projects of existing TLP techniques.

Note that this repository is not the official implmentation of the related methods. Some of the implmented TLP approaches also need further parameter tuning to achive the best performance on different datasets.

This repository will keep updating to include some other TLP methods, task settings, dynamic graph datasets, etc.

### Usage

Some representative methods based on non-negative matrix factorization (NMF) are implemented via Matlab, including *CRJMF* [1], *GrNMF* [2], *DeepEye* [3], and *AM-NMF* [4]. The source code and data of these methods (implmented by Matlab) are put under directory ./Matlab. For each method, [method_name].m provides the functions that give the definitions of encoder and decoder, where the loss function is derived during the optimization of encoder. [method_name]_demo.m demonstrates how to use these functions. To run the demonstration code, please first unzip the data.zip in ./Matlab/data.

*TMF* [5] and *LIST* [6] are TLP methods based on the generic matrix factorization, which are implmented via PyTorch in this repository. Moreover, some deep learning (DL) based approaches are also implmented via PyTorch, including *Dyngraph2vec* [7], *DDNE* [8], *E-LSTM-D* [9], *GCN-GAN* [10], *NetworkGAN* [11], and *STGSN* [12]. The source code and data of these methods (implmented by Matlab) are put under directory ./Python. For each method, a set of classes are used to define the encoder, decoder, and loss function, which are put under the directory ./Python/[method_name]. Furthermore, [method_name].py demonstrates how to use these classes. To run the demonstration code, please first unzip the data.zip in ./Python/data.

### Other Open-Source Projects

There are some other open-source implementation for several TLP approaches, including [TLSI](https://github.com/linhongseba/Temporal-Network-Embedding) (Temporal Latnet Space Index) [13], [MLjFE](https://github.com/xkmaxidian/MLjFE) [14], [EvolveGCN](https://github.com/IBM/EvolveGCN) [15], [CTDNE](https://github.com/LogicJake/CTDNE) [16], [M2DNE](https://github.com/rootlu/MMDNE) [17], [DyRep](https://github.com/uoguelph-mlrg/LDG/blob/master/dyrep.py) [18], [TGAT](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs) [19], and [CAW](http://snap.stanford.edu/caw/) [20].

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

If you have any questions, you can contact the author via [mengqin_az@foxmail.com].