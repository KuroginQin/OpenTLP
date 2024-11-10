[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![Stars](https://img.shields.io/github/stars/KuroginQin/OpenTLP?color=yellow)  ![Forks](https://img.shields.io/github/forks/KuroginQin/OpenTLP?color=blue&label=Fork)

# OpenTLP

This repository is the open-source project of a survey paper entitled "[**Temporal Link Prediction: A Unified Framework, Taxonomy, and Review**](https://dl.acm.org/doi/10.1145/3625820)", which has been accepted by **ACM Computing Surveys**. It refactors or implements some representative temporal link prediction (TLP, a.k.a. dynamic link prediction) methods (especially for those do not provide their source code) based on the unified encoder-decoder framework and terminologies (regarding task settings, taxonomy, etc.) introduced in our survey paper. In addition, this repository also summarizes some other open-source projects regarding TLP. We will keep updating this repository to include some other (SOTA or classic) TLP methods, task settings, dynamic graph datasets, etc.

Note that this repository is not the official implementation of related methods. Some of the implemented TLP approaches also need careful parameter tuning to achieve the best performance on different datasets with different task settings.

## Abstract
Dynamic graphs serve as a generic abstraction and description of the evolutionary behaviors of various complex systems (e.g., social networks and communication networks). Temporal link prediction (TLP) is a classic yet challenging inference task on dynamic graphs, which predicts possible future linkage based on historical topology. The predicted future topology can be used to support some advanced applications on real-world systems (e.g., resource pre-allocation) for better system performance. This survey provides a comprehensive review of existing TLP methods. Concretely, we first give the formal problem statements and preliminaries regarding data models, task settings, and learning paradigms that are commonly used in related research. A hierarchical fine-grained taxonomy is further introduced to categorize existing methods in terms of their data models, learning paradigms, and techniques. From a generic perspective, we propose a unified encoder-decoder framework to formulate all the methods reviewed, where different approaches only differ in terms of some components of the framework. Moreover, we envision serving the community with an open-source project OpenTLP that refactors or implements some representative TLP methods using the proposed unified framework and summarizes other public resources. As a conclusion, we finally discuss advanced topics in recent research and highlight possible future directions.

## Citing
If you find this project useful for your research, please cite our survey paper.
```
@article{qin2023temporal,
  title={Temporal link prediction: A unified framework, taxonomy, and review},
  author={Qin, Meng and Yeung, Dit-Yan},
  journal={ACM Computing Surveys},
  volume={56},
  number={4},
  pages={1--40},
  year={2023},
  publisher={ACM New York, NY, USA}
}

```
If you have any questions regarding this repository, you can contact the author via [mengqin_az@foxmail.com].

## Outline

- [OpenTLP](#opentlp)
  - [Abstract](#abstract)
  - [Citing](#citing)
  - [Outline](#outline)
  - [Notations](#notations)
  - [Implemented or Refactored TLP Methods](#implemented-or-refactored-tlp-methods)
  - [Other Open Source Projects Regarding TLP](#other-open-source-projects-regarding-tlp)
  - [Public Datasets for TLP](#public-datasets-for-tlp)
  - [Other Related Survey Papers](#other-related-survey-papers)
  - [Advanced Applications Supported by TLP](#advanced-applications-supported-by-tlp)
  - [References](#references)

## Notations

- **ESSD** - Evenly-Sampled Snapshot Sequence Description, a.k.a. Discrete-Time Dynamic Graph (DTDG) in other literature.

- **UESD** - Unevenly-Sampled Edge Sequence Description, a.k.a. Continuous-Time Dynamic Graph (CTDG) in other literature.

(We argure that CTDG cannot precisely describe the corresponding data model. Although the time index associated with each edge is defined on a continuous domain in the second data model, the edge sequence used to encode the dynamic topology is still discrete. The term of *continuous* description may be ambiguous in some cases. Therefore, we use ESSD & UESD instead of DTDG & CTDG.)

- **DI** - Direct Inference.

- **OTI** - Online Training & Inference.

- **OTOG** -  Offline Training & Online Generalization.

- **HQ** - Able to Derive High-Quality Prediction Results for Weighted TLP.

- **D/L-Dep** - the original version of a method cannot support the weighted TLP but it can be easily extended to tackle such a setting by replacing an appropriate decoder or loss.

## Implemented or Refactored TLP Methods

We implement some representative TLP methods using MATLAB and PyTorch, with the corresponding code and data put under the direcories './Matlab' and './Python'.

For each method implemented by MATLAB, [method_name].m provides the functions that give the definitions of encoder and decoder, where the loss function is derived during the optimization of encoder. [method_name]_demo.m demonstrates how to use these functions. To run the demonstration code, please first unzip the data.zip in ./Matlab/data.

For each method implemented by PyTorch, a set of classes are used to define the encoder, decoder, and loss function, which are put under the directory './Python/[method_name]'. Furthermore, [method_name].py demonstrates how to use these classes. To run the demonstration code, please first unzip the data.zip in ./Python/data. In particular, these methods (implemented by PyTorch) can be speeded up via GPUs.

Details of the implemented TLP methods are summarized as follows.

| Methods | Publication Venues | Data Models | Paradigms | Level | Attributes | Weighted TLP | Language |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| CRJMF [[1]](https://dl.acm.org/doi/abs/10.1145/2063576.2063744?casa_token=d6PWd64Tl78AAAAA%3Af_H3CVdubxYNhcdzi0EEVnsrXs8uv7z5fGUdEpJuXYdVvBsInSJosQn8Pfv-tVD1HF1ce7EMYtU5uA) | CIKM 2011 | ESSD | OTI | 1 | Static | Able | MATLAB |
| GrNMF [[2]](https://www.sciencedirect.com/science/article/pii/S0378437117313316?casa_token=DZAy-hjY-f0AAAAA:4T_uyy39tKD59i5aXVINCPB0MN5WVArzPz5cFPq2ZZ_4c_RJXPefNxE_Rvj2gJso75z69RUYhUM) | Physica A 2018 | ESSD | OTI | 1 | N/A | Able | MATLAB |
| DeepEye [[3]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8268733) | BDMA 2018 | ESSD | OTI | 1 | N/A | Able | MATLAB |
| AM-NMF [[4]](https://dl.acm.org/doi/pdf/10.1145/3229543.3229546) | SIGCOMM 2018 NetAI WKSP | ESSD | OTI | 1 | N/A | Able | MATLAB |
| TMF [[5]](https://dl.acm.org/doi/pdf/10.1145/3018661.3018669) | WSDM 2017 | ESSD | OTI | 1 | N/A | Able | Python |
| LIST [[6]](https://www.ijcai.org/Proceedings/2017/0467.pdf) | IJCAI 2017 | ESSD | OTI | 1 | N/A | Able | Python |
| Dyngraph2vec [[7]](https://www.sciencedirect.com/science/article/abs/pii/S0950705119302916?casa_token=jleYHxT0wYkAAAAA:8Xc-44_qvCjOTz3AIAngfjMY736p0GCIPOW65eNhHIeH-cEEKJVMBN0eXDNRGIQxIHiiDa6meAw) | KBS 2020 | ESSD | OTOG | 1 | N/A | Able | Python |
| DDNE [[8]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8365780) | IEEE Access 2018 | ESSD | OTOG | 1 | N/A | D/L-Dep | Python |
| E-LSTM-D [[9]](https://ieeexplore.ieee.org/document/8809903) | Trans on SMC: Sys 2019 | ESSD | OTOG | 1 | N/A | Able | Python |
| GCN-GAN [[10]](https://ieeexplore.ieee.org/document/8737631) | InfoCom 2019 | ESSD | OTOG | 1 | N/A | Able (HQ) | Python |
| NetworkGAN [[11]](https://ieeexplore.ieee.org/document/8736786/) | Trans on Cyb 2019 | ESSD | OTOG | 1 | N/A | Able (HQ) | Python |
| STGSN [[12]](https://www.sciencedirect.com/science/article/abs/pii/S0950705121000095?casa_token=t7qWld7kbUUAAAAA:sZXXCqJgvK6SgP7vKMBTiv7YFA-kikEJQqMfxmdlcMy-P4NiE3g1JkMAYWayg4DGfGcKPOGJL7s) | KBS 2021 | ESSD | OTOG | 2 | Dynamic | Able | Python |

## Other Open Source Projects Regarding TLP

| Methods | Publication Venues | Data Models | Paradigms | Level | Attributes | Weighted TLP|
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| TLSI [[13]](https://ieeexplore.ieee.org/document/7511675/) [(Code)](https://github.com/linhongseba/Temporal-Network-Embedding) | TKDE 2016 | ESSD | OTI | 1 | N/A | Able |
| MLjFE [[14]](https://www.sciencedirect.com/science/article/pii/S0031320321003976?casa_token=Gub6BWKZu04AAAAA:JRCE5S2P_DSqcbcb2TTAaUPO66EKZAjyxReCCMU_tubFaIhoMFkIMgKBGpFBVdUyBcz2BRXmugo) [(Code)](https://github.com/xkmaxidian/MLjFE) | Pattern Recognition 2022 | ESSD | OTI | 1 | N/A | Able |
| E-LSTM-D [[9]](https://ieeexplore.ieee.org/document/8809903) [(Code)](https://github.com/jianz94/E-lstm-d/) | Trans on SMC: Sys 2019 | ESSD | OTOG | 1 | N/A | Able |
| EvolveGCN [[15]](https://ojs.aaai.org/index.php/AAAI/article/view/5984/5840) [(Code)](https://github.com/IBM/EvolveGCN) | AAAI 2020 | ESSD | OTOG | 2 | Dynamic | D/L-Dep |
| CTDNE [[16]](https://dl.acm.org/doi/pdf/10.1145/3184558.3191526) [(Code)](https://github.com/LogicJake/CTDNE) | WWW 2018 Companion | UESD | OTOG | 1 | N/A | Unable |
| M2DNE [[17]](https://dl.acm.org/doi/abs/10.1145/3357384.3357943?casa_token=gHmbPKCe6SIAAAAA%3AYnNL6vEKb244ITZZshHsdaEjXQLDrN6g6qMixzzEw92Uvtv3SDt1lqSVsxZTAbkURAzKoRDkBdxteQ) [(Code)](https://github.com/rootlu/MMDNE) | CIKM 2019 | UESD | OTOG | 1 | N/A | Unable |
| DyRep [[18]](https://par.nsf.gov/servlets/purl/10099025) [(Code)](https://github.com/uoguelph-mlrg/LDG/blob/master/dyrep.py) | ICLR 2019 | UESD | OTOG | 2 | Dynamic | Unable |
| TGAT [[19]](https://arxiv.org/pdf/2002.07962.pdf) [(Code)](https://github.com/StatsDLMathsRecomSys/Inductive-representation-learning-on-temporal-graphs) | ICLR 2020 | UESD | OTOG | 2 | Dynamic | Unable |
| CAW [[20]](https://arxiv.org/pdf/2101.05974.pdf) [(Code)](http://snap.stanford.edu/caw/) | ICLR 2021 | UESD | OTOG | 2 | Dynamic | Unable |
| DySAT [[21]](https://dl.acm.org/doi/pdf/10.1145/3336191.3371845?casa_token=m_WIXUGiQzAAAAAA:9S2TsDsXckE3nN-X_LZi0cC7RpZ06vSKPS8XoEGm48Oq78nxFykwc9XuJY9c7o6V3WNwCE1olbVQcg) [(Code)](https://github.com/aravindsankar28/DySAT) | WSDM 2020 | ESSD | OTOG | 2 | Dynamic | Unable |
| TREND [[22]](https://dl.acm.org/doi/abs/10.1145/3485447.3512164?casa_token=stprVUyQgxMAAAAA%3AOu46mChOHrvnfUndpx5bra0mlgyNQ-RSHodQR284gg1HUdl3Dbj2bI-aBV_DMSxAQdD0ccuaSWpGZQ) [(Code)](https://github.com/WenZhihao666/TREND) | WWW 2022 | UESD | OTOG | 2 | Static | Unable |
| DyGNN [[23]](https://dl.acm.org/doi/pdf/10.1145/3397271.3401092) [(Code)](https://github.com/alge24/dygnn) | SIGIR 2020 | UESD | OTOG | 1 | N/A | Unable  |
| IDEA [[24]](https://ieeexplore.ieee.org/iel7/69/4358933/10026343.pdf?casa_token=WyWGel0ps3kAAAAA:5cKey6Lg4NS0NWaEpkAGKD4NsmJFqNsS0lnQ88q1ssl8-gNuRAWWk-DC8LIGA6WrnE_MAX-Law) [(Code)](https://github.com/KuroginQin/IDEA) | TKDE 2023 | ESSD | OTOG | 2 | Static | Able (HQ) |
| GSNOP [[25]](https://dl.acm.org/doi/abs/10.1145/3539597.3570465?casa_token=v-iRtU8PPZkAAAAA%3AEma5KziE6VruVIN8V_xzom58GLGcyVNDmZlya5CsHeOgqjWNzaH-buxe15FwxEppqvx79wJT9jfzNg) [(Code)](https://github.com/RManLuo/GSNOP) | WSDM 2023 | UESD | OTOG | 2 | Dynamic | Unable |
| TGN [[26]](https://grlplus.github.io/papers/58.pdf) [(Code)](https://github.com/twitter-research/tgn) | ICML 2020 WKSP | UESD | OTOG | 2 | Dynamic | Unable |
| MNCI [[27]](https://dl.acm.org/doi/abs/10.1145/3404835.3463052?casa_token=YNsyAI4l2zoAAAAA%3AFzclItIGp9VmpJXp88q7v7XwV9vPYeHE2lBUTf-rtNbnaI3yJvWlhwlReheF8vL_GERyXSBGKjAyyQ) [(Code)](https://github.com/MGitHubL/MNCI) | SIGIR 2021 | UESD | OTOG | 2 | N/A | Unable |
| TDGNN [[28]](https://dl.acm.org/doi/abs/10.1145/3366423.3380073?casa_token=jLhry0KLfTkAAAAA%3AM3P8hualerbyJaM34lWGwHAqIuN9d6lkh2nN5fqTIfo57Zx-3pKY7ifTQi-XMb7VRdAuHx_Lt7y4Yw) [(Code)](https://github.com/Leo-Q-316/TDGNN) | WWW 2020 | UESD | OTOG | 1 | Static | Unable |
| HTNE [[29]](https://dl.acm.org/doi/pdf/10.1145/3219819.3220054?casa_token=Ddsd4L9GsSkAAAAA:JJbGLEFDF82wccb0txK4FfEMVDbiADcxeU8sp2itTxsZcOhlRQ_VD206kJJ9GyRpEvwqfDAKu17b2w) [(Code)](http://zuoyuan.github.io/files/htne.zip) | KDD 2018 | UESD | OTI | 1 | N/A | Unable |
| SGR [[30]](https://ieeexplore.ieee.org/abstract/document/10183879) [(Code)](https://github.com/yinyanting123/SRG) | TKDE 2023 | ESSD | OTOG | 1 | N/A | Able |
| EdgeBank [[31]](https://proceedings.neurips.cc/paper_files/paper/2022/file/d49042a5d49818711c401d34172f9900-Paper-Datasets_and_Benchmarks.pdf) [(Code)](https://github.com/fpour/DGB) | NIPS 2022 | UESD | OTOG | 2 | Dynamic | Unable |
| GraphMixer [[32]](https://openreview.net/pdf?id=ayPPc0SyLv1) [(Code)](https://github.com/CongWeilin/GraphMixer) | ICLR 2023 | UESD | OTOG | 2 | Dynamic | Unable |
| DyGFormer [[33]](https://proceedings.neurips.cc/paper_files/paper/2023/file/d611019afba70d547bd595e8a4158f55-Paper-Conference.pdf) [(Code)](https://github.com/yule-BUAA/DyGLib) | NIPS2023 | UESD | OTOG | 2 | Dynamic | Unable |

## Public Datasets for TLP

| Datasets | Scenarios | Nodes | Edges | Wei Links | Min Time Granularity | Data Models | Levels |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| [Social Evolution](http://realitycommons.media.mit.edu/socialevolution.html) | Social Network | Cell phones | Bluetooth signal, calls, or messages between cell phones | No | 1 min | UESD | 2 |
| [CollegeMsg](https://snap.stanford.edu/data/CollegeMsg.html) | Online social network | App users | Messages sent from a source user to a destination user | No | 1 sec | UESD | 2 |
| [Wiki-Talk](https://snap.stanford.edu/data/wiki-talk-temporal.html) | Online social network | Wikipedia users | Relations that a user edits another user's talk page | No | 1 sec | UESD | 2 |
| [Enron](http://konect.cc/networks/enron/) | Email network | Email users | Emails from a source user to a destination user | No | 1 sec | UESD | 2 |
| [Reddit-Hyperlink](https://snap.stanford.edu/data/soc-RedditHyperlinks.html) | Hyperlink network | Subreddits | Hyperlinks from one subreddit to another | No | 1 sec | UESD | 2 |
| [DBLP](https://dblp.uni-trier.de/xml/) | Paper collaboration network | Paper authors | Collaboration relations between authors | No | 1 day | UESD | 2 |
| [AS-733](https://snap.stanford.edu/data/as-733.html) | BGP autonomous systems of Internet | BGP routers | Who-talks-to-whom communication between routers | No | 1 day | ESSD | 2 |
| [Bitcoin-Alpha](https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html) | Bitcoin transaction network | Bitcoin users | Trust scores between users | Yes |1 sec | UESD | 2 |
| [Bitcoin-OTC](https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html) | Bitcoin transaction network | Bitcoin users | Trust scores between users | Yes | 1 sec | UESD | 2 |
| [UCSB-Mesh](https://ieee-dataport.org/open-access/crawdad-ucsbmeshnet) | Wireless mesh network | Wireless routers | Link quality (in terms of expected transmission time) between routers | Yes | 1 min | ESSD | 1 |
| [NumFabric](https://github.com/shouxi/numfabric) | (Simulated) data center network | Host servers | Traffic (in terms of KB) between host servers | Yes | 1e-6 sec | UESD | 1 |
| [UCSD-WTD](http://www.sysnet.ucsd.edu/wtd) | WiFi mobility network | Access points/PDA devices | Signal strength (in terms of dBm) between access points and PAD devices | Yes | 1 sec | UESD | 2 |
| [UNSW-IoT](https://iotanalytics.unsw.edu.au/iottraces.html) | IoT network | IoT devices | Traffic (in terms of KB) between IoT devices | Yes | 1e-6 sec | UESD | 2 |
| [WIDE](https://mawi.wide.ad.jp/mawi) | Internet backbone | Host servers/user devices | Traffic (in terms of KB) between servers/devices | Yes | 1e-6 sec | UESD | 2 |

## Other Related Survey Papers

**Dynamic Network Embedding** (DNE):

- Kazemi, Seyed Mehran, et al. [Representation Learning for Dynamic Graphs: A Survey](https://www.jmlr.org/papers/volume21/19-447/19-447.pdf). JMLR 21.1 (2020): 2648-2720.

- Xue, Guotong, et al. [Dynamic Network Embedding Survey](https://www.sciencedirect.com/science/article/abs/pii/S0925231221016234?casa_token=dAXjY5X0S6cAAAAA:PGvwJI9tA11Rbj0yNiTrJJZnd_3YYjGBVeniVM3wb5B379h_36BrZ3kO5hLvrsVnFggipH0ksys). Neurocomputing 472 (2022): 212-223.

- Barros, Claudio DT, et al. [A Survey on Embedding Dynamic Graphs](https://dl.acm.org/doi/abs/10.1145/3483595?casa_token=5sIpuuKRIP4AAAAA%3AfB2EcuVp4EZPwb40uztI0d6y1TPBrMvryRVgjgNPRA7iPr9pwCM1bhVsSg9yJIQjppKeyIb5GSxreA). ACM Computing Surveys (CSUR) 55.1 (2021): 1-37.

- Skarding, Joakim, Bogdan Gabrys, and Katarzyna Musial. [Foundations and Modeling of Dynamic Networks Using Dynamic Graph Neural Networks: A survey](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9439502). IEEE Access 9 (2021): 79143-79168.

- Yang, Leshanshui, Sébastien Adam, and Clément Chatelain. [Dynamic Graph Representation Learning with Neural Networks: A Survey](https://arxiv.org/pdf/2304.05729.pdf). arXiv preprint arXiv:2304.05729 (2023).

(Note that there remain some gaps between the research on DNE and TLP. On the one hand, some classic TLP approaches (e.g., neighbor similarity & graph summarization) are not based on the DNE framework. On the other hand, some DNE models are *task-dependent* with specific model architectures and objectives designed only for TLP.  Moreover, most *task-independent* DNE techniques can only support simple TLP settings based on some common but naive strategies, e.g., treating the prediction of unweighted links as binary edge classification. Existing survey papers on DNE lack detailed discussions regarding whether and how a DNE method can be used to handle different settings of TLP.)

**Temporal Link Prediction** (TLP):

- Haghani, Sogol, and Mohammad Reza Keyvanpour. [Temporal Link Prediction: Techniques and Challenges](https://csit.am/2017/Proceedings/AICM/AICM6.pdf). Computer Science & Information Technologies. Yerevan (2017).

- Divakaran, Aswathy, and Anuraj Mohan. [Temporal Link Prediction: A Survey](https://link.springer.com/article/10.1007/s00354-019-00065-z)." New Generation Computing 38 (2020): 213-258.

## Advanced Applications Supported by TLP

**Intrusion Detection, Threat Prediction, & Lateral Movement Inference for Cyber Security**

- King, Isaiah J., and H. Howie Huang. [Euler: Detecting Network Lateral Movement via Scalable Temporal Link Prediction](https://dl.acm.org/doi/abs/10.1145/3588771). ACM TOPS (2023).

- Zhao, Jun, et al. [Cyber Threat Prediction Using Dynamic Heterogeneous Graph Learning](https://www.sciencedirect.com/science/article/pii/S0950705121011564). KBS 240 (2022): 108086.

- Khoury, Joseph, et al. [Jbeil: Temporal Graph-Based Inductive Learning to Infer Lateral Movement in Evolving Enterprise Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10646732). IEEE Symposium on Security and Privacy (SP): 3644-3660.

- Li, Xiaohui, et al. [Combating Temporal Composition Inference by High-Order Camouflaged Network Topology Obfuscation](https://www.sciencedirect.com/science/article/pii/S0167404824002864). Computers & Security 144 (2024): 103981.

**Channel Allocation in Wireless IoT Networks**

- Gao, Weifeng, et al. [Edge-Computing-Based Channel Allocation for Deadline-Driven IoT Networks](https://ieeexplore.ieee.org/abstract/document/8998165). IEEE TII 16.10 (2020): 6693-6702.

**Burst Traffic Detection & Dynamic Routing in Optical Networks**

- Vinchoff, Connor, et al. [Traffic prediction in optical networks using graph convolutional generative adversarial networks](https://ieeexplore.ieee.org/abstract/document/9203477). IEEE ICTON 2020.

- Aibin, Michał, et al. [On short-and long-term traffic prediction in optical networks using machine learning](https://ieeexplore.ieee.org/abstract/document/9492437). IEEE ONDM 2021.

**Dynamic Routing & Topology Control in Mobile Ad Hoc Networks**
- Guan, Quansheng, et al. [Prediction-Based Topology Control and Routing in Cognitive Radio Mobile Ad Hoc Networks](https://ieeexplore.ieee.org/abstract/document/5560849/). IEEE TOVT 59.9 (2010): 4443-4452.

- Liao, Ziliang, Linlan Liu, and Yubin Chen. [A Novel Link Prediction Method for Opportunistic Networks Based on Random Walk and a Deep Belief Network](https://ieeexplore.ieee.org/abstract/document/8962072). IEEE Access 8 (2020): 16236-16247.

**Dynamics Simulation & Conformational Analysis of Molecules**

- Ashby, Michael Hunter, and Jenna A. Bilbrey. [Geometric learning of the conformational dynamics of molecules using dynamic graph neural networks](https://arxiv.org/abs/2106.13277). arXiv preprint arXiv:2106.13277 (2021).

**Traffic Demand Prediction in Urban Computing**

- Liu, Chenxi, et al. [Foreseeing private car transfer between urban regions with multiple graph-based generative adversarial networks](https://link.springer.com/article/10.1007/s11280-021-00995-z). WWWJ 25.6 (2022): 2515-2534.

- Wang, Qiang, et al. [TGAE: Temporal Graph Autoencoder for Travel Forecasting](https://ieeexplore.ieee.org/abstract/document/9889163). IEEE T-ITS (2022).

**Log Anomaly Detection in Distributed Computing**

- Yan, Lejing, Chao Luo, and Rui Shao. [Discrete Log Anomaly Detection: A Novel Time-Aware Graph-Based Link Prediction Approach](https://www.sciencedirect.com/science/article/pii/S0020025523011611). Information Sciences 647 (2023): 119576.

**Trust Evaluation in BitCoin Transaction**

- Wang, Jie, et al. [TrustGuard: GNN-Based Robust and Explainable Trust Evaluation With Dynamicity Support](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10398459). IEEE TDSC (2024).

**Dynamic Scene Modeling**

- Kurenkov, Andrey, et al. [Modeling Dynamic Environments with Scene Graph Memory](https://proceedings.mlr.press/v202/kurenkov23a/kurenkov23a.pdf). ICML 2023: 17976-17993.

## References
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

[26] Rossi, Emanuele, et al. Temporal Graph Networks for Deep Learning on Dynamic Graphs. ICML Workshop on Graph Representation Learning, 2020.

[27] Liu, Meng, and Yong Liu. Inductive Representation Learning in Temporal Networks via Mining Neighborhood and Community Influences. ACM SIGIR, 2021.

[28] Qu, Liang, et al. Continuous-Time Link Prediction via Temporal Dependent Graph Neural Network. ACM WWW, 2020.

[29] Zuo, Yuan, et al. Embedding Temporal Network via Neighborhood Formation. ACM KDD, 2018.

[30] Yin, Yanting, et al. Super Resolution Graph With Conditional Normalizing Flows for Temporal Link Prediction. IEEE TKDE, 2023.

[31] Poursafaei, Farimah, et al. Towards Better Evaluation for Dynamic Link Prediction. NIPS, 2022.

[32] Cong, Weilin, et al. Do We Really Need Complicated Model Architectures for Temporal Networks? ICLR, 2023.

[33] Yu, Le, et al. Towards Better Dynamic Graph Learning: New Architecture and Unified Library. NIPS, 2023.