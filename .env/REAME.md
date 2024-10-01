# **SimRN: Trajectory Similarity Learning in Road Networks based on Distributed Deep Reinforcement Learning**
SimRN: An effective and efficient trajectory similarity learning framework for road networks. SimRN consists of three key modules: the spatio-temporal prompt information extraction (STP) module, the trajectory representation based on DRL (TrajRL) module, and the graph contrastive learning (GCL) module. The STP module captures spatio-temporal features from road networks to improve the training of the trajectory representation. The TrajRL module automatically selects optimal parameters and enables parallel training, improving both trajectory representation and the efficiency of similarity computations. The GCL module employs a self-supervised contrastive learning paradigm to generate sufficient samples while preserving spatial constraints and temporal dependencies of trajectories. 

## Requirements

* Python 3.7
* Tensorflow 2.15.0
* CUDA 11.5
* NVIDIA 3090 GPUs

Please refer to the source code to install all required packages in Python.

## Datasets
We use two real-life datasets: T-Drive and Porto, which can be downloaded from the URLs (https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/) and (https://archive.ics.uci.edu/dataset/339/). 

## To Run Experimental Case

+ Trajectory preprocessing

1. run "agents.RN_preprocess_1.py" to generate the prompt information of the road network.

2. run "agents.spatial_matrix_2.py" to compute the distance matrix of the road vertices.

3. run "agents.traj_preprocess_3.py" to preprocess the trajectories, containing road network node embedding, date embedding, etc.

+ Model training and similarity computation

1. run "state_main.py" to start the agent "TrajRL" to make decisons, containing action selecting, sample generations, model training, and similarity computation.

Note that, (i) the ground truth of six non-learning-based methods on two datasets occupies too much memory, as these matrices are not stored in this repository; (ii) the datasets are stored in ".env/Env/data".