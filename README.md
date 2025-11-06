# MAKGLPIN
A Deep Learning Framework for Accurate lncRNA-Protein Interaction Prediction via Multi-Kernel Graph Embeddings and Self-Attention

## Project Overview

## Publication
Shivani Saxena, Ahsan Z Rizvi, A Deep Learning Framework for Accurate lncRNA-Protein Interaction Prediction via Multi-Kernel Graph Embeddings and Self-Attention, _Springer Nature Interdisciplinary Sciences - Computational Life Sciences (INSC)  _, 2025, (Under Review)

## Data Description
The dataset used in this project is stored in a data directory and includes lncRNA and protein feautures for training and testing. These datasets contain arrays where each sample corresponds to different features. 

## Installation
1. Prerequisites
Make sure you have following libraries installed.
* matplotlib==3.10.1
* numpy==2.2.5
* pandas==2.2.3
* scikit_learn==1.6.1
* torch==2.4.1
* torch_geometric==2.6.1
* torch_sparse==0.6.18+pt24cu121
* networkx==2.4
* scipy==1.5.4 

2. Run

To train the model, run the main.py script. You can customize training parameters such as the number of epochs, learning rate, and other hyperparameters by passing arguments to the script.

```bash
python3 code/main.py --epoch XXXX --lr XXXX
```
## Contact
Dr Ahsan Z Rizvi, ahsan.rizvi@iar.ac.in
