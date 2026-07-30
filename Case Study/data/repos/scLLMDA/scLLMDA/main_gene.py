import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import roc_auc_score
import scanpy as sc

import dataset_bert


genome = "scLLMDA/data/genome/mm10.fa.h5"
ds = dataset_bert.saveSingleCellDataset(dataset_bert.load_adata("scLLMDA/data/scATAC/MosA1_WholeBrainA.h5ad"), seq_len=1344, genome=genome)
embedding_path = "scLLMDA/out/MosA1_WholeBrainA/embeddings_genome.npz"
#print("end")


for i in range(len(ds)):

    embedding, cell_data, *_ = ds[i]  

    
    if i % 100 == 0:
        print(i)
        


    
ds.save_all_data(embedding_path)

saved_data = np.load(embedding_path)
print("嵌入形状:", saved_data['embedding_mean'].shape)  


