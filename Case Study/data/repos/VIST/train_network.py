'''
library for analysis code
'''
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
import torch.optim as optim
import scipy
import statistics
import random
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


import pyreadr

import sys
sys.path.insert(1, 'path to SNOW.py')
from SNOW import *

'''
load data
'''
anndata = sc.read_h5ad('path to data')
anndata.obs['time_by_rep'] = anndata.obs.time2.copy(deep=True)
anndata.obs.time_by_rep[anndata.obs.Repeats == 'LD_2'] = anndata.obs.time_by_rep[anndata.obs.Repeats == 'LD_2'] + 24
anndata.obs.time_by_rep[anndata.obs.Repeats == 'DD_2'] = anndata.obs.time_by_rep[anndata.obs.Repeats == 'DD_2'] + 24



anndata = anndata[anndata.obs.experiment == 'CLK856_LD']
sc.pp.highly_variable_genes(anndata, flavor='seurat_v3', n_top_genes=2000)
gene_idx = anndata.var.highly_variable

anndata = anndata[:, gene_idx]
NvarGenes = gene_idx.sum()

Ndim = 32
smoothPower = 40
batchsize = 300



'''
initialize neural network
'''
model = SNOW_v1(NvarGenes, 
                128, 
                Ndim, 
                timeDim, 
                fixStd = True,
                fixVal = 0.5
                   ) # default 128 # was 16 and 64

trained_model = train_net(anndata, model, 'time_by_rep', itr = 500, model_name = 'path to save model', smoothPower=smoothPower)
