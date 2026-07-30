import warnings
warnings.filterwarnings("ignore")
import argparse
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops
import scanpy as sc


from dataset import load_ATAC_dataset
from data_utils import load_fixed_splits, adj_mul
import pandas as pd
from tqdm import tqdm
import pickle

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--train_name", nargs="+", default=['MosA1'])
parser.add_argument("--test_name", nargs="+", default=['WholeBrainA'])
args = parser.parse_args()
train_name_list = args.train_name
test_name = args.test_name

print(train_name_list)
print(test_name)



data_dir = "scLLMDA/out/MosA1_WholeBrainA/embeddings_cell.h5ad"
save_path = "scLLMDA/out/MosA1_WholeBrainA"


input_filename_edge = 'scLLMDA/out/MosA1_WholeBrainA/edge.txt'  
output_filename_source_edge = 'scLLMDA/out/MosA1_WholeBrainA/source64/bert/raw/bert_edgelist.txt'  
output_filename_target_edge = 'scLLMDA/out/MosA1_WholeBrainA/target64/bert/raw/bert_edgelist.txt'  


output_filename_source_feature = 'scLLMDA/out/MosA1_WholeBrainA/source64/bert/raw/bert_docs.txt'  
output_filename_target_feature = 'scLLMDA/out/MosA1_WholeBrainA/target64/bert/raw/bert_docs.txt'  


output_filename_source_label = 'scLLMDA/out/MosA1_WholeBrainA/source64/bert/raw/bert_labels.txt'  
output_filename_target_label = 'scLLMDA/out/MosA1_WholeBrainA/target64/bert/raw/bert_labels.txt'  






sample_ratio = 0.1
edge_ratio = 0.0
save_unknown = False
save_rare = False
no_smote = False


# load and preprocess data
dataset, adata, le, train_shape, test_shape, label_mapping = load_ATAC_dataset(data_dir, train_name_list, test_name, sample_ratio, edge_ratio, save_path, save_unknown, save_rare, no_smote)



print(adata)
print(adata.obs['CellType'])






def convert_edgelist(input_filename, output_filename_source,output_filename_target):
    directory = os.path.dirname(output_filename_source)


    os.makedirs(directory, exist_ok=True)
    directory = os.path.dirname(output_filename_target)


    os.makedirs(directory, exist_ok=True)
    
    
    with open(input_filename, 'r') as infile:
        lines = infile.readlines()

    with open(output_filename_source, 'w') as outfile_source:
        with open(output_filename_target, 'w') as outfile_target:
    
            for line in lines:
                
                numbers = line.split(' ')
                
                
                if int(numbers[0]) < train_shape:
                    new_line = line.strip().replace(' ', ',') + '\n'
                    outfile_source.write(new_line)
                else:
                    modified_numbers = [str(int(num) - train_shape) for num in numbers]
                    outfile_target.write(",".join(modified_numbers) + '\n')
    
        

convert_edgelist(input_filename_edge, output_filename_source_edge,output_filename_target_edge)




numpy_array = dataset.graph['node_feat'].cpu().numpy()

def convert_docs(output_filename_source,output_filename_target):
    directory = os.path.dirname(output_filename_source)


    os.makedirs(directory, exist_ok=True)
    directory = os.path.dirname(output_filename_target)


    os.makedirs(directory, exist_ok=True)

    with open(output_filename_source, 'w') as outfile_source:
        with open(output_filename_target, 'w') as outfile_target:
    
            i = 0            
            for row in numpy_array:
        
                if i < train_shape:
                    outfile_source.write(','.join(map(str, row)) + '\n')
                    i += 1
                else:
                    outfile_target.write(','.join(map(str, row)) + '\n')
                    i += 1
                        
convert_docs(output_filename_source_feature,output_filename_target_feature)


data = dataset.label.cpu()
def convert_labels(output_filename_source,output_filename_target):
    numpy_array = data.numpy().astype(int)
    directory = os.path.dirname(output_filename_source)


    os.makedirs(directory, exist_ok=True)
    directory = os.path.dirname(output_filename_target)


    os.makedirs(directory, exist_ok=True)

    with open(output_filename_source, 'w') as outfile_source:
        with open(output_filename_target, 'w') as outfile_target:
    
            i = 0            
            for row in numpy_array:
                #print(row)
        
                if i < train_shape:
                    outfile_source.write(str(row) + '\n')
                    i += 1
                else:
                    outfile_target.write(str(row) + '\n')
                    i += 1
                        

convert_labels(output_filename_source_label,output_filename_target_label)




