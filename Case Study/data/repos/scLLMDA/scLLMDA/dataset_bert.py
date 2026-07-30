import numpy as np
import h5py
from torch.utils.data import Dataset
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix
import logging
logger = logging.getLogger(__name__)

def load_adata(data) -> AnnData:
    '''
    load data as AnnData

    data: path to the input h5ad file

    return: AnnData
    '''
    adata = sc.read_h5ad(data)
    if adata.X.max() > 1:
        logger.info("binarized")
        adata.X.data = (adata.X.data > 0).astype(np.float32)
    return adata

class SingleCellDataset(Dataset):
    '''
    preprocess data and make dataset
    data: AnnData
    genome: reference genome
    seq_len: length to extend/trim sequences to
    return: dataset
    '''
    def __init__(self, data: AnnData, genome, seq_len=1344):
        # gene need to be accessible in 1% cells
        sc.pp.filter_genes(data, min_cells=int(round(0.01 * data.shape[0])))
        self.data = data
        self.seq_len = seq_len
        # load genome
        self.genome = h5py.File(genome, 'r')
        self.obs = self.data.obs.copy()
        del self.data.obs
        self.var = self.data.var.copy()
        del self.data.var
        self.X = csr_matrix(self.data.X.T)
        del self.data.X
        if "chr" in self.var.keys():
            self.chroms = self.var["chr"]
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        # Retrieve sequence with a center length of seq_len from peak.
        chrom, start, end = self.var["chr"][index], self.var["start"][index], self.var["end"][index]
        mid = (int(start) + int(end)) // 2
        left, right = mid - self.seq_len//2, mid + self.seq_len//2
        left_pad, right_pad = 0, 0
        if left < 0:
            left_pad = -left_pad
            left = 0
        if right > self.genome[chrom].shape[0]:
            right_pad = right - self.genome[chrom].shape[0]
            right = self.genome[chrom].shape[0]
        seq = self.genome[chrom][left:right]
        # imputation
        if len(seq) < self.seq_len:
            seq = np.concatenate((
                np.full(left_pad, -1, dtype=seq.dtype),
                seq,
                np.full(right_pad, -1, dtype=seq.dtype),
            ))
        return seq, self.X[index].toarray().flatten()
import torch
from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.configuration_bert import BertConfig
import h5py
import numpy as np
import scanpy as sc
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix


import os
import numpy as np
import torch

from scipy.sparse import csr_matrix
import h5py
import scanpy as sc

class saveSingleCellDataset(Dataset):
    '''
    preprocess data and make dataset

    data: AnnData
    genome: reference genome
    seq_len: length to extend/trim sequences to

    return: dataset
    '''
    def __init__(self, data: AnnData, genome, seq_len=1344, save_file='embeddings_cell_data.npz'):
        
        sc.pp.filter_genes(data, min_cells=int(round(0.01 * data.shape[0])))
        print(data)
        self.data = data
        self.seq_len = seq_len
        
        self.genome = h5py.File(genome, 'r')
        self.obs = self.data.obs.copy()
        del self.data.obs
        self.var = self.data.var.copy()
        del self.data.var
        self.X = csr_matrix(self.data.X.T) 
        del self.data.X

        if "chr" in self.var.keys():
            self.chroms = self.var["chr"]

        
        self.int_to_base = {0: 'N', 1: 'A', 2: 'C', 3: 'G', 4: 'T', -1: 'N'}

        
        self.tokenizer = AutoTokenizer.from_pretrained("DNABERT-2-117M", trust_remote_code=True)
        config = BertConfig.from_pretrained("DNABERT-2-117M")
        self.model = AutoModel.from_pretrained("DNABERT-2-117M", trust_remote_code=True, config=config)

       
        self.save_file = save_file

       
        self.embedding_mean = []
        self.embedding_max = []
        self.combine_embedding = []
        self.all_cell_data = []

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        
        '''
        chrom, start, end = self.var["chr"][index], self.var["start"][index], self.var["end"][index]
        mid = (int(start) + int(end)) // 2
        left, right = mid - self.seq_len // 2, mid + self.seq_len // 2
        left_pad, right_pad = 0, 0
        if left < 0:
            left_pad = -left_pad
            left = 0
        if right > self.genome[chrom].shape[0]:
            right_pad = right - self.genome[chrom].shape[0]
            right = self.genome[chrom].shape[0]
        seq = self.genome[chrom][left:right]

        
        if len(seq) < self.seq_len:
            seq = np.concatenate((
                np.full(left_pad, -1, dtype=seq.dtype),
                seq,
                np.full(right_pad, -1, dtype=seq.dtype),
            ))

        
        dna_sequence = ''.join([self.int_to_base.get(int(i), 'N') for i in seq])
        #print(dna_sequence)
        #print(len(dna_sequence))
        '''
        
        chrom, start, end = self.var["chr"][index], self.var["start"][index], self.var["end"][index]
        seq = self.genome[chrom][int(start):int(end)]
        dna_sequence = ''.join([self.int_to_base.get(int(i), 'N') for i in seq])
        
        #print(dna_sequence)
        #print(len(dna_sequence))
        


        
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        inputs = self.tokenizer(dna_sequence, return_tensors='pt')["input_ids"].to(device)
       
        with torch.no_grad():  
            hidden_states = self.model(inputs)[0]  
        
        embedding_mean = torch.mean(hidden_states[0], dim=0).cpu().numpy()          
                                                        
        embedding_max = torch.max(hidden_states[0], dim=0)[0].cpu().numpy()
        
        commbine_embedding = np.concatenate((embedding_mean, embedding_max), axis=0)


        
        cell_data = self.X[index].toarray().flatten()

        
        self.embedding_mean.append(embedding_mean)
        self.embedding_max.append(embedding_max)
        self.combine_embedding.append(commbine_embedding)
        self.all_cell_data.append(cell_data)

        return embedding_mean, embedding_max, commbine_embedding, cell_data
        
    def save_all_data(self,save_path=None):
        if save_path is None:
            save_path = self.save_file
        
        np.savez(save_path, embedding_mean=np.array(self.embedding_mean), embedding_max=np.array(self.embedding_max), combine_embedding=np.array(self.combine_embedding), cell_data=np.array(self.all_cell_data))
        print(f"所有的 embedding 和 cell_data 已保存到 {save_path}")


import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Sequence, Tuple, List, Union

import torch
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Dataset
import h5py
from scipy.sparse import csr_matrix
import scanpy as sc
from anndata import AnnData
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps"),
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)

import os
import json
import logging
from typing import List
from scipy.sparse import csr_matrix
import h5py

# Function to generate k-mer string from a DNA sequence
import os
import json
import logging
from typing import List
from scipy.sparse import csr_matrix

import h5py

# Function to generate k-mer string from a DNA sequence
import os
import json
import logging
from typing import List
from scipy.sparse import csr_matrix
import torch
import h5py

# Function to generate k-mer string from a DNA sequence
import os
import json
import logging
import h5py
import torch
import scanpy as sc
from typing import List
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset

def generate_kmer_str(sequence: str, k: int) -> str: 
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i + k] for i in range(len(sequence) - k + 1)])

def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
    return kmer

def get_sequence(var, genome, int_to_base, seq_len, index):
    """
    Retrieve a sequence from the genome based on the given index.
    
    Parameters:
    - var: Contains chromosomal and positional information
    - genome: Reference genome data
    - int_to_base: Dictionary to convert integer representation of bases to DNA bases
    - seq_len: Length to extend/trim sequences to
    - index: Index of the sequence to retrieve

    Returns:
    - dna_sequence: The DNA sequence corresponding to the given index
    """
    chrom = var["chr"][index]
    start = var["start"][index]
    end = var["end"][index]
    mid = (int(start) + int(end)) // 2
    left = mid - seq_len // 2
    right = mid + seq_len // 2

    left_pad, right_pad = 0, 0
    if left < 0:
        left_pad = -left
        left = 0
    if right > genome[chrom].shape[0]:
        right_pad = right - genome[chrom].shape[0]
        right = genome[chrom].shape[0]

    seq = genome[chrom][left:right]
    # Convert numerical representation of genome to DNA bases (e.g., ATCG)
    dna_sequence = ''.join([int_to_base.get(int(i), 'N') for i in seq])

    return dna_sequence

class fineSingleCellDataset(Dataset):
    '''
    Preprocess data and make dataset
    data: AnnData object containing single-cell data
    genome: Reference genome file path
    seq_len: Length to extend/trim sequences to
    '''

    def __init__(self, data, genome, seq_len=1344, kmer=5, data_path="", tokenizer=None):
        import anndata
        # Filter genes accessible in at least 1% of cells
        sc.pp.filter_genes(data, min_cells=int(round(0.01 * data.shape[0])))

        self.int_to_base = {0: 'N', 1: 'A', 2: 'C', 3: 'G', 4: 'T'}
        self.data = data
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data_path = data_path
        self.kmer = kmer

        # Load genome
        self.genome = h5py.File(genome, 'r')
        self.obs = self.data.obs.copy()
        del self.data.obs
        self.var = self.data.var.copy()
        del self.data.var
        self.X = csr_matrix(self.data.X.T)  # Observed data, used as labels
        del self.data.X
        if "chr" in self.var.keys():
            self.chroms = self.var["chr"]
        peak_number = self.X.shape[0]
        print(f"Peak number: {peak_number}")

        # Generate all sequences
        self.all_seq = [get_sequence(self.var, self.genome, self.int_to_base, self.seq_len, i) for i in range(peak_number)]
        # Generate or load k-mer representations
        self.kmers = load_or_generate_kmer(self.data_path, self.all_seq, self.kmer)
        

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        text = self.kmers[index]
        # Tokenize the generated k-mer string
        output = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=self.seq_len,
            truncation=True,
        )
        input_ids = output["input_ids"].squeeze(0)  # Remove batch dimension
        label = self.X[index].toarray().flatten()
        #return {'input_ids': input_ids, 'labels': torch.tensor(label, dtype=torch.long)}
        #return input_ids, torch.tensor(label, dtype=torch.float32)
        return input_ids, label




