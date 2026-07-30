import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import scanpy as sc
import random
import os


gene_embedding = "scLLMDA/out/MosA1_WholeBrainA/embeddings_genome.npz"
np_save = "scLLMDA/out/MosA1_WholeBrainA/embeddings_cell.npy"
process_data = "scLLMDA/data/scATAC/MosA1_WholeBrainA.h5ad"
adata_write = "scLLMDA/out/MosA1_WholeBrainA/embeddings_cell.h5ad"
                
seed = 42

if float(torch.version.cuda) >= 10.2:                           
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
random.seed(seed)               
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)



class EmbeddingCellDataset(Dataset):
    def __init__(self, embeddings, cell_data):
        self.embeddings = embeddings
        self.cell_data = cell_data
    def __len__(self):
        return len(self.embeddings)
    def __getitem__(self, index):
        return self.embeddings[index], self.cell_data[index]
    

class RegressionModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[512, 256, 128], dropout=0.3, use_reg_cell=False):
        super(RegressionModel, self).__init__()
        self.use_reg_cell = use_reg_cell

        layers = []
        in_features = input_size
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = hidden

        self.backbone = nn.Sequential(*layers)
        self.cell_embedding = nn.Linear(hidden_sizes[-1], output_size)

    def get_embedding(self):
        return self.cell_embedding.state_dict()["weight"]

    def forward(self, x):
        x = self.backbone(x)
        x = self.cell_embedding(x)
        if self.use_reg_cell:
            reg = torch.norm(self.cell_embedding.weight, p=2)
        else:
            reg = None
        return x, reg

    




@torch.no_grad()
def test_model(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    for embeddings, cell_data in tqdm(loader, desc="Validating"):
        embeddings = embeddings.to(device)
        cell_data = cell_data.to(device)

       
        output = model(embeddings)
        #print(output[0].shape)
        #print(output[1].shape)
        output = torch.sigmoid(output[0])  
        output = output.cpu().numpy()
        cell_data = cell_data.cpu().numpy()

        all_preds.append(output)
        all_labels.append(cell_data)

    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    auc_scores = []
    for i in range(all_labels.shape[1]):
        auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
        auc_scores.append(auc)
    avg_auc = np.mean(auc_scores)
    return avg_auc

def sango_test_model(model, loader):
    
    #compute AUC score on the loader
    
    model.eval()
    all_label = list()
    all_pred = list()

    for it, (seq, target) in enumerate(tqdm(loader)):
        seq = seq.to(device)
        output = model(seq)  
        output = output.detach()
        output = torch.sigmoid(output).cpu().numpy()
        target = target.numpy().astype(np.int8)
        all_pred.append(output)
        all_label.append(target)

    all_pred = np.concatenate(all_pred, axis=0) # (n_peaks, n_cells)
    all_label = np.concatenate(all_label, axis=0)
    val_auc = list()
    test_inds = range(all_pred.shape[0])
    for i in tqdm(test_inds, desc="Calculating AP"):
        val_auc.append(roc_auc_score(all_label[i], all_pred[i]))
    val_auc = np.array(val_auc)
    return val_auc

saved_data = np.load(gene_embedding)
embeddings = saved_data['embedding_mean']
cell_data = saved_data['cell_data']
reg_weights = 0         
use_reg_cell=True         


dataset = EmbeddingCellDataset(embeddings, cell_data)


train_size = len(dataset) - 2000
val_size = 2000
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


input_size = embeddings.shape[1]  
output_size = cell_data.shape[1] 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RegressionModel(input_size=input_size, output_size=output_size, use_reg_cell=use_reg_cell).to(device)


criterion = nn.BCEWithLogitsLoss()  
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 400
patience = 15
best_score =0.0
wait = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for embeddings, cell_data in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        embeddings = embeddings.to(device)
        cell_data = cell_data.to(device)
        
        outputs = model(embeddings)
        
        if use_reg_cell:
            loss = criterion(outputs[0], cell_data) + reg_weights * outputs[1]
        
        loss = criterion(outputs[0], cell_data)  

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
 
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

   

embedding = model.get_embedding().detach().cpu().numpy()


np.save(np_save, embedding)  
print("模型的权重已保存为 embedding")

print("训练完成。")





file_path = np_save
data_embedding = np.load(file_path)


print(data_embedding)
print("Shape:", data_embedding.shape)  
print("Data type:", data_embedding.dtype)  



data=sc.read_h5ad(process_data)
adata = sc.AnnData(
        data_embedding,
        obs=data.obs,
    )
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)
adata.write_h5ad(adata_write)
