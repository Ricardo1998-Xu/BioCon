# coding=utf-8
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from argparse import ArgumentParser
from dual_gnn.cached_gcn_conv import CachedGCNConv    
from dual_gnn.dataset.DomainData import DomainData
from dual_gnn.ppmi_conv import PPMIConv
import random
import numpy as np
import torch
import torch.functional as F
from torch import nn
import torch.nn.functional as F
import itertools
from sklearn.metrics import f1_score
from torch.cuda.amp import GradScaler 
import scanpy as sc
import anndata
import pickle
from anndata import AnnData



# 计算宏平均F1分数
def compute_macro_f1(preds, labels):
    return f1_score(labels.cpu(), preds.cpu(), average='macro')  


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = ArgumentParser()
parser.add_argument("--source", type=str, default='bert')                
parser.add_argument("--target", type=str, default='bert')
parser.add_argument("--name", type=str, default='gda')
parser.add_argument("--seed", type=int,default=500)
parser.add_argument("--gda", type=bool,default=True)
parser.add_argument("--encoder_dim", type=int, default=16)
parser.add_argument("--path_len", type=int, default=3)
parser.add_argument("--lr", type=float, default=3e-3)
parser.add_argument("--patience", type=int, default=200)
parser.add_argument("--weight", type=int, default=1)
parser.add_argument("--train_name", type=str, default='MosA1')
parser.add_argument("--test_name", type=str, default='WholeBrainA')

args = parser.parse_args()
seed = args.seed
use_gda = args.gda                    
encoder_dim = args.encoder_dim                 
path_len = args.path_len
lr = args.lr
epochs = 200
patience = args.patience
weight = args.weight


if args.train_name in ['MosP1', 'MosA1', 'MosM1']:
    epoch_end =  55
elif args.train_name == 'MouseBrain(10x)':
    epoch_end = 122
elif args.train_name in ['WholeBrainA', 'WholeBrainB']:
    # 处理唯一例外
    if args.train_name == 'WholeBrainB' and args.test_name == 'MosM1':
        epoch_end = 138
    epoch_end = 22





id = "source: {}, target: {}, seed: {}, gda: {}, encoder_dim: {}"\
    .format(args.source, args.target, seed, use_gda,  encoder_dim)          

#print(id)

rate = 0.0

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



classes = None
dataset = DomainData("scLLMDA/out/MosA1_WholeBrainA/source64/{}".format(args.source), name=args.source)        
source_data = dataset[0]
#print(source_data)
#print(type(source_data))
#print(dataset.num_features)
#dataset

source_classes = dataset.num_classes
#print(dataset.num_classes)


dataset = DomainData("scLLMDA/out/MosA1_WholeBrainA/target64/{}".format(args.target), name=args.target)


target_classes = dataset.num_classes
print(dataset.num_classes)
if target_classes > source_classes:
       classes = target_classes
else :
    classes = source_classes


target_data = dataset[0]


source_data = source_data.to(device)        
target_data = target_data.to(device)        


class GNN(torch.nn.Module):                                        
    def __init__(self, base_model=None, type="gcn", **kwargs):          
        super(GNN, self).__init__()

        if base_model is None:                                          
            weights = [None, None]
            biases = [None, None]
            
        else:                                                               
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]


        self.dropout_layers = [nn.Dropout(0.1) for _ in weights]               
        self.type = type                                                    

        model_cls = PPMIConv if type == "ppmi" else CachedGCNConv                  

        self.conv_layers = nn.ModuleList([                                  
            model_cls(dataset.num_features, 100,                                
                     weight=weights[0],                                             
                     bias=biases[0],
                      **kwargs),
            model_cls(100, encoder_dim,
                     weight=weights[1],
                     bias=biases[1],
                      **kwargs)
        ])

    def forward(self, x, edge_index, cache_name):                                   
        for i, conv_layer in enumerate(self.conv_layers):               
            x = conv_layer(x, edge_index, cache_name)                      
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
        return x


class GradReverse(torch.autograd.Function):                         
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * rate
        return grad_output, None


class GRL(nn.Module):                                                   
    def forward(self, input):
        return GradReverse.apply(input)


loss_func = nn.CrossEntropyLoss().to(device)            


encoder = GNN(type="gcn").to(device)                        
if use_gda:                                                  
    ppmi_encoder = GNN(base_model=encoder, type="ppmi", path_len = path_len).to(device)




cls_model = nn.Sequential(
    nn.Linear(encoder_dim, classes),                    
).to(device)


domain_model = nn.Sequential(                                           
    GRL(),                                                  
    nn.Linear(encoder_dim, 40), 
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(40, 2),
).to(device)







class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query_proj = nn.Linear(in_channels, in_channels)
        self.key_proj = nn.Linear(in_channels, in_channels)
        self.value_proj = nn.Linear(in_channels, in_channels)

    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)    
        Q = self.query_proj(stacked)           
        K = self.key_proj(stacked)             
        V = self.value_proj(stacked)           

        attention_scores = (Q @ K.transpose(-2, -1)) / (Q.size(-1) ** 0.5)  
        attention_weights = F.softmax(attention_scores, dim=-1)

        attended = attention_weights @ V
        outputs = attended.sum(dim=1)  
        return outputs




        

att_model = Attention(encoder_dim).cuda()                                       



models = [encoder, cls_model, domain_model]                                        
if use_gda:                                                             
    models.extend([ppmi_encoder, att_model])

params = itertools.chain(*[model.parameters() for model in models])                                 
optimizer = torch.optim.Adam(params, lr=lr)
                                   


def gcn_encode(data, cache_name, mask=None):
    encoded_output = encoder(data.x, data.edge_index, cache_name)           
    #print(encoded_output.shape)
    if mask is not None:
        encoded_output = encoded_output[mask.to(torch.bool)]
    return encoded_output


def ppmi_encode(data, cache_name, mask=None):
    encoded_output = ppmi_encoder(data.x, data.edge_index, cache_name)
    if mask is not None:
        encoded_output = encoded_output[mask.to(torch.bool)]
    return encoded_output


def encode(data, cache_name, mask=None):                    
    gcn_output = gcn_encode(data, cache_name, mask)
    if use_gda:
        ppmi_output = ppmi_encode(data, cache_name, mask)
        outputs = att_model([gcn_output, ppmi_output])
        return outputs
    else:
        return gcn_output

def predict(data, cache_name, mask=None):
    encoded_output = encode(data, cache_name, mask)
    #print('encoded_output')
    #print(encoded_output.shape)                
    logits = cls_model(encoded_output) 
    #print(logits)
    #print(type(logits))
    #print(logits.shape)
    return logits                           


def evaluate(preds, labels):
    corrects = preds.eq(labels)
    accuracy = corrects.float().mean()
    macro_f1 = compute_macro_f1(preds, labels)
    return accuracy, macro_f1

pre_source = None
pre_test = None

def test(data, cache_name, mask=None):
    for model in models:
        model.eval()
    logits = predict(data, cache_name, mask)
    preds = logits.argmax(dim=1)                
    global pre_source,pre_test
    if cache_name == 'source':
        pre_source = preds
    else:
        pre_test = preds
    #print("掩码")
    labels = data.y if mask is None else data.y[mask.to(torch.bool)]
    #print(labels.shape)
    accuracy, macro_f1 = evaluate(preds, labels)

    return accuracy, macro_f1


embedding_train = None
embedding_test = None


def train(epoch):
    for model in models:                                        
        model.train()
    optimizer.zero_grad()                               

    global rate                                             
    
    
    rate = min((epoch + 1) / epochs, 0.05)                         
    #rate = 0.05

    #一共生成四种embedding，再进行统一
    encoded_source = encode(source_data, "source")                      
    encoded_target = encode(target_data, "target")                          
    
    global embedding_train, embedding_test
    embedding_train = encoded_source
    embedding_test = encoded_target

    #print('encoded_target')
    #print(encoded_target.shape)
    source_logits = cls_model(encoded_source)                                   


    # use source classifier loss:
    cls_loss = loss_func(source_logits, source_data.y)          
    #print("Epoch: {}, loss: {}".format(epoch,cls_loss))                  

    for model in models:
        for name, param in model.named_parameters():                                
            if "weight" in name:                                                
                cls_loss = cls_loss                                
    

    if use_gda:
        # use domain classifier loss:
        source_domain_preds = domain_model(encoded_source)
        target_domain_preds = domain_model(encoded_target)

        source_domain_cls_loss = loss_func(                                                 
            source_domain_preds,                                    
            torch.zeros(source_domain_preds.size(0)).type(torch.LongTensor).to(device)
        )
        target_domain_cls_loss = loss_func(
            target_domain_preds,
            torch.ones(target_domain_preds.size(0)).type(torch.LongTensor).to(device)
        )
        loss_grl = source_domain_cls_loss + target_domain_cls_loss
        loss = cls_loss + weight * loss_grl                                          
        print("epoch:{}, loss: {}".format(epoch,loss))     
        

    else:
        loss = cls_loss
        print("loss: {}".format(loss)) 

    
    optimizer.zero_grad()                                       
    loss.backward()                                                 
    optimizer.step()                                        

    return loss
    


best_source_acc = 0.0
best_target_acc = 0.0
best_sorce_F1 = 0.0
best_target_F1 = 0.0
best_epoch = 0.0
best_path_len = path_len
best_lr = lr


embedding_train_best = None
embedding_test_best = None                                 
pre_train_best = None
pre_test_best = None 

min_loss = 1e10
patience_cnt = 0

for epoch in range(1, epochs):                 
    loss = train(epoch)
    #source_correct , source_macro_f1= test(source_data, "source", source_data.test_mask)
    source_correct , source_macro_f1= test(source_data, "source")
    target_correct , target_macro_f1= test(target_data, "target")
    #print
    print("source_acc: {} ,source_marco_F1: {} ".format(source_correct, source_macro_f1))
    
    
    if epoch == epoch_end:
        best_target_acc = target_correct
        best_source_acc = source_correct
        best_target_F1 = target_macro_f1
        best_source_F1 = source_macro_f1
        best_epoch = epoch
        embedding_train_best = embedding_train
        #print(embedding_train_best.shape)
        #print(type(embedding_train_best))
        embedding_test_best = embedding_test
        #print(embedding_test_best.shape)
        #print(type(embedding_test_best))
        pre_train_best = pre_source
        pre_test_best = pre_test
        #print(pre_train_best.shape)
        #print(pre_test_best.shape)
        break
       

                    
        
print("=============================================================")
line = "{} - Epoch: {} , path_len:{}, lr:{}, \nbest_source_acc: {}, best_target_acc: {} , best_source_marco_F1: {}, best_target_marco_F1: {}\n"\
    .format(id, best_epoch, best_path_len, best_lr, best_source_acc, best_target_acc,best_source_F1,best_target_F1)




print(line)



