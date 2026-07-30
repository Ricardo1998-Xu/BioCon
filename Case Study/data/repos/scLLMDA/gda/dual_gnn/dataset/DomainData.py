
import os.path as osp
import torch
import numpy as np
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)
from torch_geometric.io import read_txt_array
from torch_geometric.utils import remove_self_loops
from torch_geometric.data import Data, DataLoader
import random
import os

seed = 200
if float(torch.version.cuda) >= 10.2:                           #设置seed
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


class DomainData(InMemoryDataset):
    r"""The protein-protein interaction networks from the `"Predicting
    Multicellular Function through Multi-layer Tissue Networks"
    <https://arxiv.org/abs/1707.04638>`_ paper, containing positional gene
    sets, motif gene sets and immunological signatures as features (50 in
    total) and gene ontology sets as labels (121 in total).

    Args:
        root (string): Root directory where the dataset should be saved.
        split (string): If :obj:`"train"`, loads the training dataset.
            If :obj:`"val"`, loads the validation dataset.
            If :obj:`"test"`, loads the test dataset. (default: :obj:`"train"`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """
    def __init__(self,
                 root,
                 name,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.name = name
        #self.root = root
        super(DomainData, self).__init__(root, transform, pre_transform, pre_filter)        #做了很多事情，初始化了，下面的三个函数全部使用到了,把txt变成了pt的形式，  生成了data.pt
        #print(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])                            #Data(x=[5088, 64], edge_index=[2, 22875], y=[5088], train_mask=[5088], val_mask=[5088], test_mask=[5088])
        #print(self.data)                                                                       #mask就是用train valid test    ，  self.data与主函数的dataset[0]对应   ，又是内置函数实现的，很夸张
        #print(type(self.data))                                                                 
        #print(self.slices)                                                                         #self.slices为空
        #print(type(self.slices))


    @property
    def raw_file_names(self):
        return ["docs.txt", "edgelist.txt", "labels.txt"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process_test(self):
        #print("used")
        edge_path = osp.join(self.raw_dir, '{}_edgelist.txt'.format(self.name))             #获取/home/guansheng/code/SANGO-main copy/UDAGCN_data_Large/source/再加上acm ,edgelist.txt
        edge_index = read_txt_array(edge_path, sep=',', dtype=torch.long).t()
        #print(edge_index[0:10])                  
        #print(type(edge_index))
        #print(edge_index.shape)                            #[2, 22875]

        docs_path = osp.join(self.raw_dir, '{}_docs.txt'.format(self.name))                     #获取feature的地址
        f = open(docs_path, 'rb')                                                                   #rb，二进制打开方式
        content_list = []
        for line in f.readlines():
            line = str(line, encoding="utf-8")
            content_list.append(line.split(","))                                                #把所有的数据读取进来
        #print(content_list[0:10])
        #print(len(content_list))
        x = np.array(content_list, dtype=float)                 #feature转化为numpy
        x = torch.from_numpy(x).to(torch.float)                     #feature转化为tensor
        #print(x.shape)                                     #[12240, 64]
        #print(x[0:10])                                     

        label_path = osp.join(self.raw_dir, '{}_labels.txt'.format(self.name))      #读取标签txt
        f = open(label_path, 'rb')
        content_list = []
        for line in f.readlines():
            line = str(line, encoding="utf-8")
            line = line.replace("\r", "").replace("\n", "")
            content_list.append(line)
        y = np.array(content_list, dtype=int)
        y = torch.from_numpy(y).to(torch.int64)                                         #获取标签的的tensor
        #print(y.shape)                 #tensor([2, 1, 4, 2, 3, 0, 3, 2, 0, 0])
        #print(y[0:10])

        data_list = []
        data = Data(edge_index=edge_index, x=x, y=y)                    #生成 edge , feature , label
        #print(data)                                             #Data(x=[12240, 64], edge_index=[2, 53199], y=[12240])

        #用来生成训练，验证，测试的数据的
        random_node_indices = np.random.permutation(y.shape[0])             #随机打乱12240个数，然后 0.7训练集  0.1验证集   0.2测试集
        #print(random_node_indices)
        #print(type(random_node_indices))
        #print(random_node_indices.shape)                       #(12240,)
        training_size = int(len(random_node_indices) * 0.7)
        val_size = int(len(random_node_indices) * 0.1)
        train_node_indices = random_node_indices[:training_size]                #取随机总数(12240)的0.7
        val_node_indices = random_node_indices[training_size:training_size + val_size]          #取随机总数(12240)的0.1
        test_node_indices = random_node_indices[training_size + val_size:]          #0.2

        #print(y.shape[0])                          #12240
        train_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
        train_masks[train_node_indices] = 1                             #一个12240的tensor train的掩码置为1
        #print(train_masks)
        #print(type(train_masks))
        #print(train_masks.shape)                       #torch.Size([12240])    
        #print(train_node_indices)
        #print(type(train_node_indices))
        #print(train_node_indices.shape)            #(8568,)
        val_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
        val_masks[val_node_indices] = 1
        test_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
        test_masks[test_node_indices] = 1

        data.train_mask = train_masks                           #训练的掩码
        data.val_mask = val_masks
        data.test_mask = test_masks


        #print(self.pre_transform)                  #None
        if self.pre_transform is not None:                  #无用
            data = self.pre_transform(data)

        data_list.append(data)                                      #上面定义过了
        
        data, slices = self.collate([data])                 ##正好对应self.data和main函数的dataset[0]

        #print(self.processed_paths[0])
        #torch.save((data, slices), self.processed_paths[0])            #用来存储data.pt的数据的

    
    
    def process(self):
        edge_path = osp.join(self.raw_dir, '{}_edgelist.txt'.format(self.name))             #获取/home/guansheng/code/SANGO-main copy/UDAGCN_data_Large/source/再加上acm ,edgelist.txt
        edge_index = read_txt_array(edge_path, sep=',', dtype=torch.long).t()

        docs_path = osp.join(self.raw_dir, '{}_docs.txt'.format(self.name))
        f = open(docs_path, 'rb')
        content_list = []
        for line in f.readlines():
            line = str(line, encoding="utf-8")
            content_list.append(line.split(","))              
        x = np.array(content_list, dtype=float)
        x = torch.from_numpy(x).to(torch.float)

        label_path = osp.join(self.raw_dir, '{}_labels.txt'.format(self.name))
        f = open(label_path, 'rb')
        content_list = []
        for line in f.readlines():
            line = str(line, encoding="utf-8")
            line = line.replace("\r", "").replace("\n", "")
            content_list.append(line)
        y = np.array(content_list, dtype=int)
        y = torch.from_numpy(y).to(torch.int64)

        data_list = []
        data = Data(edge_index=edge_index, x=x, y=y)

        random_node_indices = np.random.permutation(y.shape[0])
        training_size = int(len(random_node_indices) * 0.7)
        val_size = int(len(random_node_indices) * 0.1)
        train_node_indices = random_node_indices[:training_size]
        val_node_indices = random_node_indices[training_size:training_size + val_size]
        test_node_indices = random_node_indices[training_size + val_size:]

        train_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
        train_masks[train_node_indices] = 1
        val_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
        val_masks[val_node_indices] = 1
        test_masks = torch.zeros([y.shape[0]], dtype=torch.uint8)
        test_masks[test_node_indices] = 1

        data.train_mask = train_masks
        data.val_mask = val_masks
        data.test_mask = test_masks


        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data_list.append(data)

        data, slices = self.collate([data])                 

        #print(self.processed_paths[0])
        torch.save((data, slices), self.processed_paths[0])
