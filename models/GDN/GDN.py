import numpy as np
import torch
import pickle
from hydra.utils import get_original_cwd, to_absolute_path
import matplotlib.pyplot as plt
import random
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import time
from .datasets.TimeDataset import TimeDataset

import math
import torch.nn.functional as F

from .util.env import get_device, set_device
from .util.preprocess import adjList2_edgeIndex, construct_data
from .util.net_struct import get_feature_map, get_fc_adj_list
from .util.iostream import printsep

import pandas as pd
from .graph_layer import GraphLayer
import sys
from ..normal_model import normal_model

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()

def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    # (2, batch_size*edge_num)
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()
    # print("batch_edge_index",batch_edge_index)
    # print("batch_edge_index.shape",batch_edge_index.shape)
    # print("org_edge_index.shape",org_edge_index.shape)

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()


class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num = 512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num-1:
                modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear( layer_in_num, inter_num ))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out



class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()
        print("in_channel",in_channel,"out_channel",out_channel,"inter_dim",inter_dim,"heads",heads,"node_num",node_num)


        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):

        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
        out = self.bn(out)
        return self.relu(out)


class GDN(normal_model):
    def __init__(self, 
    cfg,
    input_feature_size, 
    window_labels=None
    ):

        super().__init__( cfg,"GDN", input_feature_size, window_labels)
        print("cfg",cfg)

        embed_dim = self.cfg.model.sensor_embed_dim
        out_layer_inter_dim=self.cfg.model.out_layer_inter_dim
        out_layer_num= self.cfg.model.out_layer_num
        topk=self.cfg.model.topk
        window_size = self.cfg.model.window_size
        
        print("input_feature_size",input_feature_size, "embed_dim",embed_dim,"out_layer_inter_dim",out_layer_inter_dim, "out_layer_dim",out_layer_num,"topk",topk)
        edge_set_num = 1
        print("edge_set_num",edge_set_num)

        self.window_size = window_size
        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None
        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.embedding = nn.Embedding(input_feature_size, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(window_size, embed_dim, inter_dim=2*embed_dim, heads=1) for i in range(edge_set_num)
        ])
        self.out_layer = OutLayer(embed_dim*edge_set_num, input_feature_size, out_layer_num, inter_num = out_layer_inter_dim)
        self.dp = nn.Dropout(0.2)
        self.count=0



        self.init_params()
    def saveModel(self):
        torch.save(self.state_dict(), get_original_cwd()+"/model/" +
                   self.datasetName+"/"+self.modelName+"/model.pth")
    
    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def Construct_fullConnected_edge_index(self,startIndex,numOfNode,adjK):

        node_i = []
        node_j = []
        fullRange = 51

        for i in range(numOfNode):
            node_i.extend([startIndex+i for x in range(adjK)])
            node_j.extend([(startIndex+j)% fullRange for j in range(adjK)])
            

        return node_i , node_j
    def construct_edge_index(self):
        total_node_i = []
        total_node_j = []
        adjK = 15
        node_i,node_j = self.Construct_fullConnected_edge_index(0,5,adjK)
        total_node_i.extend(node_i)
        total_node_j.extend(node_j)
        node_i,node_j = self.Construct_fullConnected_edge_index(5,11,adjK)
        total_node_i.extend(node_i)
        total_node_j.extend(node_j)
        node_i,node_j = self.Construct_fullConnected_edge_index(16,9,adjK)
        total_node_i.extend(node_i)
        total_node_j.extend(node_j)
        node_i,node_j = self.Construct_fullConnected_edge_index(25,9,adjK)
        total_node_i.extend(node_i)
        total_node_j.extend(node_j)
        node_i,node_j = self.Construct_fullConnected_edge_index(34,13,adjK)
        total_node_i.extend(node_i)
        total_node_j.extend(node_j)
        node_i,node_j = self.Construct_fullConnected_edge_index(47,4,adjK)
        total_node_i.extend(node_i)
        total_node_j.extend(node_j)
        # print("total_node_i",total_node_i)
        # print("total_node_j",total_node_j)
        edge_index = np.array([total_node_j,total_node_i])
        print("edge_index.shape",edge_index.shape)
        return edge_index


    def saveEdge_index(self):
        savePath = get_original_cwd()+"/result/" + self.datasetName+"/edge_index.txt"
        edge_index = self.gated_edge_index.detach().cpu().numpy()
        # print("edge_index",edge_index)
        # edge_index = self.construct_edge_index()
        print("edge_index",edge_index)
        with open(savePath,"wb") as f:
            pickle.dump(edge_index,f)

    def testing_all(self, dataset: pd.DataFrame) -> pd.DataFrame:
        self.testPreprocessing()
        self.dataset_window_index = dataset.index
        self.dataset_window_column = dataset.columns
        self.build_edge_index_and_dataLoader(dataset,"test")
        test_loader = self.test_dataloader
        count = 0
        results = [0]*(self.window_size)
        for batch, target, attack_labels, edge_index in test_loader:
            batch, target, edge_index = [item.float().to(device) for item in [batch, target, edge_index]]
            count +=1
            print("testing batch iter:", count)

            with torch.no_grad():
                # w1,_ = self.encoder(batch)
                # w1=self.decoder(w1)
                loss = self.caculateMSE(batch,self.edge_index_sets,target)
                results.extend(loss.cpu().numpy())

            # del w1
            torch.cuda.empty_cache()
        results = self.testPostProcessing(results)
        return results
   

    def training_all(self,  dataset, opt_func=torch.optim.Adam) -> pd.DataFrame:
        print("============="+self.modelName+" start training==============")
        self.build_edge_index_and_dataLoader(dataset,"train")
        train_loader = self.train_dataloader
        val_loader = self.val_dataloader

        self.train(True)
        optimizer1 = opt_func(list(self.parameters()),lr=0.001)

        for epoch in range(self.target_epochs):
            epoch_loss = []
            for batch, target, attack_labels, edge_index in train_loader:
                batch, target, edge_index = [item.float().to(device) for item in [batch, target, edge_index]]

                # Train AE1
                loss1 = self.training_step(batch,target)
                loss1.backward()
                optimizer1.step()
                optimizer1.zero_grad()

                epoch_loss.append(loss1.item())

            print("epoch",epoch,"loss",loss1)
            # result = self.evaluate(val_loader, epoch+1)
            # self.epoch_end(epoch, result)
        self.saveModel()
        self.saveEdge_index()
        print("=================="+self.modelName+" end training==============")
    

    def training_step(self, batch,target):
        loss = self.caculateMSE(batch,self.edge_index_sets,target)
        loss = torch.mean(loss)
        return loss

    def caculateMSE(self, data,edge_index_sets,target):

        x = data.clone().detach()

        # batch_size ,feature_dim , window_size
        batch_num, input_feature_size, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()

        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            # edge_num = edge_index.shape[1]
            # cache_edge_index = self.cache_edge_index_sets[i]

            # if cache_edge_index is None or cache_edge_index.shape[1] != edge_num*batch_num:
            #     self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, input_feature_size).to(device)
            
            
            # all_embeddings.shape (input_feature_size,embedded_dim)
            all_embeddings = self.embedding(torch.arange(input_feature_size).to(device))

            weights_arr = all_embeddings.detach().clone()
            # all_embeddings.shape (batch_num*input_feature_size,embedded_dim)
            all_embeddings = all_embeddings.repeat(batch_num, 1)
            # print("all_embeddings.shape",all_embeddings.shape)

            weights = weights_arr.view(input_feature_size, -1)

            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1))
            cos_ji_mat = cos_ji_mat / normed_mat

            topk_num = self.topk

            # topk_indices_ji :(input_feature_size,topk)
            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]
            

            self.learned_graph = topk_indices_ji

            # gated_i .shape : (1,input_feature_size *topk_num)
            gated_i = torch.arange(0, input_feature_size).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            # print("gated_i",gated_i)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            # print("gated_j",gated_j)
            # gated_edge_index.shape:(2 ,topk_num)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)
            self.gated_edge_index = gated_edge_index
            # print("gated_edge_index",gated_edge_index)

            # batch_gated_edge_index.shape :( 2 , batch_num * node_Num*topk_num)
            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, input_feature_size).to(device)
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=input_feature_size*batch_num, embedding=all_embeddings)

            
            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        # x.shape (batch_size*input_feature_size,embedded_dim)
        x = x.view(batch_num, input_feature_size, -1)
        # x.shape (batch_size,input_feature_size,embedded_dim)


        indexes = torch.arange(0,input_feature_size).to(device)

        out = torch.mul(x, self.embedding(indexes))
        
        out = out.permute(0,2,1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, input_feature_size)

        # print("data.shape",data.shape)
        # print("x.shape",x.shape)
        # print("out.shape",out.shape)
        return torch.mean((out-target)**2,axis=1)
    def build_edge_index_sets(self,dataset):
        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_adj_list(dataset)
        fc_edge_index = adjList2_edgeIndex(fc_struc, list(dataset.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)
        self.edge_index_sets = []
        self.edge_index_sets.append(fc_edge_index)

    def build_edge_index_and_dataLoader(self,dataset,mode,labels=None):

        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_adj_list(dataset)
        fc_edge_index = adjList2_edgeIndex(fc_struc, list(dataset.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)
        self.edge_index_sets = []
        self.edge_index_sets.append(fc_edge_index)


        if labels == None:
            dataset_indata = construct_data(dataset, feature_map, labels=0)
        else:
            dataset_indata = construct_data(dataset, feature_map, labels=labels.tolist())
        # print("fc_struc",fc_struc)
        # print("fc_edge_index",fc_edge_index)
        # print("train_dataset_indata",type(train_dataset_indata))
        # print("train_dataset_indata",len(train_dataset_indata))
        # print("train_dataset_indata",train_dataset_indata[:2])


        cfg = {
            'slide_win': self.window_size,
            'slide_stride': 1
        }

        dataset = TimeDataset(dataset_indata, fc_edge_index, mode=mode, config=cfg)

        ########## output result

        if mode=="train":
            train_dataloader, val_dataloader = self.get_loaders(dataset, 5, self.batch_size, val_ratio = 0.2)
            self.train_dataloader = train_dataloader
            self.val_dataloader = val_dataloader
        else:
            self.test_dataloader = DataLoader(dataset, batch_size=self.batch_size,
                            shuffle=False, num_workers=self.cfg.hyperParam.worker)


    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        # split train_dataset to train and validation
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)


        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=True,num_workers=self.cfg.hyperParam.worker)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False,num_workers=self.cfg.hyperParam.worker)

        return train_dataloader, val_dataloader