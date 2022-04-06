
from re import A
import time
import gc
from torch.nn import parameter
from types import DynamicClassAttribute
import pickle
import copy
from models.GDN.GDN import GDN
from typing_extensions import get_origin
from numpy.core.numeric import outer
from sklearn.utils import validation
import torch
from torch import optim
import torch.nn as nn
from dataPreprocessing import *
from hydra.utils import get_original_cwd, to_absolute_path
from adtk.transformer import *
import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import numpy as np
from .normal_model import normal_model



from torch.optim import optimizer

from utils import *
device = get_default_device()

# test



class Encoder(nn.Module):
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, math.ceil(in_size/2))
        self.linear2 = nn.Linear(math.ceil(in_size/2), math.ceil(in_size/4))
        self.linear3 = nn.Linear(math.ceil(in_size/4), math.ceil(in_size/5))
        self.relu = nn.ReLU(True)
        self.count=0

    def forward(self, w):
        out = self.linear1(w)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        z = self.relu(out)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(math.ceil(out_size/5), math.ceil(out_size/4))
        self.linear2 = nn.Linear(math.ceil(out_size/4), math.ceil(out_size/2))
        self.linear3 = nn.Linear(math.ceil(out_size/2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        w = self.sigmoid(out)
        return w


class MY_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, output_dim, window_size):
        super().__init__()
        self.window_size = window_size
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        # print("in forward x.shape",x.shape)
        x = torch.reshape(x, (-1, self.window_size, self.input_dim))
        # print("in forward x.shape",x.shape)
        # print("x.shape",x.shape)
        # print("x.size(0)",x.size(0))
        # x shape torch.Size([2048,5,1])
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)
        out, hidden = self.lstm(x, (h0, c0))
        self.hidden = hidden
        # print("out",out)
        # print("out shape",out.shape)
        # out shape torch.Size([2048, 5, 64])
        #out = self.fc(out[:, -1, :])
        # print("out forward x.shape",x.shape)
        # print("out forward out.shape",out.shape)
        x = torch.reshape(x, (-1, self.window_size*self.input_dim))
        out = torch.reshape(out, (-1, self.window_size*self.output_dim))
        # print("out forward x.shape",x.shape)
        # print("out forward out.shape",out.shape)
        return out

    def getHidden(self):
        return self.hidden


class USAD(normal_model):
    def __init__(self, cfg, input_feature_dim, w_size,z_size,window_labels=None):
        super().__init__( cfg, "USAD",input_feature_dim, window_labels)
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)

    def training_step(self, batch):
        n = self.epoch
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
        loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
        return loss1+loss2

    # def training_all(self, epochs, dataset, opt_func=torch.optim.Adam) -> pd.DataFrame:
    #     print("============="+self.modelName+" start training==============")
    #     train_loader, val_loader = self.preProcessTrainingData(dataset)
    #     self.train()
    #     self.history = []
    #     optimizer1 = opt_func(
    #         list(self.encoder.parameters())+list(self.decoder1.parameters()))
    #     optimizer2 = opt_func(
    #         list(self.encoder.parameters())+list(self.decoder2.parameters()))
    #     for epoch in range(epochs):
    #         for [batch] in train_loader:
    #             # batch 是window_size*input_features_dim的一維陣列
    #             batch = to_device(batch, device)

    #             # Train AE1
    #             loss1= self.training_step(batch)
    #             loss1.backward()
    #             optimizer1.step()
    #             optimizer1.zero_grad()


    #         result = self.evaluate(val_loader, epoch+1)
    #         self.epoch_end(epoch, result)
    #         self.history.append(result)
    #     self.saveModel()
    #     print("=================="+self.modelName+" end training==============")
    def caculateMSE(self,batch):
        alpha =0.5
        beta = 0.5

        w1 = self.decoder1(self.encoder(batch))
        w2 = self.decoder2(self.encoder(w1))
        out = alpha*batch + beta*batch
        loss_plot = alpha*((batch-w1)**2) + beta*((batch-w2)**2)
        self.add_result_2_input_output_window_result(batch, out, loss_plot)
        loss = alpha*torch.mean((batch-w1)**2, axis=1) + beta*torch.mean((batch-w2)**2, axis=1)
        return loss



class AutoencoderModel(normal_model):
    def __init__(self,cfg, w_size, z_size, input_feature_dim,   window_labels):
        super().__init__(cfg, "Autoencoder",

                          input_feature_dim, window_labels)
        self.encoder = Encoder(w_size, z_size)
        self.decoder = Decoder(z_size, w_size)
        self.cfg=  cfg
        self.mse = nn.MSELoss()
        # if self.cfg.model.attention == True:
        #     self.attention = attentionLayer(input_feature_dim,1)


    def caculateMSE(self, batch):

        z = self.encoder(batch)
        out = self.decoder(z)
        # loss1 = self.mse(batch,out)
        loss1 = (out-batch)**2

        self.add_result_2_input_output_window_result(batch[:, -1*self.input_feature_dim:],
                                                     out[:, -1*self.input_feature_dim:], loss1[:, -1*self.input_feature_dim:])

        # if self.cfg.model.attention==True:
        #     loss1 = self.attention(loss1)

        loss1 = torch.mean(loss1, axis=1)
        return loss1

   



class LSTM_VAE_ENCODER(nn.Module):
    def __init__(self, input_feature_dim, hidden_dim, num_layers, latent_size, windows_size, bidirection=False):
        super(LSTM_VAE_ENCODER, self).__init__()

        self.windows_size = windows_size
        self.num_layers = num_layers
        
        if hidden_dim ==0:
            hidden_dim = 1
        self.hidden_dim = hidden_dim
        self.input_feature_dim = input_feature_dim
        self.num_layers = 2

        self.lstm = MY_LSTM(input_feature_dim, self.hidden_dim,
                            self.num_layers, self.hidden_dim, windows_size)
        self.hidden2mean = nn.Linear(hidden_dim, latent_size)
        self.hidden2logv = nn.Linear(hidden_dim, latent_size)

    def forward(self, x):

        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out = self.lstm(x)
        hidden = self.lstm.getHidden()
        hidden = hidden[0]
        hidden = hidden[0]
        # print("hidden.shape",hidden.shape)

        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        z = mean + std * eps
        # print("z.shape",z.shape)
        z.reshape(z.shape[0], 1, z.shape[1])
        # print("z.shape",z.shape)

        kld = -0.5 * torch.mean(1 + logv - mean.pow(2) - logv.exp()).to('cuda')

        return z, kld, hidden


class LSTM_VAE_DECODER(nn.Module):
    def __init__(self, latent_size, num_layers, output_feature_dim, windows_size, bidirection=False):
        super(LSTM_VAE_DECODER, self).__init__()
        if bidirection == True:
            self.bidirection = 2
        else:
            self.bidirection = 1

        self.windows_size = windows_size
        self.num_layers = num_layers
        self.latent_size = latent_size
        self.output_feature_dim = output_feature_dim
        self.num_layers = 2
        self.hidden_size = output_feature_dim
        self.input_feature_dim = latent_size

        #self.lstm = MY_LSTM(latent_size ,self.output_feature_dim, self.num_layers,self.output_feature_dim,windows_size)
        self.lstm = nn.LSTM(
            self.input_feature_dim, self.output_feature_dim, num_layers, batch_first=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)
        out, hidden = self.lstm(x, (h0, c0))
        return out


class dynamic_anomaly_threshold(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dynamic_anomaly_threshold_NN = nn.Linear(hidden_size, 1)
        self.loss_hidden_set = []
        # self.testing_loss_hidden_set=[]

    def add_data(self, loss, hidden):
        # print("loss",loss,"hidden",hidden)
        # print("type loss",type(loss),"type hidden",type(hidden))
        # print("loss.shape",loss.shape,"hidden.shape",hidden.shape)
        self.loss_hidden_set.append(
            {"loss": loss.detach(), "hidden": hidden.detach()})
        # self.loss_hidden_set.append({"loss":torch.randn(1).cuda(),"hidden":torch.randn(1,46).cuda()})

    # def add_testing_data(self,hidden):
    #   self.testing_loss_hidden_set.append({"hidden":hidden})
    def training_all(self):
        optimizer = torch.optim.Adam(list(self.parameters()))
        count = 0
        for epoch in range(40):
            print("epoch ", epoch)
            for loss_hidden in self.loss_hidden_set:
                count += 1
                pred = self.dynamic_anomaly_threshold_NN(loss_hidden["hidden"])
                loss = torch.mean((loss_hidden["loss"] - pred)**2)
                print("count", count, "loss", loss, "pred",
                      pred[:10], "label", loss_hidden["loss"][:10])
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def testing_all(self):
        result = []
        with torch.no_grad():
            for loss_hidden in self.loss_hidden_set:
                result.append(self.dynamic_anomaly_threshold_NN(
                    loss_hidden["hidden"]))

        return result

    def getThreshold(self):
        return self.testing_all()


class LSTM_VAE(normal_model):
    def __init__(self,cfg,  hidden_size, latent_size, input_feature_dim, windows_size,  window_labels):
        super().__init__(cfg, "LSTM_VAE",  input_feature_dim, window_labels)
        self.windows_size = windows_size
        self.input_feature_dim = input_feature_dim
        self.num_layers = 2
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.output_feature_dim = input_feature_dim

        # self.anomaly_threshold_estimater = dynamic_anomaly_threshold(self.hidden_size)
        self.encoder = LSTM_VAE_ENCODER(
            input_feature_dim, self.hidden_size, self.num_layers, self.latent_size, windows_size)
        self.decoder = LSTM_VAE_DECODER(
            self.latent_size, self.num_layers, self.output_feature_dim, windows_size)


    def caculateMSEAndKLD(self, batch):
        # print("batch.shape ",batch.shape)
        # print("batch",batch)
        latent, kld, hidden = self.encoder(batch)
        # hidden_history_list=[]
        # if action == "training":
        #       hidden_history_list.append(hidden)
        # print("latent.shape ",latent.shape)
        latent = torch.reshape(latent, (latent.shape[0], 1, latent.shape[1]))
        out = self.decoder(latent)
        out = torch.reshape(out, (out.shape[0], out.shape[2]))
        batchFinal = torch.reshape(
            batch, (out.shape[0], -1, out.shape[1]))[:, -1, :]
        # print("w1.shape",w1.shape)
        # print("batchFinal.shape ",batchFinal.shape)
        loss1 = (batchFinal-out)**2
        self.add_result_2_input_output_window_result(batchFinal, out, loss1)
        loss1 = torch.mean(loss1, axis=1)

        # print("hidden",hidden.shape)
        # print("loss1.shape",loss1.shape)

        # self.anomaly_threshold_estimater.add_data(loss1,hidden)
        return loss1, kld

    def caculateMSE(self, batch):
        loss1, _ = self.caculateMSEAndKLD(batch)
        return loss1

    def training_step(self, batch):
        epoch = self.epoch
        kld_times = 0
        if epoch <= 10:
            kld_times = 0
        if epoch > 10 and epoch < 20:
            kld_times = (epoch - 10)*0.1
        elif epoch >= 20:
            kld_times = 1

        loss, kld = self.caculateMSEAndKLD(batch)
        loss = torch.mean(loss)
        loss += kld_times * kld
        return loss


class CNN_LSTM(normal_model):
    def __init__(self,cfg, latent_size, input_feature_dim, windows_size,   window_labels=None):
        super().__init__(cfg, "CNN_LSTM",  input_feature_dim, window_labels)

        self.windows_size = windows_size
        self.latent_size = latent_size
        self.output_feature_dim = input_feature_dim
        self.num_layers = 2

        kernel_size = 3
        self.LSTM_output_size = int(self.input_feature_dim)
        padding = int((kernel_size-1)/2)
        self.conv1 = nn.Conv1d(in_channels=input_feature_dim,
                               out_channels=self.latent_size, kernel_size=kernel_size, padding=padding)
        self.lstm = nn.LSTM(
            self.latent_size, self.LSTM_output_size, self.num_layers, batch_first=True)
        self.FC = nn.Linear(self.LSTM_output_size, self.input_feature_dim)
        # self.bias = nn.Parameter(torch.zeros(self.input_feature_dim))

    def caculateMSE(self, batch):

        # convert batch shape to [epoch,window_size,input_feature_dim]
        batch = batch.reshape(batch.shape[0], -1, self.input_feature_dim)
        # print("batch.shape ",batch.shape)
        latent = batch.permute(0, 2, 1)
        # print("latent.shape ",latent.shape)

        latent = self.conv1(latent)

        latent = latent.permute(0, 2, 1)
        # print("--latent.shape ",latent.shape)

        h0 = torch.zeros(self.num_layers, latent.size(0),
                         self.LSTM_output_size).to(device)
        c0 = torch.zeros(self.num_layers, latent.size(0),
                         self.LSTM_output_size).to(device)
        out, hidden = self.lstm(latent, (h0, c0))
        out = self.FC(out)
        # out = out + self.bias

        loss1 = torch.mean((batch-out)**2, axis=1)

        self.add_result_2_input_output_window_result(
            batch[:, -1, :], out[:, -1, :], loss1)

        loss1 = torch.mean(loss1, axis=1)
        # self.input_output_windows_result["input"].extend(batch.detach().cpu().numpy()),"output":out.detach().cpu().numpy(),"loss":loss1.detach().cpu().numpy()})
        return loss1



class MEMAE(normal_model):
    def __init__(self,cfg, input_size, latent_size, input_feature_dim,   window_labels):
        super().__init__(cfg, "MEMAE",  input_feature_dim, window_labels)
        w_size = input_size
        w_size_div_2 = w_size//2
        w_size_div_4 = w_size//4
        w_size_div_8 = w_size//8

        self.encoder = bootstraps_encoder(input_size,latent_size)
        self.decoder = bootstraps_decoder(latent_size,input_size)
        mem_dim = 120
        self.mem_rep = MemModule(mem_dim=mem_dim, fea_dim=latent_size)

    def forward(self,batch):
        z = self.encoder(batch)
        res_mem = self.mem_rep(z)
        z = res_mem['output']
        att = res_mem['att']
        out= self.decoder(z)
        return out,att



    def caculateMSEAndAttrEntropy(self, batch):
        # print("batch.size()",batch.size())
        out,att = self.forward(batch)

        loss1 = (batch-out)**2
        # 這邊不能只 給window的最後一個 要整段window都給
        self.add_result_2_input_output_sliding_window_result(batch,out, loss1)
        loss1 = torch.mean(loss1, axis=1)

        return loss1,att

    def caculateMSE(self, batch):
        loss1, _ = self.caculateMSEAndAttrEntropy(batch)
        return loss1

    def training_step(self, batch):
        loss, att= self.caculateMSEAndAttrEntropy(batch)

        entropy_loss = torch.mean((-att) * torch.log(att + 1e-12))
        loss = torch.mean(loss)
        loss += 0.0002*entropy_loss
        return loss

    # def training_step(self, batch):
    #     # loss, att= self.caculateMSEAndAttrEntropy(batch)

    #     # entropy_loss = torch.mean((-att) * torch.log(att + 1e-12))
    #     # loss = torch.mean(loss)
    #     # loss += 0.0002*entropy_loss
    #     z = self.encoder(batch)
    #     out = self.decoder(z)
    #     return torch.mean((batch-out)**2)



class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(
            self.mem_dim, self.fea_dim))  # M x C
        self.bias = None
        self.shrink_thres = shrink_thres
        # self.hard_sparse_shrink_opt = nn.Hardshrink(lambd=shrink_thres)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # do inner product
        # T 因該是batch
        # Fea x Mem^T, (TxC) x (CxM) = TxM
        att_weight = F.linear(input, self.weight)
        att_weight = F.softmax(att_weight, dim=1)  # TxM
        # ReLU based shrinkage, hard shrinkage for positive value
        if(self.shrink_thres > 0):
            att_weight = self.hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            # att_weight = F.softshrink(att_weight, lambd=self.shrink_thres)
            # normalize???
            att_weight = F.normalize(att_weight, p=1, dim=1)
            # att_weight = F.softmax(att_weight, dim=1)
            # att_weight = self.hard_sparse_shrink_opt(att_weight)
        mem_trans = self.weight.permute(1, 0)  # Mem^T, MxC
        # AttWeight x Mem^T^T = AW x Mem, (TxM) x (MxC) = TxC
        output = F.linear(att_weight, mem_trans)
        return {'output': output, 'att': att_weight}  # output, att_weight

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )
    def hard_shrink_relu(self,input, lambd=0, epsilon=1e-12):
        output = (F.relu(input-lambd) * input) / \
            (torch.abs(input - lambd) + epsilon)
        return output


# NxCxHxW -> (NxHxW)xC -> addressing Mem, (NxHxW)xC -> NxCxHxW
# BxC -> (Bx1)xC -> addressing Mem, (Bx1)xC -> BxC
class MemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025, device='cuda'):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim  
        self.fea_dim = fea_dim  
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

    def forward(self, input):
        x = input.contiguous()  # [64, 3]

        y_and = self.memory(x)

        y = y_and['output']
        att = y_and['att']

        return {'output': y, 'att': att}



class DAGMM(normal_model):
    def __init__(self,cfg,input_feature_size,window_labels):
        super(DAGMM, self).__init__(cfg, "DAGMM",  input_feature_size, window_labels)
        self.latent_dim = 3  # 1 + 2
        self.n_gmm = 2

        self.lambda_energy = 0.1
        self.lambda_cov_diag = 0.005
        self.input_feature_size = input_feature_size
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_feature_size, self.input_feature_size//2),
            nn.BatchNorm1d(self.input_feature_size//2),
            nn.Tanh(),
            nn.Linear(self.input_feature_size//2, math.ceil(self.input_feature_size/4)),
            nn.BatchNorm1d(math.ceil(self.input_feature_size/4)),
            nn.Tanh(),
            nn.Linear(math.ceil(self.input_feature_size/4), math.ceil(self.input_feature_size/8)),
            nn.BatchNorm1d(math.ceil(self.input_feature_size/8)),
            nn.Tanh(),
            nn.Linear(math.ceil(self.input_feature_size/8), 1),
            nn.BatchNorm1d(1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(1, math.ceil(self.input_feature_size/8)),
            nn.BatchNorm1d(math.ceil(self.input_feature_size/8)),
            nn.Tanh(),
            nn.Linear(math.ceil(self.input_feature_size/8), math.ceil(self.input_feature_size/4)),
            nn.BatchNorm1d(math.ceil(self.input_feature_size/4)),
            nn.Tanh(),
            nn.Linear(math.ceil(self.input_feature_size/4), self.input_feature_size//2),
            nn.BatchNorm1d(self.input_feature_size//2),
            nn.Tanh(),
            nn.Linear(self.input_feature_size//2, self.input_feature_size),
            nn.BatchNorm1d(self.input_feature_size),
        )

        self.estimation = nn.Sequential(
            nn.Linear(self.latent_dim, 10),
            # nn.BatchNorm1d(10),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(10, self.n_gmm),
            nn.Softmax(dim=0)
        )
        self.count=0

    def caculateGaussian(self, x):


        z_c = self.encoder(x)
        output = self.decoder(z_c)

        cos = torch.nn.functional.cosine_similarity(x, output, dim=1)  # cosine similarity
        euc = (x-output).norm(2, dim=1) / x.norm(2, dim=1)  # relative euclid distance

        z_r = torch.cat([cos.unsqueeze(-1), euc.unsqueeze(-1)], dim=1)  # [batch size, z_r dim:2]
        z = torch.cat([z_c, z_r], dim=1)  # [batch size, z_c dim + z_r dim]

        gamma = self.estimation(z)  # [batch size, n_gmm] [64, 2]
        phi = torch.sum(gamma, dim=0) / gamma.size(0)
        mu = torch.sum(gamma.unsqueeze(-1) * z.unsqueeze(1), dim=0) / torch.sum(gamma, dim=0).unsqueeze(-1)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))  # [64, 2, 5]
        z_mu_outer = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)  # [64, 2, 5, 5]

        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_outer, dim=0)\
              / torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

        return {'output': output, 'z': z, 'gamma': gamma, 'phi': phi, 'mu': mu, 'cov': cov}  # 여기까지 확인 완료!

    def compute_energy(self, z, phi, mu, cov, size_average=True):
        k, D, _ = cov.size()

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))

        cov_inverse = []
        det_cov = []
        cov_diag = 0
        eps = 1e-12

        for i in range(k):
            # K x D x D
            cov_k = cov[i] + (torch.eye(D)*eps).cuda()
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))  # [1, 5, 5]

            det_cov.append(torch.cholesky((2*np.pi) * cov_k).diag().prod().unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())

        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov)

        # maybe avoid overflow
        exp_term_tmp = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        max_val = torch.max(exp_term_tmp.clamp(min=0), dim=1, keepdim=True)[0]

        exp_term = torch.exp(exp_term_tmp - max_val)
        sample_energy = -max_val.squeeze() - torch.log(
            torch.sum(phi.unsqueeze(0) * exp_term / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)

        if size_average:
            sample_energy = torch.mean(sample_energy)

        return sample_energy, cov_diag

    def training_step(self, input):
        outputs = self.caculateGaussian(input)
        output = outputs['output']
        energy, cov_diag = self.compute_energy(outputs['z'], outputs['phi'], outputs['mu'], outputs['cov'])
        recon_error = torch.mean((output - input) ** 2)
        loss = recon_error + self.lambda_energy * energy + self.lambda_cov_diag * cov_diag

        return loss

    def caculateMSE(self, batch):
        outputs = self.caculateGaussian(batch)
        out = outputs['output']
        batch_error = torch.mean((out- batch) ** 2, 1)

        print("batch_error.shape",batch_error.shape,"out",out.shape)
        self.add_result_2_input_output_sliding_window_result(batch,out, (batch-out)**2)
        self.add_result_2_input_output_window_result(batch,out, (batch-out)**2)

        return batch_error




class RANCoders(normal_model):
    def __init__(
            self, 
            cfg,
            w_size,z_size,
            input_feature_dim,window_labels,
            n_estimators: int = 2,
            max_features: int = 51,
            latent_dim: int = 2, 
            delta: float = 0.05,
            activation: str = 'linear',
            output_activation: str = 'linear',

    ):
        super(RANCoders, self).__init__(cfg,"RANCoders",input_feature_dim,window_labels)
        self.input_feature_dim = input_feature_dim
        self.max_features = self.cfg.model.max_features
        self.n_estimators = self.cfg.model.n_estimators 
        self.latent_dim = self.cfg.model.latent_dim
        self.featureSelectName = self.cfg.model.featureSelect

        self.max_features = min(self.max_features,self.input_feature_dim)

        self.count=0
        featureSelect = self.getFeatureSelect(self.featureSelectName)
        print("enModelNameList",self.cfg.model.enModel)
        self.buildModule(featureSelect,self.cfg.model.enModel)

    def buildModule(self,featureSelect,enModelNameList):
        self.enModels= nn.ModuleDict({
            'enModel_{}'.format(i): self.getEleModel(enModelNameList[i%len(enModelNameList)],self.max_features)
            for i in range(self.n_estimators)
        })
        # self.randsamples = nn.Parameter(torch.from_numpy(
        #             featureSelect
        #         # np.array([[1,2,3]])
        #         ).cuda(),requires_grad=False).long().unsqueeze(2).expand(-1,self.max_features,self.batch_size).transpose(1,2)
        self.randsamples = featureSelect

    def getEleModel(self,enModelName,input_size):
        window_labels = None
        latent_dim = self.latent_dim
        out_size = input_size
        # w_size = input_feature_size 
        # z_size= math.ceil(input_feature_size/5)
        # w_size = 一整個window 的latent 變成一維
        # hidden_size = 3
        print('latent_dim',latent_dim,"input_size",input_size,"self.window_size",self.window_size)
        


        if enModelName == "B_Autoencoder":
            model = bootstraps_Autoencoder(input_size,latent_dim,out_size)
        elif enModelName == "USAD":
            model = USAD(self.cfg,input_size,input_size, latent_dim,window_labels)
        elif enModelName == "DAGMM":
            model = DAGMM(self.cfg,input_size,window_labels)
        elif enModelName == "Autoencoder":
            model = AutoencoderModel(self.cfg,input_size, latent_dim, input_size,window_labels)
        elif enModelName == "LSTM_VAE":
            model = LSTM_VAE(self.cfg,math.ceil(input_size/2), latent_dim, input_size, self.window_size,window_labels)
        elif enModelName == "CNN_LSTM":
            model = CNN_LSTM(self.cfg,latent_dim, input_size, self.window_size,window_labels)
        elif enModelName == "MEMAE":
            model = MEMAE(self.cfg,input_size,latent_dim, input_size, window_labels)
        elif enModelName == "GDN":
            config = copy.deepcopy(self.cfg)
            config.model = self.cfg.model.GDN
            model = GDN(config,input_size,window_labels)
        else:
            print("model name not found",enModelName)
            exit()
        return model

    def getFeatureSelect(self,featureSelectName):
        if featureSelectName == "random":
            return self.randomFeatureSelect()
        elif featureSelectName == "const":
            return self.constFeatureSelect()
        elif featureSelectName == "GDN":
            return self.GDNFeatureSelect()

    def randomFeatureSelect(self):
        featureSelect= np.concatenate(
                [
                np.sort(np.random.choice(
                    self.input_feature_dim, replace=False, size=(1, self.max_features),
                )) for i in range(self.n_estimators)
                ]
            )
        print("featureselect.shape",featureSelect.shape)
        return featureSelect

    def constFeatureSelect(self):
        edge_index_path = get_original_cwd()+"/result/total/edge_index.txt"
        with open(edge_index_path,'rb') as f:
            edge_index = pickle.load(f)
        print("edge_index",edge_index)

        adjList = self.edge_index_2_adjList(edge_index)
        print("adjList",adjList)

        self.useGNNSetConfig(adjList)
        feature_selector = np.array([
            np.sort(np.array(adjList[i]))for i in range(self.n_estimators) 
            ])
        print("featureselect.shape",feature_selector.shape)
        return feature_selector

    def GDNFeatureSelect(self):
        edge_index = self.loadEdge_index()
        adjList = self.edge_index_2_adjList(edge_index)
        self.useGNNSetConfig(adjList)
        feature_selector = np.array([
            np.sort(np.array( adjList[i])) for i in range(self.n_estimators) 
            ])
        print("featureselect.shape",feature_selector.shape)
        return feature_selector
        
            
    def loadEdge_index(self):
        edge_index_path = get_original_cwd()+"/result/" + self.datasetName+"/edge_index.txt"
        with open(edge_index_path,'rb') as f:
            edge_index = pickle.load(f)
        # print("edge_index",edge_index)
        return edge_index
    def useGNNSetConfig(self,adjList):
        self.n_estimators = len(adjList) 
        self.max_features = len(adjList[0])
        

    def edge_index_2_adjList(self,edge_index):
        adjList = {}
        for i in range(len(edge_index[0])):
            nodeJ = edge_index[0][i]
            nodeI = edge_index[1][i]
            if nodeI in adjList:
                adjList[nodeI].append(nodeJ)
            else:
                adjList[nodeI] = [nodeJ]
        self.topK = len(adjList[0])
        self.max_features=self.topK
        return adjList




    def trainAllModel(self, inputs):
        print("self.randsmpales.shape",self.randsamples.shape)
        
        for i in range(self.n_estimators):
            curModel = self.enModels['enModel_{}'.format(i)]
            print("-------start trainAllModel model:",i,"curModel",curModel)
            start = time.time()
            curModel.training_all(inputs.iloc[:,self.randsamples[i]])
            end = time.time()
            print("-------end model:",i,"time:",end-start)
            del curModel
            gc.collect()

    def testAllModel(self, inputs):
        
        z = []
        for i in range(self.n_estimators):
            curModel = self.enModels['enModel_{}'.format(i)]
            print("-------testAllModel model:",i,"curModel",curModel)
            z.append(curModel.testing_all(inputs.iloc[:,self.randsamples[i]]).values)
            curModel.clean()

        # z = [ self.enModels['enModel_{}'.format(i)].get_anomaly_score() for i in range(self.n_estimators)]
        out = np.asarray(z)
        print("out",out)
        print("out.shape",out.shape)
        return out

    def training_all(self, dataset):
        print("============="+self.modelName+" start training==============")
        self.trainAllModel(dataset)
        self.saveModel()
        print("=================="+self.modelName+" end training==============")

    def testing_all(self,dataset):
        self.testPreprocessing()
        self.dataset_window_index = dataset.index
        self.dataset_window_column = dataset.columns

        each_estimater_all_loss = self.testAllModel(dataset)
        # batch_loss = np.reshape(each_estimater_batch_loss,(self.n_estimators,self.batch_size,-1))
        # batch = batch.expand(out.shape[0],batch.shape[0],batch.shape[1])
        # o_hi = np.transpose(o_hi,(1,0,2))
        # loss = (out-batch)**2
        all_loss= np.transpose(each_estimater_all_loss,(1,0))
        print("all_loss.shape 2",all_loss.shape)
        # print("2batch_loss.shape",batch_loss.shape)
        # loss = torch.reshape(loss,(loss.shape[0],-1))
        all_loss =  np.mean(all_loss,axis=1)
        print("all_loss.shape 3",all_loss.shape)
        all_loss = self.testPostProcessing(all_loss)
        # print("3all_loss.shape",all_loss.shape)
        return all_loss

    def saveModel(self):
        # super().saveModel()
        # print("self.state_dict()",self.state_dict())
        torch.save(self.state_dict(), get_original_cwd()+"/model/" +
                   self.datasetName+"/"+self.modelName+"/model.pth")
        with open(get_original_cwd()+"/model/" +
        self.datasetName+"/"+self.modelName+"/randsamples.pth","wb") as f:
            pickle.dump(self.randsamples,f)

    def loadModel(self):
        # super().loadModel()
        checkpoint = torch.load(
            get_original_cwd()+"/model/"+self.datasetName+"/"+self.modelName+"/model.pth")
        self.load_state_dict(checkpoint)
        # print("checkpoint",checkpoint)
        with open(get_original_cwd()+"/model/" +
        self.datasetName+"/"+self.modelName+"/randsamples.pth","rb") as f:
            self.randsamples = pickle.load(f)

class bootstraps_Autoencoder(normal_model):
    def __init__(self, in_size,latent_dim: int,out_size):
        super().__init__()
        self.encoder = bootstraps_encoder(in_size,latent_dim)
        self.decoder = bootstraps_decoder(latent_dim,out_size)
    def forward(self,batch):
        z = self.encoder(batch)
        out  = self.decoder(z)
        return out
    def training_step(self,batch):
        return torch.mean(self.forward(batch))
    def caculateMSE(self,batch):
        return torch.mean(self.forward(batch),dim=1)

        

class bootstraps_encoder(nn.Module):
    count =0
    def __init__(self, in_size,latent_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_size, math.ceil(in_size/2))
        self.linear2 = nn.Linear(math.ceil(in_size/2), math.ceil(in_size/4))
        self.latent = nn.Linear(math.ceil(in_size/4),latent_dim)
        self.relu = nn.ReLU(True)
        

    def forward(self, w):
        # self.count +=1
        # print("bootstraps self.count",self.count)
        # if self.count ==1:
        #     print("w",w)

        out = self.linear1(w)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.latent(out)
        z= self.relu(out)
        return z
    
class bootstraps_decoder(nn.Module):
    def __init__(self,latent_dim,out_size):
        super().__init__()
        self.latent = nn.Linear(latent_dim,math.ceil(out_size/4))
        self.linear2 = nn.Linear(math.ceil(out_size/4), math.ceil(out_size/2))
        self.linear1 = nn.Linear(math.ceil(out_size/2), out_size)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        out = self.latent(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear1(out)
        w = self.sigmoid(out)
        return w

# class attentionLayer(nn.Module):
#     def __init__(self,input_size,output_size):
#         super().__init__()
#         self.attention = nn.Linear(input_size,output_size,bias=False)
#     def forward(self,input):
#         return self.attention(input)
