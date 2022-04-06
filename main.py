#!/usr/bin/env python
# coding: utf-8

# parameter

import sys
sys.path.append("../")
import pickle
import argparse
import torch.utils.data as data_utils
from dataPreprocessing import *
from models.model import *
from utils import *
import torch.nn as nn
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from adtk.transformer import *
from adtk.detector import *
from src.adtk.detector._detector_hd import ML_Detector
from adtk.detector import OutlierDetector
from sklearn.neighbors import LocalOutlierFactor
from enum import Enum
from models.GDN.GDN import GDN


from hydra import compose, initialize
from omegaconf import OmegaConf
from omegaconf import DictConfig
import hydra
from adtk.pipe import Pipenet
from adtk.aggregator import *
from threshold_select import get_threshold_select_obj





config_file_path = "config"


#########################

# get_ipython().system('nvidia-smi -L')

class detector_type(Enum):
    adtk = 1
    ourModel = 2

class execution:
    def __init__(self,cfg,modelName,datasetName,n_epochs,batch_size,window_size):
        self.cfg = cfg
        self.datasetName = datasetName
        self.dataPreprocessingObj = DataProcessing(cfg,cfg["dataPreprocessing"])
        self.modelName = modelName
        self.device = get_default_device()
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.window_size = window_size

        
    def get_ML_Model(self,input_feature_size,window_labels = None):

        # w_size = 一整個window 的input 變成一維
        w_size = input_feature_size * self.window_size
        # w_size = 一整個window 的latent 變成一維
        # hidden_size = 3
        hidden_size = input_feature_size//3 
        if "latent_size" in self.cfg.dataset:
            hidden_size = self.cfg.dataset["latent_size"]
        z_size = self.window_size*hidden_size

        if self.modelName == "USAD":
            model = USAD(self.cfg,input_feature_size,w_size, z_size,window_labels)
        elif self.modelName == "RANCoders":
            model = RANCoders(self.cfg,w_size,z_size,input_feature_size,window_labels)
        elif self.modelName == "DAGMM":
            model = DAGMM(self.cfg,input_feature_size,window_labels)
        elif self.modelName == "Autoencoder":
            model = AutoencoderModel(self.cfg,w_size, z_size, input_feature_size,window_labels)
        elif self.modelName == "LSTM_VAE":
            model = LSTM_VAE(self.cfg,input_feature_size-5, hidden_size,
                             input_feature_size, self.window_size,window_labels)
        elif self.modelName == "CNN_LSTM":
            model = CNN_LSTM(self.cfg,hidden_size, input_feature_size, self.window_size,window_labels)
        elif self.modelName == "MEMAE":
            model = MEMAE(self.cfg,w_size,z_size,w_size, input_feature_size, window_labels)
        elif self.modelName == "GDN":
            model = GDN(self.cfg,input_feature_size,window_labels)

        # elif self.modelName == "MEMAE":
        #     model = GDN(edge_index_sets, len(feature_map), 
        #         dim=train_config['dim'], 
        #         input_dim=train_config['slide_win'],
        #         out_layer_num=train_config['out_layer_num'],
        #         out_layer_inter_dim=train_config['out_layer_inter_dim'],
        #         topk=train_config['topk']
        #     ).to(self.device)
        else:
            print("model name not found")
            exit()

        model = to_device(model, self.device)
        return model
    def get_dectector(self,input_feature_size,y_True=None):
        if self.modelName == "OutlierDetector":
            print("-------modelName OutlieDetector")
            Dectector_obj= OutlierDetector(LocalOutlierFactor(contamination=0.05))
            adtkOrOurModel = detector_type.adtk
        else:
            model = self.get_ML_Model(input_feature_size,y_True)
            print("----------- model--------- \n",model)
            Dectector_obj= ML_Detector(model)
            adtkOrOurModel = detector_type.ourModel

        return Dectector_obj,adtkOrOurModel

    
    def train(self):

        normal,input_feature_size = self.dataPreprocessingObj.HandleNormalData(self.window_size,self.datasetName)

        # labels,self.origin_attack,attack,input_feature_size = self.dataPreprocessingObj.HandleAnomalyData(
        #     window_size,datasetName)

        Detector_obj,detectorType= self.get_dectector(input_feature_size)
        Detector_obj.fit(normal)
        # paramters =  Detector_obj.get_params()
        # print("train finish parameter: ",paramters)
        # print("train finish parameter[model]: ",paramters['model'])
        # print("train finish parameter type[model]: ",type(paramters['model']))
        # if detectorType ==  detector_type.adtk:
        #     pickle.dump(paramters,open("model/"+self.datasetName+"/"+self.modelName+".pkl","wb"))
        # Detector_obj.get_params())
        
        

    def test(self):
        self.LabelObj,self.y_True,origin_attack,self.attack,self.input_feature_size = self.dataPreprocessingObj.HandleAnomalyData(
            self.window_size,self.datasetName)

        # normal, input_feature_size = self.dataPreprocessingObj.HandleNormalData(normal_data_path,window_size)

        Detector_obj,detectorType= self.get_dectector(self.input_feature_size,self.y_True.values)
        # Detector_obj.fit(attack)
        # if detectorType ==  detector_type.adtk:
        #     parameters =  pickle.load(open("model/"+self.datasetName+"/"+self.modelName+".pkl","rb"))
            # print("parametrs shape",len(parameters))
            # print("type parametrs ",type(parameters))
            # print("parameters:",parameters)
            # Detector_obj.set_params(**parameters)
        y_anomaly= Detector_obj.detect(self.attack)
        
    
        
        self.handle_testing_result(origin_attack,detectorType,y_anomaly,Detector_obj,self.y_True)
    

    def handle_testing_result(self,origin_attack,detectorType,y_anomaly:pd.Series,Detector_obj,y_True:pd.Series):
        
        if detectorType== detector_type.adtk:
            # plot(self.attack, anomaly=y_anomaly, ts_linewidth=1, ts_markersize=3, anomaly_color='red', anomaly_alpha=0.3, curve_group='all');
            evaluateResult(y_True.values,y_anomaly.values,0,self.modelName,self.datasetName)
        else:
            model = Detector_obj.getModel()
            # y_anomaly_score= model.get_anomaly_score()
            y_anomaly_score= y_anomaly
            if self.modelName== "MEMAE":
                input_output_window_result = model.get_input_output_sliding_window_result()
            else:
                input_output_window_result = model.get_input_output_window_result()
            input_window= input_output_window_result["input"]
            output_window =input_output_window_result["output"]
            loss= input_output_window_result["loss"]
            self.print_testing_result(y_True,y_anomaly_score)
            self.plotResult(origin_attack,input_window,output_window,loss)



    def print_testing_result(self,y_True:pd.Series,y_anomaly_score:np.array):
        pd.DataFrame(y_anomaly_score).to_csv("anomaly_score.csv")
        pd.DataFrame(y_True).to_csv("label.csv")
        ROC(y_True.values, y_anomaly_score, self.modelName,self.datasetName)
        make_figure(y_True,y_anomaly_score)
        self.threshold_select_obj = get_threshold_select_obj(self.cfg,y_True.index)
        self.threshold_select_obj.run(y_anomaly_score,y_True)



    def plotResult(self,origin_attack,input_window,output_window,loss):
        if self.modelName == "RANCoders":
            return 
        plotFeatureObj=plotFeature(origin_attack,self.modelName,self.datasetName,self.window_size,self.input_feature_size,self.y_True)
        if self.modelName == "MEMAE":
            plotFeatureObj.plot_sliding_window_feature(origin_attack,input_window,output_window,loss,self.LabelObj)
        elif type(input_window) is not list:
            plotFeatureObj.plot_time_slot_feature(self.dataPreprocessingObj,origin_attack,input_window,output_window,loss)

    
    

    
# @hydra.main(config_path="conf", config_name="config")
@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) :
    print("*******************************config******************")
    print(OmegaConf.to_yaml(cfg))
    print("*******************************config end***************")
    batch_size = cfg.hyperParam.batch_size
    window_size = cfg.dataPreprocessing.sliding_window.window_size
    n_epochs = cfg.hyperParam.epochs

    executionObj = execution(cfg,cfg.model.name,cfg.dataset.name,n_epochs,batch_size,window_size)
    if cfg.action.name == "test":
        executionObj.test()
    else:
        executionObj.train()



if __name__ == "__main__":
    argparserObj = argparse.ArgumentParser()
    # argparserObj.add_argument(
    #     "--model", type=str, help="{USAD|autoencoder|LSTM_USAD|LSTM_VAE|CNN_LSTM}")
    # argparserObj.add_argument("--action", type=str, help="{test|train}")
    # argparserObj.add_argument("--dataset", type=str, help="{SWAT|WADI|SWATDebug}")
    # args = argparserObj.parse_args()
    # argparserObj.add_argument("--config", type=str)
    # args = argparserObj.parse_args()
    # config_file_path = args.config
    main()
    