
from ast import parse
import numpy as np
from torch.functional import norm
import torch.utils.data as data_utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import precision_recall_fscore_support as score

import sys
sys.path.append("../")
from src.adtk.transformer._transformer_hd import *
from sklearn.metrics import roc_curve,roc_auc_score

import sys

from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import SeasonalAD
from adtk.transformer import *
from utils import *
from adtk.detector import RegressionAD
from sklearn.linear_model import LinearRegression
from adtk.detector import PcaAD
from dataLoader import *

class DataProcessing:
    def __init__(self,cfg,dataPreprocessingCfg):
        self.dataPreprocessingCfg = dataPreprocessingCfg
        # self.sliding_window_size = dataPreprocessing
        self.cfg=cfg
        self.sliding_window_step = 1
        if "sliding_window" in  cfg.model:
            self.sliding_window_step = cfg.model.sliding_window.window_step
    
    def dataPreprocessing(self,dataset):
        print("-----before dataPreprocessing data.shape",dataset.shape)
        # Transform all columns into float64 
        dataset = dataset.astype(float)
        # print("dataset dtype",dataset.dtypes)
        dataset = validate_series(dataset)
        if "CorrelationMatrix"in self.dataPreprocessingCfg:
            window_size =self.dataPreprocessingCfg["CorrelationMatrix"]["window"]
            dataset = correlationMatrix(window_size = window_size).transform(dataset)
        if "PcaProjection" in self.dataPreprocessingCfg:
            dim = self.dataPreprocessingCfg["PcaProjection"]["dim"]
            dataset = PcaProjection(k=dim).fit_transform(dataset)

        if "DoubleRollingAggregate" in self.dataPreprocessingCfg:
            agg = self.dataPreprocessingCfg["DoubleRollingAggregate"]['agg']
            window = self.dataPreprocessingCfg["DoubleRollingAggregate"]['window']
            dataset = DoubleRollingAggregate(agg=agg,window=window).transform(dataset)
            dataset.dropna(inplace=True)
        if "RollingAggregate" in self.dataPreprocessingCfg:
            agg = self.dataPreprocessingCfg["RollingAggregate"]['agg']
            window = self.dataPreprocessingCfg["RollingAggregate"]['window']
            dataset = RollingAggregate(agg=agg,window=window).transform(dataset)
            dataset.dropna(inplace=True)

        if self.cfg['action']['name'] == 'train':
            if "downSample" in self.dataPreprocessingCfg:
                window = self.dataPreprocessingCfg["downSample"]['window']
                dataset = downSample(window).transform(dataset)

        # if self.dataPreprocessingCfg["ClassicSeasonalDecomposition"]["use"]:
        #     dataset = ClassicSeasonalDecomposition().fit_transform(dataset)
        
        dataset = normalization().transform(dataset)
        print("---afterPreprocessing dataset\n",dataset.head(2))
        feature_dim = dataset.shape[1]
        return dataset,feature_dim
    
    
    def HandleNormalData(self,window_size,datasetName):
        # normal,attack,labels = WADI_loadData(normal_data_path,attack_data_path)
        dataset_loader = self.get_dataset_loader(datasetName)
        origin_normal,_= dataset_loader.loadNormalData()
        normal,input_feature_size=self.dataPreprocessing(origin_normal)
        normal = seq2window(window_size,self.sliding_window_step).transform(normal)
        print("!!!!!!!final dataset shape",normal.shape)
        with open("training_data_info.txt","w") as f :
            print("without preprocessing dataset.shape",normal.shape,file = f)
            print("after preprocessing dataset.shape",normal.shape,file = f)
            print("input_feature_size",input_feature_size,file = f)

            
        return normal,input_feature_size
    def HandleAnomalyData(self,window_size,datasetName):
        # normal,attack,labels = WADI_loadData(normal_data_path,attack_data_path)
        dataset_loader = self.get_dataset_loader(datasetName)
        origin_without_preProcess_attack,labels,_= dataset_loader.loadAnomalyData()
        attack,input_feature_size=self.dataPreprocessing(origin_without_preProcess_attack)
        attack = seq2window(window_size,self.sliding_window_step).transform(attack)

        # print("origin label",labels)
        self.LabelObj = Label(labels,window_size,attack.index)
        y_True = self.LabelObj.handleLabel()
        # print("dataset.index",attack.index)
        # print("label",self.LabelObj.getLabel())
        # print("dataset",attack)

        print("label info:\n",self.LabelObj.y_True.value_counts())
        print("-----final dataset shape",attack.shape)
        with open("testing_data_info.txt","w") as f:
            print("without preprocessing dataset.shape",origin_without_preProcess_attack.shape,file = f)
            print("after preprocessing dataset.shape",attack.shape,file = f)
            print("label info:\n",self.LabelObj.y_True.value_counts(),file = f)
            print("input_feature_size",input_feature_size,file = f)

        return self.LabelObj,y_True,origin_without_preProcess_attack,attack,input_feature_size

    def get_dataset_loader(self,datasetName):
        if datasetName == "WADI":
            return WADI_dataset_loader()
        elif datasetName == "SWAT":
            return SWAT_dataset_Loader()
        elif datasetName == "NSL_KDD":
            return NSL_KDD_dataset_Loader()
        elif datasetName == "PSM":
            return PSM_dataset_Loader()
        elif datasetName == "SMD":
            return SMD_dataset_Loader()
        elif datasetName == "SWAT_P1":
            return SWAT_P1_dataset_loader()
        elif datasetName == "SWATDebug":
            return SWATDebug_dataset_loader()
        elif datasetName == "SWAT2000":
            return SWAT2000_dataset_loader()
        elif datasetName == "WADIDebug":
            return WADIDebug_dataset_loader()
        elif datasetName == "Chinatown":
            return Chinatown()
        elif datasetName == "Crop":
            return Crop()
        elif datasetName == "Wafer":
            return Wafer()
        elif datasetName == "DistalPhalanxOutlineCorrect":
            return DistalPhalanxOutlineCorrect()
        else:
            print("!!!! datasetname "+datasetName+" not found")
            exit()


class Label:
    def __init__(self,original_labels,sliding_window_size,datasetIndex):
        self.original_labels = original_labels
        self.sliding_window_size = sliding_window_size
        self.afterprocessing_datasetIndex= datasetIndex
    
    def handleLabel(self):
        self.y_True = self.seqLabel_2_WindowsLabels(self.original_labels.values,self.afterprocessing_datasetIndex)
        # print("y_True",self.y_True)
        return self.y_True

    # def cropLabelDueToDatasetSizeChange(self,lables,datasetSize):
    #     return pd.Series(index = self.original_labels.index[-1*:],data = y_True_values[-1*size:])
    def getWindowIndexByWindowLastIndex(self,windowLastIndex):
        window_column_index_loc= self.original_labels.index.get_loc(windowLastIndex)
        windowIndex =  self.original_labels.index[window_column_index_loc-self.sliding_window_size+1:window_column_index_loc+1]
        return windowIndex
        # print("index_loc",window_column_index_loc,"index",window_column_index,"windows_labels",windows_labels[-1])

        
    def seqLabel_2_WindowsLabels(self, labels,afterprocessing_datasetIndex):
        # 將seq label 變成windows_labels
        windows_labels = []
        # print('datasetindex',datasetIndex)
        for window_column_index in afterprocessing_datasetIndex:
            window_column_index_loc= self.original_labels.index.get_loc(window_column_index)
            windows_labels.append(list(np.int_(
                labels[window_column_index_loc-self.sliding_window_size+1:window_column_index_loc+1])))
            # print("index_loc",window_column_index_loc,"index",window_column_index,"windows_labels",windows_labels[-1])
        # 這邊就是windows裡面只要有一個是anomaly整段windows都標記成anomaly
        y_True = [True if (np.sum(window) > 0)
                    else False for window in windows_labels]

        y_True = np.array(y_True)
        return pd.Series(index = self.afterprocessing_datasetIndex,data = y_True)
    def getLabel(self):
        return self.y_True

