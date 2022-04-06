from ast import parse
import matplotlib.dates as md
from collections import Counter
from sklearn import metrics
from IPython.display import display

import numpy as np
from numpy.lib.function_base import rot90
from pandas.io.parsers import read_csv
import torch.utils.data as data_utils
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Subset
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import torch
from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.metrics import roc_curve, roc_auc_score

import sys

from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import SeasonalAD
import codecs, json 
from hydra.utils import get_original_cwd, to_absolute_path

code_testing_mode = True
    

        
    
def resultLog(message,printToScreen=False):
    if printToScreen == True:
        print(message)
    with open("result.txt","a") as f:
        print(message,file=f)

    

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def plot_history(history, modelName,datasetName):
    losses1 = [x['val_loss1'] for x in history]
    if modelName == "USAD":
        losses2 = [x['val_loss2'] for x in history]
        plt.plot(losses2, '-x', label="loss2")
    plt.plot(losses1, '-x', label="loss1")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Losses vs. No. of epochs')
    plt.grid()
    # plt.savefig("result/"+datasetName+"/"+modelName+"/history")
    plt.savefig("history")


def histogram(y_test, y_pred):
    plt.figure(figsize=(12, 6))
    plt.hist([y_pred[y_test == 0],
              y_pred[y_test == 1]],
             bins=20,
             color=['#82E0AA', '#EC7063'], stacked=True)
    plt.title("Results", size=20)
    plt.grid()
    plt.savefig("histogram")



def ROC(y_True, y_pred, modelName,datasetName,plot_fig=True):
    if y_True.shape[0] != y_pred.shape[0]:
        print("!!!!!!!!!!y_True.sahpe",y_True.shape,"y_pred.shape",y_pred.shape,"mismatch")
        y_True = y_True[-y_pred.shape[0]:]
    fpr, tpr, tr = roc_curve(y_True, y_pred)
    auc = roc_auc_score(y_True, y_pred)
    resultLog("AUC:"+str(auc))
    idx = np.argwhere(np.diff(np.sign(tpr-(1-fpr)))).flatten()[0]

    if plot_fig == True:
        plt.figure()
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.plot(fpr, tpr, label="AUC="+str(auc))
        plt.plot(fpr, 1-fpr, 'r:')
        plt.plot(fpr[idx], tpr[idx], 'ro')
        plt.legend(loc=4)
        plt.grid()
        plt.show()
        # plt.savefig("result/"+datasetName+"/"+modelName+"/ROC")
        plt.savefig("ROC")
        plt.clf()
    return tr[idx]


def printDataInfo(dataset):
    abnormalCount = 0
    normalCount = 0
    for label in dataset["Normal/Attack"]:
        if label == "Normal":
            normalCount += 1
        else:
            abnormalCount += 1
    print("#####data info########")
    print("number of anomaly :", abnormalCount)
    print("number of normal :", normalCount)
    print("################")



def evaluateResult(y_True, y_pred):
    fpr, tpr, tr = roc_curve(y_True, y_pred)
    auc = roc_auc_score(y_True, y_pred)
    idx = np.argwhere(np.diff(np.sign(tpr-(1-fpr)))).flatten()[0]

    threshold = tr[idx]
    # threshold = 0.058
    y_pred_anomaly = [1 if(x >= threshold) else 0 for x in y_pred]
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for index, item in enumerate(y_pred_anomaly):
        if y_pred_anomaly[index] == 1 and y_True[index] == 1:
            TP += 1
        elif y_pred_anomaly[index] == 0 and y_True[index] == 0:
            TN += 1
        elif y_pred_anomaly[index] == 1 and y_True[index] == 0:
            FP += 1
        elif y_pred_anomaly[index] == 0 and y_True[index] == 1:
            FN += 1

    recall = float(TP/(TP+FN+0.001))
    precision = float(TP/(TP+FP+0.001))
    print("-------ROC result------------")
    print("threshold: ",threshold)
    print("auc:",auc)
    print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)
    print("precision:", precision)
    print("recall:", recall)
    print("F1 score", 2*precision*recall / (precision+recall + 0.001))
    print("TPR", TP/(TP+FN))
    print("FPR", FP/(TN+FP))
    print("-------------------")



# def evaluateResult(y_True, y_pred, threshold, modelName,datasetName):
#     y_pred_anomaly = [1 if(x >= threshold) else 0 for x in y_pred]
#     TP = 0
#     TN = 0
#     FP = 0
#     FN = 0
#     for index, item in enumerate(y_pred_anomaly):
#         if y_pred_anomaly[index] == 1 and y_True[index] == 1:
#             TP += 1
#         elif y_pred_anomaly[index] == 0 and y_True[index] == 0:
#             TN += 1
#         elif y_pred_anomaly[index] == 1 and y_True[index] == 0:
#             FP += 1
#         elif y_pred_anomaly[index] == 0 and y_True[index] == 1:
#             FN += 1

#     recall = float(TP/(TP+FN))
#     precision = float(TP/(TP+FP))
#     if recall == 0 and precision ==0:
#         return
#     print("-------result------------")
#     print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)
#     print("precision:", precision)
#     print("recall:", recall)
#     print("F1 score", 2*precision*recall / (precision+recall))
#     print("TPR", TP/(TP+FN))
#     print("FPR", FP/(TN+FP))
#     print("-------------------")
#     with open("result.txt", 'a') as resultFile:
#         print("-------------------", file=resultFile)
#         print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN, file=resultFile)
#         print("precision:", precision, file=resultFile)
#         print("recall:", recall, file=resultFile)
#         print("F1 score", 2*precision*recall /
#               (precision+recall), file=resultFile)
#         print("TPR", TP/(TP+FN), file=resultFile)
#         print("FPR", FP/(TN+FP), file=resultFile)
#         print("-------------------", file=resultFile)




def confusion_matrix(target, predicted, perc=False):

    data = {'y_Actual':    target,
            'y_Predicted': predicted
            }
    df = pd.DataFrame(data, columns=['y_Predicted', 'y_Actual'])
    confusion_matrix = pd.crosstab(df['y_Predicted'], df['y_Actual'], rownames=[
                                   'Predicted'], colnames=['Actual'])

    if perc:
        sns.heatmap(confusion_matrix/np.sum(confusion_matrix),
                    annot=True, fmt='.2%', cmap='Blues')
    else:
        sns.heatmap(confusion_matrix, annot=True, fmt='d')
    plt.savefig("confusion_matrix")






class plotFeature:
    def __init__(self,origin_dataset,modelName,datasetName,window_size,input_feature_dim,y_True):
        self.origin_dataset=origin_dataset
        self.modelName = modelName
        self.datasetName = datasetName
        self.window_size = window_size
        self.input_feature_dim = input_feature_dim
        self.y_True = y_True

    def getAttackFeatureInfo(self,attackFeatureInfo_csv_path):
        df = pd.read_csv(attackFeatureInfo_csv_path)
        df= df.dropna(axis=0,subset=["End Time"])
        date_list = [x.split()[0]+" " for x in df["Start Time"].astype(str)]
        date_list = pd.Series(date_list)
        df["End Time"] = pd.to_datetime((date_list + df["End Time"] ).str.strip(),format="%d/%m/%Y %H:%M:%S")
        df["Start Time"] = pd.to_datetime(df["Start Time"].str.strip(),format="%d/%m/%Y %H:%M:%S")
        # df.drop(df.tail(5).index,inplace=True)
        # print("df",df[["End Time","Start Time"]])
        # print("4 df\n",df[["Start Time","End Time"]])
        return df
    def set_anomaly_time_dict(self,anomaly_info):
        result={}
        for index,row in anomaly_info.iterrows():
            featureNameArray = row["Attack Point"]
            for featureName  in featureNameArray.split(","):
                featureName = featureName.strip().replace("-","")
                result.setdefault(featureName,[]).append((row["Start Time"],row["End Time"]))

        return result
        # print("feature_info_dict\n",self.feature_info_dict)
            
    def get_anomaly_FeatureNameList(self,anoamly_time_dict):
        featureNameList =  [x for x in anoamly_time_dict]
        return featureNameList[:5]

    def dataFrameTime2Index(self,dataFrameTime):
        df = self.origin_dataset.reset_index()
        
        index = df[df["Timestamp"] == dataFrameTime].index
        return index
    def seqIndex2WindowIndex(self,index):
        return index - self.window_size +1

    def get_anomaly_xtick_dict(self,anomaly_time_dict):
        result={}
        for key,values in anomaly_time_dict.items():
            per_Feature_result=[]
            [per_Feature_result.extend(x) for x in values]
            result[key] = per_Feature_result
        return result

    def featureName2index(self,featureName):
        count=0
        for column in self.origin_dataset.columns:
            if column == featureName:
                return count
            else:
                count+=1

    def plot_FeatureAndInput_output_loss(self,featureNameList,input_window,output_window,loss):
        self.plot_input_output_Score(featureNameList,input_window,output_window,loss,"Feature_input_output_loss")

    def plot_anomalyFeature(self,anomaly_info_path,input_window,output_window,loss):
        anomaly_info=self.getAttackFeatureInfo(anomaly_info_path)
        anomaly_time_dict= self.set_anomaly_time_dict(anomaly_info)
        anomaly_featureNameList =  self.get_anomaly_FeatureNameList(anomaly_time_dict)


        anomalyFeatureNameList = anomaly_featureNameList
        self.plot_input_output_Score(self.finalDatasetColumns[:5],input_window,output_window,loss,
        "anomaly_xtickx_Feature_input_output_loss",self.get_anomaly_xtick_dict(anomaly_time_dict))
        self.plot_input_output_Score(anomalyFeatureNameList,input_window,output_window,loss,
        "anomalyFeature_input_output_loss",self.get_anomaly_xtick_dict(anomaly_time_dict))
        print("----------plot_anomalyFeature end---------")
    def plotOriginalFeature(self):
        featureNameList = self.origin_dataset.columns[:5]
        self.plot_Feature(self.origin_dataset,featureNameList,"original_feature")

    def plot_Feature(self,dataset,featureNameList,outputImageName):
        plt.figure(figsize=(100, 10))
        # fig,ax  = plt.subplots(3,1)
        plt.title("features")

        for index, featureName in enumerate(featureNameList):
            ax = plt.subplot(len(featureNameList), 1, index+1)
            # print("plotOriginalData featurename",featureName)
            data = dataset[featureName]
            data = data.astype(float)
            plt.plot(dataset.index,data,label="data",color='b')
            plt.gca().set_title(featureName)

        # plt.savefig("result/"+self.datasetName+"/"+self.modelName+"/"+outputImageName)
        plt.savefig(outputImageName)
        plt.close()



    def plot_input_output_Score(self,featureNameList, input_Features_list,
     output_Features_list, loss_,output_image_name,xticks_dict=None,figure_size=100):
        plt.figure(figsize=(figure_size, 30))

        for indexOfFeatureNameList,featureName in enumerate(featureNameList):
            xticks = None
            # dimIndex = self.featureName2index(featureName) 
            input_Feature_list=input_Features_list[featureName]
            output_Feature_list=output_Features_list[featureName]
            loss = loss_[featureName]
            if xticks_dict != None:
                if featureName in xticks_dict.keys():
                    xticks = xticks_dict[featureName]


            self.plot_input_output_Score_singleFeature(
                 featureNameList,indexOfFeatureNameList, input_Feature_list, output_Feature_list, loss,xticks)

        plt.legend()
        plt.savefig(output_image_name)
        plt.close()

    def plot_input_output_Score_singleFeature(self,featureNameList,indexOfFeatureNameList, inputFeature_list, 
        outputFeature_list ,loss,anomalyTime_xticks=None):
        featureName = featureNameList[indexOfFeatureNameList]
        ax = plt.subplot(len(featureNameList), 1, indexOfFeatureNameList+1)

        plt.plot(inputFeature_list.index, inputFeature_list,label="input",color='b')
        plt.plot(outputFeature_list.index, outputFeature_list,label="output",color='k')
        plt.plot(loss.index, loss ,label="loss",color='r')
        xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
        plt.gca().xaxis.set_major_formatter(xfmt)
        if anomalyTime_xticks != None:
            plt.xticks(anomalyTime_xticks,rotation=270)
        plt.gca().set_title(featureName)

    def writeAnomalyScoreSortByTime(self,loss):
        dataFrameTimeList = []
        for key,value in self.anomaly_time_dict.items():
            for tupleTime in value:
                # print("tupleTime",tupleTime)
                dataFrameTimeList.extend(tupleTime)
        # print("dataFrameTimeList",dataFrameTimeList)

        with open("loss_cause.txt","w") as f:
            for anomalyStartTime in dataFrameTimeList:
                print("-----------anomaly anomalyStartTime",anomalyStartTime,file=f)
                for curDataFramTime in pd.date_range(anomalyStartTime-timedelta(seconds=1),periods=3,freq="S"):
                    # curDataFramTime = anomalyStartTime
                    index = self.seqIndex2WindowIndex(self.dataFrameTime2Index(curDataFramTime))
                    # print("dateFrameTime:",dataFrameTime,"index:",index)
                    print("===time:",curDataFramTime,file=f)
                    # print("loss",loss[index],"origin_data",self.origin_dataset.loc[curDataFramTime])
                    df = pd.DataFrame(index = self.origin_dataset.columns,data = {"loss":np.array(loss[index]).flatten(),"origin_data":self.origin_dataset.loc[curDataFramTime]})
                    df = df.sort_values(by=['loss'],ascending=False).T
                    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
                        print(df,file=f)
                    # print("index of feature which is main cause of anomaly:\n",loss[index].argsort().reverse(),file=f)
                    # print("loss:",dataFrameTime,"\n index of feature which is main cause of anomaly:\n",loss[index].argsort(),file=f)
                    # print("time:",dataFrameTime,"loss:",loss[index].argsort())

    def plot_time_slot_feature(self,dataPreprocessingObj,origin_attack,input_window:pd.DataFrame,output_window:pd.DataFrame,loss:pd.DataFrame):
        self.plotOriginalFeature()
        dataset_loader=  dataPreprocessingObj.get_dataset_loader(self.datasetName)

        if len(input_window) and len(input_window.columns) == len(origin_attack.columns)  :
            input_window.columns = origin_attack.columns
            output_window.columns = origin_attack.columns
            loss.columns = origin_attack.columns
        self.finalDatasetColumns = input_window.columns
        if len(input_window):
            plotFeatureNameList = input_window.columns[:5]
            self.plot_FeatureAndInput_output_loss(plotFeatureNameList,input_window,output_window,loss)
        if len(input_window) and dataset_loader.has_anomaly_info() and len(input_window.columns) == len(origin_attack.columns):
            self.plot_anomalyFeature(dataset_loader.get_anomaly_info_path(),input_window,output_window,loss)

    
    def restore_column_sliding_window_feature(self,input,output,loss,columns):
        input.columns = columns
        output.columns = columns
        loss.columns = columns
        return input ,output ,loss
    def restore_sliding_window_input_output_loss_index(self,input,output,loss,index):
        print("input",input)
        print("index",index)
        input.index= index
        output.index= index
        loss.index= index
        return input ,output ,loss

    def plot_sliding_window_feature(self,origin_attack,input_window,output_window,loss,LabelObj):
        # input_window：會是一個list 裡面的item會是一個dataframe
        # print("input_window",input_window)
        # print("type(input_window)",type(input_window))
        plot_window_index_list = [0]

        for index in plot_window_index_list:
            cur_input_window =  input_window[index]
            cur_output_window =  output_window[index]
            cur_loss =  loss[index]

            if len(cur_input_window) and len(cur_input_window.columns) == len(origin_attack.columns):
                cur_input_window ,cur_output_window,cur_loss = self.restore_column_sliding_window_feature(cur_input_window,cur_output_window,cur_loss,origin_attack.columns)

            cur_input_window ,cur_output_window,cur_loss = self.restore_sliding_window_input_output_loss_index (cur_input_window,cur_output_window,cur_loss,LabelObj.getWindowIndexByWindowLastIndex(LabelObj.afterprocessing_datasetIndex[index]))
            featureNameList = cur_input_window.columns[:5]

            self.plot_input_output_Score(featureNameList,cur_input_window,cur_output_window,cur_loss,"sliding_window_Feature_input_output_loss",figure_size = 30)
        





def make_figure(y_test, error):
    atk, nrm = Counter(y_test)[1], Counter(y_test)[0]

    # make figure
    sns.set(style='whitegrid', rc={"grid.linewidth": 0.5, 'grid.linestyle': '--'})
    plt.figure(dpi=80)

    # ROC curve
    fpr, tpr, threshold = metrics.roc_curve(y_test, error, drop_intermediate=False)
    plt.plot(fpr, tpr, linestyle='-')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='_nolegend_')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(linestyle='--')
    plt.show()
    plt.savefig("ROC")

    # AUC
    print("AUC:", metrics.auc(fpr, tpr))

    # performance table
    fpr_, tpr_, thr_ = fpr, tpr, threshold
    tpr, fpr, acc, rec, pre, spe, f1, thr = [list() for i in range(8)]
    for r in range(90, 100):
        r *= 0.01
        tpr.append(tpr_[np.where(tpr_ >= r)[0][0]])
        fpr.append(fpr_[np.where(tpr_ >= r)[0][0]])
        acc.append((tpr[-1] * atk + (1 - fpr[-1]) * nrm) / (atk + nrm))
        rec.append(tpr[-1])
        pre.append(tpr[-1] * atk / (tpr[-1] * atk + fpr[-1] * nrm))
        spe.append(1 - fpr[-1])
        f1.append(2 * rec[-1] * pre[-1] / (rec[-1] + pre[-1]))
        thr.append(thr_[np.where(tpr_ >= r)[0][0]])
    df = pd.DataFrame({
        'TPR': tpr, 'FPR': fpr, 'Threshold': thr, 'Accuracy': acc,
        'Specifity': spe, 'Precision': pre, 'Recall': rec, 'F1-score': f1
    })
    df = df.round(3)

    # display(df)
    print(df)