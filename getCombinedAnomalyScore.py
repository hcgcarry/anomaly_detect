from datetime import date
import numpy as np
import json
import argparse
import os
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd

result_dir = "result/combined/"
def log(msg):
    with open(result_dir+"result.txt","a") as f:
        print(msg,file = f)

def gethydraInfo(file_path):
    file_path += ".hydra/config.yaml"
    log("hydra"+file_path)
    with open(file_path,"r") as f:
        content =  f.read()
        log(content)
    

def getAnomalyScore(file_path):
    anomaly_score_file_path = file_path +"anomaly_score.csv"

    y_anomaly_score = pd.read_csv(anomaly_score_file_path ,index_col=0)
    # print("y_anomaly_score index",y_anomaly_score.index)
    y_anomaly_score = y_anomaly_score.astype(float)
    return y_anomaly_score
def getLabel(file_path):
    # file_path = "/model/"+datasetName+"/"+modelName+"/anomalyScore.txt"
    label_file_path= file_path +"label.csv"
    label = pd.read_csv(label_file_path,index_col=0)
    # print("label index",label.index)
    return label


def getAnoamlyScore_filePath():
    dirPath = "outputs/"
    dateDir = os.listdir(dirPath)
    # print("dateDir",dateDir)
    dateDir = sorted(dateDir,key = lambda s: tuple(s.split("-")))
    # print("dateDir sort",dateDir)
    dateDir = dateDir[-1]
    dirPath += dateDir
    secondDir= os.listdir(dirPath)
    # print("secondDir",secondDir)
    targetSecondDir= sorted(secondDir,key = lambda s: tuple(s.split("-")))
    result =  []
    result.append(dirPath + "/"+targetSecondDir[-1]+"/")
    result.append(dirPath + "/"+targetSecondDir[-3]+"/")

    # print("result",result)
    return result

    
def getThreshold(y_test, y_pred):
    # print("y_test",y_test)
    # print("y_pred",y_pred)
    fpr, tpr, tr = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    idx = np.argwhere(np.diff(np.sign(tpr-(1-fpr)))).flatten()[0]

    return tr[idx]
    

def preProcessAnomalyScore(score1,score2,label):
    if score1.shape[0] > score2.shape[0]:
        score1 = score1.iloc[-score2.shape[0]:]
    if score1.shape[0] < score2.shape[0]:
        score2 = score2.iloc[-score1.shape[0]:]
    # print("score1.shape",score1.shape,"score2.shape",score2.shape)
    label = label.iloc[-score1.shape[0]:]
    score1.index = label.index
    score2.index = label.index
    return score1,score2,label

def evaluateResult(y_True, y_pred_anomaly):
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

    recall = float(TP/(TP+FN))
    precision = float(TP/(TP+FP))
    if recall == 0 and precision ==0:
        return
    with open("result/combined/result.txt", 'a') as resultFile:
        print("-------------------", file=resultFile)
        print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN, file=resultFile)
        print("precision:", precision, file=resultFile)
        print("recall:", recall, file=resultFile)
        print("F1 score", 2*precision*recall /
              (precision+recall), file=resultFile)
        print("TPR", TP/(TP+FN), file=resultFile)
        print("FPR", FP/(TN+FP), file=resultFile)
        print("-------------------", file=resultFile)

def getCombinedAnomalyScore(filePathList):
    anomaly_score_list = []
    for filePath in filePathList:
        anomaly_score_list.append(getAnomalyScore(filePath))
        gethydraInfo(filePath)
        label = getLabel(filePath)
    # np.array(anomaly_score_list)
    # print("anomaly_score_list.shape",np.array(anomaly_score_list).shape)
    # print("anomaly_score_list",anomaly_score_list)
    anomaly_score_1 = anomaly_score_list[0]
    anomaly_score_2 = anomaly_score_list[1]
    anomaly_score_1,anomaly_score_2 ,label= preProcessAnomalyScore(anomaly_score_1,anomaly_score_2,label)
    
    # print("anomaly_score_1",anomaly_score_1)

    threshold1 = getThreshold(label.values.flatten(),anomaly_score_1.values.flatten())
    threshold2 = getThreshold(label.values.flatten(),anomaly_score_2.values.flatten())

    y_pred_anomaly_1 = anomalyScore2anomaly(anomaly_score_1, threshold1)
    y_pred_anomaly_2 = anomalyScore2anomaly(anomaly_score_2, threshold2)

    combined_pred_anomaly= combinedAnd(y_pred_anomaly_1,y_pred_anomaly_2)
    evaluateResult(label.values,combined_pred_anomaly.values)
    # combined_pred_anomaly= combinedOr(y_pred_anomaly_1,y_pred_anomaly_2)

    # print("anomaly_score_list",anomaly_score_list)
    # print("y_pred_anomaly_1",y_pred_anomaly_1)
    # print("y_pred_anomaly_2",y_pred_anomaly_2)
    # print("combinedScore",combined_pred_anomaly)
    
    plotAnomalyScore(anomaly_score_1 ,anomaly_score_2,threshold1,threshold2,y_pred_anomaly_1,y_pred_anomaly_2,combined_pred_anomaly,label)
    return combined_pred_anomaly

def anomalyScore2anomaly(anomalyScore, threshold):
    anomaly = anomalyScore.apply(lambda x:x>=threshold)
    return anomaly

def plotAnomalyScore(anomaly_score_1,anomaly_score_2,threshold1,threshold2,y_pred_anomaly_1,y_pred_anomaly_2, combined_pred_anomaly,label):
    plt.figure(figsize=(100, 10))

    plt.plot(label.index, anomaly_score_1,'b' , label="score_1")
    plt.plot(label.index, anomaly_score_2, 'g', label="score_2")
    plt.plot(label.index, y_pred_anomaly_1,'r' , label="pred_1")
    plt.plot(label.index, y_pred_anomaly_2,'c' , label="pred_2")
    plt.plot(label.index, combined_pred_anomaly, 'm' ,label="combined_pred")
    plt.plot(label.index, label, 'y', label="ground_true")
    plt.axhline(threshold1,label = "threshold1",color='r')
    plt.axhline(threshold2,label="threshold2",color ='b')

    plt.ylabel("anomaly_score")
    plt.xlabel("time")
    plt.legend()
    plt.savefig("result/combined/anomalyScore")
    plt.close()

def combinedOr(anomaly_score_1,anomaly_score_2):
    
    return anomaly_score_1 | anomaly_score_2
def combinedAnd(anomaly_score_1,anomaly_score_2):
    return anomaly_score_1 & anomaly_score_2


if __name__ == "__main__":
    log("-------------------------------------------------new")
    filePathList =  getAnoamlyScore_filePath()
    getCombinedAnomalyScore(filePathList)



