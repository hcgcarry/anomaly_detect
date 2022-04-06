import os
from pathlib import Path
import pandas as pd

def createDirectory(model_name_list,dataset_name_list):

    for modelName in model_name_list:
        for datasetName in dataset_name_list:
            dirpath ="./result/"+datasetName+"/"+ modelName
            print("dirPath",dirpath)
            os.makedirs(dirpath,exist_ok=True)
            dirpath ="./model/"+datasetName+"/"+ modelName
            print("dirPath",dirpath)
            os.makedirs(dirpath,exist_ok=True)
            dirpath ="./model/threshold_select/"+datasetName+"/"+ modelName
            print("dirPath",dirpath)
            os.makedirs(dirpath,exist_ok=True)

def create_all_results_csv(model_name_list,dataset_name_list,file_path):
    df = pd.DataFrame(columns =model_name_list,index = dataset_name_list)
    df.index.name = "dataset"
    print(df)
    df.to_csv(file_path)



if __name__ == "__main__":
    import sys
    model_name_list = ["GDN","RANCoders","DAGMM","MEMAE","Autoencoder","CNN_LSTM","LSTM_USAD","LSTM_VAE","USAD","OutlierDetector"]
    dataset_name_list = ["NSL_KDD","SMD","SWAT_P1","PSM","SWAT2000","SWAT","WADI","SWATDebug","WADIDebug","Chinatown"]
    createDirectory(model_name_list,dataset_name_list)
    model_name_list = ["GDN","RANCoders","DAGMM","MEMAE","Autoencoder","CNN_LSTM","LSTM_VAE","USAD"]
    dataset_name_list = ["NSL_KDD","SMD","SWAT_P1","PSM","SWAT","WADI","SWATDebug","WADIDebug"]
    from os.path import exists
    if len(sys.argv) == 2:
        if sys.argv[1] == "all_results":
            create_all_results_csv(model_name_list,dataset_name_list,"result/total/all_results.csv")
        if sys.argv[1] == "f1":
            create_all_results_csv(model_name_list,dataset_name_list,"result/total/f1_precision_recall_results.csv")