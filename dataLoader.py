import pandas as pd
import numpy as np
import sktime
from sktime.utils.data_io import load_from_tsfile_to_dataframe
import h5py

# datasetDirBase = "/workspace/anomaly_detection/"
datasetDirBase = "/dataset/"

class datasetSpecificeLoader:
    def __init__(self,attackFeatureInfo_csv_path=None):
        self.attackFeatureInfo_csv_path= attackFeatureInfo_csv_path
    def loadNormalData(self):
        raise NotImplementedError()
    def loadAnomalyData(self):
        raise NotImplementedError()
    def has_anomaly_info(self):
        if self.attackFeatureInfo_csv_path != None:
            return True
        else:
            return False
    def get_anomaly_info_path(self):
        return self.attackFeatureInfo_csv_path
        
class SWAT_dataset_Loader(datasetSpecificeLoader):
    def __init__(self):
        attackFeatureInfo_csv_path= datasetDirBase + "dataset/SWAT/List_of_attacks_Final.csv"
        self.normal_data_path = datasetDirBase + "dataset/SWAT/SWaT_Dataset_Normal_v1.csv"
        self.anomaly_data_path = datasetDirBase + "dataset/SWAT/SWaT_Dataset_Attack_v0.csv"
        super().__init__(attackFeatureInfo_csv_path)

    def loadNormalData(self):
        data_path = self.normal_data_path
        labelColName = "Normal/Attack"
        timeColName = "Timestamp"
        normal = pd.read_csv(data_path)#, nrows=1000)
        # normal = normal.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
        normal = normal.drop([ labelColName] , axis = 1)
        normal[timeColName] = normal[timeColName].str.strip()
        normal["Timestamp"] = pd.to_datetime(normal[timeColName],format="%d/%m/%Y %I:%M:%S %p")
        normal.set_index("Timestamp",inplace=True)
        print("---normal data.shape",normal.shape)
        for i in list(normal): 
            normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
        normal = normal.astype(float)

        return normal,normal.shape[1]

    def loadAnomalyData(self):
        data_path = self.anomaly_data_path
        labelColName = "Normal/Attack"
        timeColName = "Timestamp"
        attack = pd.read_csv(data_path)#, nrows=1000)
        # attack = attack.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
        # labels = [ float(label!= 'Normal' ) for label  in attack[labelColName].values]
        attack[timeColName] = attack[timeColName].str.strip()
        attack[timeColName] = pd.to_datetime(attack[timeColName],format="%d/%m/%Y %I:%M:%S %p")
        attack.set_index(timeColName,inplace=True)
        labels = attack[labelColName].apply( lambda x: x=="Attack")
        attack = attack.drop([labelColName ] , axis = 1)
        for i in list(attack): 
            attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
        # attack = attack.drop(["Timestamp"])
        # plotData([],attack)
        attack = attack.astype(float)
        input_feature_size = attack.shape[1]

        return attack,labels,input_feature_size

class SWATDebug_dataset_loader(SWAT_dataset_Loader):
    def __init__(self):
        super().__init__()
        print("-----SWATDebug dataset loader init --- ")
        self.anomaly_data_path = datasetDirBase + "dataset/SWAT/SWaT_Dataset_Attack_v0_test.csv"
        self.normal_data_path = datasetDirBase + "dataset/SWAT/SWaT_Dataset_Normal_v1_test.csv"
        self.attackFeatureInfo_csv_path = None

class SWAT_P1_dataset_loader(SWAT_dataset_Loader):
    def __init__(self):
        super().__init__()
        print("-----SWAT_P1 dataset loader init --- ")
        self.anomaly_data_path = datasetDirBase + "dataset/SWAT/SWaT_Dataset_Attack_P1.csv"
        self.normal_data_path = datasetDirBase + "dataset/SWAT/SWaT_Dataset_Normal_P1.csv"
        self.attackFeatureInfo_csv_path = None

class SWAT2000_dataset_loader(SWAT_dataset_Loader):
    def __init__(self):
        super().__init__()
        print("-----SWATDebug2000 dataset loader init --- ")
        self.anomaly_data_path = datasetDirBase + "dataset/SWAT/SWaT_Dataset_Attack_v0_test.csv"
        self.normal_data_path = datasetDirBase + "dataset/SWAT/SWaT_Dataset_Normal_2000L.csv"
        self.attackFeatureInfo_csv_path = None




class WADI_dataset_loader(datasetSpecificeLoader):
    def __init__(self):
        super().__init__()
        print("-----WADI dataset loader init --- ")
        self.normal_data_path =datasetDirBase + "dataset/WADI.A1_9_Oct_2017/WADI_normal_pre_2.csv"
        self.anomaly_data_path = datasetDirBase + "dataset/WADI.A1_9_Oct_2017/WADI_Attack_pre.csv"
    def loadNormalData(self):
        data_path = self.normal_data_path
        timeColName = "Timestamp"
        normal = pd.read_csv(data_path)#, nrows=1000)
        # normal = normal.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
        # normal = normal.drop([ labelColName] , axis = 1)
        normal[timeColName] = normal["Date"].astype(str).str.cat(normal["Time"].astype(str).apply(lambda x:  x.replace(".000","") ),sep=' ') 
        normal = normal.drop(["Row","Date","Time" , "Normal/Attack" ] , axis = 1)

        normal[timeColName] = normal[timeColName].str.strip()
        normal[timeColName] = pd.to_datetime(normal[timeColName],format="%d/%m/%Y %I:%M:%S %p")
        normal.set_index(timeColName,inplace=True)

        normal= normal.fillna(0)

        for i in list(normal): 
            normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
        normal = normal.astype(float)

        return normal,normal.shape[1]

    def loadAnomalyData(self):
        data_path = self.anomaly_data_path
        labelColName = "Normal/Attack"
        timeColName = "Timestamp"
        attack = pd.read_csv(data_path)#, nrows=1000)

        attack[timeColName] = attack["Date"].astype(str).str.cat(attack["Time"].astype(str).apply(lambda x:  x.replace(".000","") ),sep=' ') 

        attack[timeColName] = attack[timeColName].str.strip()
        attack[timeColName] = pd.to_datetime(attack[timeColName],format="%d/%m/%Y %I:%M:%S %p")
        attack.set_index(timeColName,inplace=True)
        labels = attack[labelColName].apply( lambda x: x=="Attack")
        attack = attack.fillna(0)
        attack = attack.drop(["Row","Date","Time" , "Normal/Attack" ] , axis = 1)

        for i in list(attack): 
            attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
        # attack = attack.drop(["Timestamp"])
        # plotData([],attack)
        attack = attack.astype(float)
        input_feature_size = attack.shape[1]

        return attack,labels,input_feature_size

class WADIDebug_dataset_loader(WADI_dataset_loader):
    def __init__(self):

        super().__init__()
        print("-----WADIDebug dataset loader init --- ")
        self.normal_data_path =datasetDirBase + "dataset/WADI.A1_9_Oct_2017/WADI_Attack_pre_test.csv"
        self.anomaly_data_path= self.normal_data_path
    



class TS_dataLoader(datasetSpecificeLoader):
    def __init__(self):
        super().__init__()

    def loadData2DataFrame(self,data_path):
        dataset, labels= load_from_tsfile_to_dataframe(data_path)
        # print("train_x\n",train_x.iloc[0,0])
        # print("train_x.shape",train_x.shape)
        tmp_dataset= []
        for index in range(dataset.shape[0]):
            tmp_dataset.append(dataset.iloc[index,0].values)

        timestamps = pd.date_range(start='2021-01-01',periods=dataset.shape[0],freq="S")
        dataset = pd.DataFrame(data=np.array(tmp_dataset),index = timestamps)
        dataset = dataset.astype(float)
        labels = pd.Series(data = labels,index = timestamps)
        labels = labels.apply(lambda x : x== self.normalLabelValue)

        return dataset,labels


    def loadNormalData(self):
        dataset,labels = self.loadData2DataFrame(self.normal_data_path)

        for index in labels.index:
            if labels.loc[index] == False:
                dataset.drop(index,inplace=True)
                # labels.drop(index,inplace=True)

        input_feature_size = dataset.shape[1]
        return dataset,input_feature_size

    def loadAnomalyData(self):
        dataset,labels = self.loadData2DataFrame(self.anomaly_data_path)
        input_feature_size = dataset.shape[1]
        return dataset,labels,input_feature_size

class Chinatown(TS_dataLoader):
    def __init__(self):
        self.anomaly_data_path= datasetDirBase + "dataset/Univariate_ts/Chinatown/Chinatown_TRAIN.ts"
        self.normal_data_path= datasetDirBase + "dataset/Univariate_ts/Chinatown/Chinatown_TEST.ts"
        self.attackFeatureInfo_csv_path= None
        self.normalLabelValue = "1"

class Crop(TS_dataLoader):
    def __init__(self):
        self.anomaly_data_path= datasetDirBase + "dataset/Univariate_ts/Crop/Crop_TEST.ts"
        self.normal_data_path= datasetDirBase + "dataset/Univariate_ts/Crop/Crop_TRAIN.ts"
        self.attackFeatureInfo_csv_path= None
        self.normalLabelValue = "1"
class Wafer(TS_dataLoader):
    def __init__(self):
        self.anomaly_data_path= datasetDirBase + "dataset/Univariate_ts/Wafer/Wafer_TEST.ts"
        self.normal_data_path= datasetDirBase + "dataset/Univariate_ts/Wafer/Wafer_TRAIN.ts"
        self.attackFeatureInfo_csv_path= None
        self.normalLabelValue = "1"
class DistalPhalanxOutlineCorrect(TS_dataLoader):
    def __init__(self):
        self.anomaly_data_path= datasetDirBase + "dataset/Univariate_ts/DistalPhalanxOutlineCorrect/DistalPhalanxOutlineCorrect_TEST.ts"
        self.normal_data_path= datasetDirBase + "dataset/Univariate_ts/DistalPhalanxOutlineCorrect/DistalPhalanxOutlineCorrect_TRAIN.ts"
        self.attackFeatureInfo_csv_path= None
        self.normalLabelValue = "1"

# class PhalangesOutlinesCorrect(TS_dataLoader):
#     def __init__(self):
#         self.anomaly_data_path= datasetDirBase + "dataset/Univariate_ts/PhalangesOutlinesCorrect/PhalangesOutlinesCorrect_TEST.ts"
#         self.normal_data_path= datasetDirBase + "dataset/Univariate_ts/InsectWingbeatSound/InsectWingbeatSound_TRAIN.ts"
#         self.attackFeatureInfo_csv_path= None
#         self.normalLabelValue = "1"
class PSM_dataset_Loader(datasetSpecificeLoader):
    def __init__(self):

        # self.normal_data_path = datasetDirBase + "dataset/PSM/train_test.csv"
        # self.anomaly_data_path =  datasetDirBase + "dataset/PSM/test_test.csv"
        # self.label_path= datasetDirBase + "dataset/PSM/test_label_test.csv"
        self.normal_data_path =  datasetDirBase + "dataset/PSM/train.csv"
        self.anomaly_data_path = datasetDirBase + "dataset/PSM/test.csv"
        self.label_path= datasetDirBase + "dataset/PSM/test_label.csv"
        super().__init__()
        # self.label_path= datasetDirBase + "RANSynCoders/data/test_label.csv"
    def loadAnomalyData(self):
        data_path = self.anomaly_data_path
        timeColName = "timestamp_(min)"
        attack = pd.read_csv(data_path)#, nrows=1000)
        # attack = attack.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
        # labels = [ float(label!= 'Normal' ) for label  in attack[labelColName].values]
        attack[timeColName] = self.get_timestamp(attack[timeColName])

        attack.set_index(timeColName,inplace=True)
        # labels = attack[labelColName].apply( lambda x: x=="Attack")
        labels = self.loadLabel()
        # attack = attack.drop([labelColName ] , axis = 1)
        # for i in list(attack): 
        #     attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
        # attack = attack.drop(["Timestamp"])
        # plotData([],attack)
        attack= attack.fillna(0)
        attack = attack.astype(float)
        input_feature_size = attack.shape[1]
        print('attack',attack)

        return attack,labels,input_feature_size

    def loadNormalData(self):
        data_path = self.normal_data_path
        # labelColName = "Normal/Attack"
        timeColName = "timestamp_(min)"
        normal = pd.read_csv(data_path)#, nrows=1000)
        normal[timeColName] = self.get_timestamp(normal[timeColName])
        normal.set_index(timeColName,inplace=True)
        # for i in list(normal): 
        #     normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
        normal= normal.fillna(0)
        normal = normal.astype(float)
        print("normal",normal)

        return normal,normal.shape[1]
    def get_timestamp(self,origin_timestamp):
        origin_timestamp = origin_timestamp.apply(lambda x:int(x)).tolist()
        # print("origin_timestamp",origin_timestamp)
        timestamp = pd.to_datetime(origin_timestamp, unit='s',
               origin=pd.Timestamp('1960-01-01'))
        # print("timestamp",timestamp)
        return timestamp
    
    def loadLabel(self):
        timeColName = "timestamp_(min)"
        labels = pd.read_csv(self.label_path)
        labels[timeColName] = self.get_timestamp(labels[timeColName])
        # print("--labels",labels)
        labels["label"] = labels["label"].apply( lambda x: x==1)
        
        labels.set_index(timeColName,inplace=True)
        # print("labels",labels)
        return labels


class SMD_dataset_Loader(datasetSpecificeLoader):
    def __init__(self):

        # self.normal_data_path = datasetDirBase + "dataset/PSM/train_test.csv"
        # self.anomaly_data_path =  datasetDirBase + "dataset/PSM/test_test.csv"
        # self.label_path= datasetDirBase + "dataset/PSM/test_label_test.csv"
        self.normal_data_path =  datasetDirBase + "dataset/ServerMachineDataset/train/machine-1-1.txt"
        self.anomaly_data_path=  datasetDirBase + "dataset/ServerMachineDataset/test/machine-1-1.txt"
        self.label_path=  datasetDirBase + "dataset/ServerMachineDataset/test_label/machine-1-1.txt"
        super().__init__()
        # self.label_path= datasetDirBase + "RANSynCoders/data/test_label.csv"
    def loadAnomalyData(self):
        data_path = self.anomaly_data_path
        attack = pd.read_csv(data_path,header=None)#, nrows=1000)
        # attack = ttack.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
        # labels = [ float(label!= 'Normal' ) for label  in attack[labelColName].values]
        TimeStampIndex = self.get_timestamp(attack.index)

        attack.set_index(TimeStampIndex,inplace=True)
        # labels = attack[labelColName].apply( lambda x: x=="Attack")
        labels = self.loadLabel()
        # attack = attack.drop([labelColName ] , axis = 1)
        # for i in list(attack): 
        #     attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
        # attack = attack.drop(["Timestamp"])
        # plotData([],attack)
        attack= attack.fillna(0)
        attack = attack.astype(float)
        input_feature_size = attack.shape[1]
        print('attack',attack)

        return attack,labels,input_feature_size

    def loadNormalData(self):
        data_path = self.normal_data_path
        normal = pd.read_csv(data_path, header=None)
        TimeStampIndex = self.get_timestamp(normal.index)
        normal.set_index(TimeStampIndex,inplace=True)
        # for i in list(normal): 
        #     normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
        normal= normal.fillna(0)
        normal = normal.astype(float)
        print("normal",normal)

        return normal,normal.shape[1]
    def get_timestamp(self,origin_timestamp):
        # print("origin_timestamp",origin_timestamp)
        timestamp = pd.to_datetime(origin_timestamp, unit='s',
               origin=pd.Timestamp('1960-01-01'))
        # print("timestamp",timestamp)
        return timestamp
    
    def loadLabel(self):
        labels = pd.read_csv(self.label_path,header=None,names=["label"])
        TimeStampIndex = self.get_timestamp(labels.index)
        # print("--labels",labels)
        labels["label"] = labels["label"].apply( lambda x: x==1)
        
        labels.set_index(TimeStampIndex,inplace=True)
        # print("labels",labels)
        return labels


class NSL_KDD_dataset_Loader(datasetSpecificeLoader):
    def __init__(self):

        # self.normal_data_path = datasetDirBase + "dataset/PSM/train_test.csv"
        # self.anomaly_data_path =  datasetDirBase + "dataset/PSM/test_test.csv"
        # self.label_path= datasetDirBase + "dataset/PSM/test_label_test.csv"
        self.normal_data_path=  datasetDirBase + "dataset/nsl-kdd/hdf5/train_normal.hdf5"
        self.anomaly_data_path=  datasetDirBase + "dataset/nsl-kdd/hdf5/test.hdf5"
        self.label_path=  datasetDirBase + "dataset/nsl-kdd/hdf5/test.hdf5"
        super().__init__()
        # self.label_path= datasetDirBase + "RANSynCoders/data/test_label.csv"
    def loadNormalData(self):
        normal = self.get_dataset(self.normal_data_path, 'x')
        normal = pd.DataFrame(normal)
        TimeStampIndex = self.get_timestamp(normal.index)
        normal.set_index(TimeStampIndex,inplace=True)
        # for i in list(normal): 
        #     normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
        normal= normal.fillna(0)
        normal = normal.astype(float)
        print("normal",normal)

        return normal,normal.shape[1]
    def loadAnomalyData(self):
        attack = self.get_dataset(self.anomaly_data_path, 'x')
        attack = pd.DataFrame(attack)
        # attack = ttack.drop(["Timestamp" , "Normal/Attack" ] , axis = 1)
        # labels = [ float(label!= 'Normal' ) for label  in attack[labelColName].values]
        TimeStampIndex = self.get_timestamp(attack.index)

        attack.set_index(TimeStampIndex,inplace=True)
        # labels = attack[labelColName].apply( lambda x: x=="Attack")
        labels = self.loadLabel()
        # attack = attack.drop([labelColName ] , axis = 1)
        # for i in list(attack): 
        #     attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
        # attack = attack.drop(["Timestamp"])
        # plotData([],attack)
        attack= attack.fillna(0)
        attack = attack.astype(float)
        input_feature_size = attack.shape[1]
        print('attack',attack)

        return attack,labels,input_feature_size


    def get_dataset(self,filepath, tagname=None):
        with h5py.File(filepath, 'r') as hdf:
            return hdf[tagname][:]

    def get_timestamp(self,origin_timestamp):
        # print("origin_timestamp",origin_timestamp)
        timestamp = pd.to_datetime(origin_timestamp, unit='s',
               origin=pd.Timestamp('1960-01-01'))
        # print("timestamp",timestamp)
        return timestamp
    
    def loadLabel(self):
        labels = self.get_dataset(self.label_path,'y').squeeze()
        print("labels.shape",labels.shape)
        # labels = pd.read_csv(self.label_path,header=None,names=["label"])
        labels = pd.DataFrame(data = labels,columns = ["label"])
        TimeStampIndex = self.get_timestamp(labels.index)
        # print("--labels",labels)
        labels["label"] = labels["label"].apply( lambda x: x==1)
        
        labels.set_index(TimeStampIndex,inplace=True)
        # print("labels",labels)
        return labels

    
