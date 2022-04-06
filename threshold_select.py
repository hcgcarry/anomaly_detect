from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from utils import evaluateResult, resultLog
from hydra.utils import get_original_cwd, to_absolute_path
import os
import matplotlib.pyplot as plt
import pandas as pd

from src.adtk.transformer._transformer_hd import *
from utils import *


class threshold_select_class:
    def __init__(self,cfg,datasetIndex):
        self.datasetIndex = datasetIndex
        self.datasetName = cfg.dataset.name
        self.modelName = cfg.model.name

    def threshold_select(self):
        pass
    def ROC(self,y_True, y_pred):
        if y_True.shape[0] != y_pred.shape[0]:
            print("!!!!!!!!!!y_True.sahpe",y_True.shape,"y_pred.shape",y_pred.shape,"mismatch")
            y_True = y_True[-y_pred.shape[0]:]
        fpr, tpr, tr = roc_curve(y_True, y_pred)
        auc = roc_auc_score(y_True, y_pred)
        return auc


    def evaluation(self,y_pred,y_True):
        # assert(self.y_pred is not None and self.y_True is not None)
        print("================= threshold select evaluation start=====================")
        self.y_pred = y_pred
        self.y_True = y_True
        threshold_list = self.threshold_select()
        max_f1_score = 0
        max_f1_score_precision = 0
        max_f1_score_recall = 0
        max_f1_score_threshold = threshold_list[0]
        max_f1_score_TP=0
        max_f1_score_TN=0
        max_f1_score_FP=0
        max_f1_score_FN=0
        for threshold_i in threshold_list:
            precision,recall ,f1Score,TP,TN,FP,FN = self.cacualteF1scorePrecisionRecall(threshold_i)
            resultLog("threshold:{}\n,precision:{},recall:{},f1Score:{},TP:{},TN{},FP{},FN{}".format(threshold_i,precision,recall,f1Score,TP,TN,FP,FN))
            if f1Score is not None and max_f1_score < f1Score:
                max_f1_score = f1Score
                max_f1_score_precision = precision
                max_f1_score_recall = recall
                max_f1_score_threshold = threshold_i
                max_f1_score_TP = TP
                max_f1_score_TN = TN
                max_f1_score_FP = FP
                max_f1_score_FN = FN
        return max_f1_score_threshold,max_f1_score_precision,max_f1_score_recall,max_f1_score,max_f1_score_TP,max_f1_score_TN,max_f1_score_FP,max_f1_score_FN
    def run(self,y_pred,y_True):
        threshold,precision,recall,f1,TP,TN,FP,FN=  self.evaluation(y_pred,y_True)
        auc = self.ROC(y_True,y_pred)
        self.saveThresholdResult(threshold,precision,recall,f1,TP,TN,FP,FN,auc)

        


            
    def saveThresholdResult(self,max_f1_score_threshold,max_f1_score_precision,max_f1_score_recall,max_f1_score,max_f1_score_TP,max_f1_score_TN,max_f1_score_FP,max_f1_score_FN,auc):
        self.plotAnomalyScore(self.datasetIndex,max_f1_score_threshold)
        print("-------thresholdSelect")
        resultLog("=======max f1 score :\nthreshold:{}\nauc{:.2f}\nprecision:{:.2f}\nrecall:{:.2f}\nf1Score:{:.2f},\nTP:{},TN:{},FP:{},FN:{}\n============="
        .format(max_f1_score_threshold,auc,max_f1_score_precision,max_f1_score_recall,max_f1_score,max_f1_score_TP,
        max_f1_score_TN,max_f1_score_FP,max_f1_score_FN),True)

        resultCsvPath = get_original_cwd()+"/result/total/all_results.csv"
        df = pd.read_csv(resultCsvPath,index_col = "dataset")
        resultStr ="config:{}\nauc:{:.2f}\nprecision:{:.2f}\nrecall:{:.2f}\nf1Score:{:.2f},\nTP:{},TN:{},FP:{},FN:{}".format(os.getcwd(),auc,max_f1_score_precision,max_f1_score_recall,max_f1_score,max_f1_score_TP,max_f1_score_TN,max_f1_score_FP,max_f1_score_FN)
        df[self.modelName].loc[self.datasetName] = resultStr
        df.to_csv(resultCsvPath)

        resultCsvPath = get_original_cwd()+"/result/total/f1_precision_recall_results.csv"
        df = pd.read_csv(resultCsvPath,index_col = "dataset")
        resultStr ="auc:{:.2f}\nprecision:{:.2f}\nrecall:{:.2f}\nf1Score:{:.2f}".format(auc,max_f1_score_precision,max_f1_score_recall,max_f1_score)
        df[self.modelName].loc[self.datasetName] = resultStr
        df.to_csv(resultCsvPath)
        print("\n================= threshold select evaluation end=====================")
            
            
        
    def cacualteF1scorePrecisionRecall(self, threshold):
        # print("threshold",threshold)
        print("threshold.shape",threshold.shape)
        y_pred_anomaly = [1 if(x >= threshold[index]) else 0 for index,x in enumerate(self.y_pred)]
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for index, item in enumerate(y_pred_anomaly):
            if y_pred_anomaly[index] == 1 and self.y_True[index] == 1:
                TP += 1
            elif y_pred_anomaly[index] == 0 and self.y_True[index] == 0:
                TN += 1
            elif y_pred_anomaly[index] == 1 and self.y_True[index] == 0:
                FP += 1
            elif y_pred_anomaly[index] == 0 and self.y_True[index] == 1:
                FN += 1

        print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)
        if TP+FP == 0 or TP+FN ==0:
            print("TP+FP==0 or TP+FN == 0")
            return None,None,None,None,None,None,None
        recall = float(TP/(TP+FN))
        precision = float(TP/(TP+FP))
        if recall == 0 and precision ==0:
            print("recall == 0 ,precision ==0")
            return None,None,None,None,None,None,None
        f1score = 2*precision*recall / (precision+recall)
        # print("-------result------------")
        # print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)
        # print("precision:", precision)
        # print("recall:", recall)
        # print("F1 score", f1score)
        # print("TPR", TP/(TP+FN))
        # print("FPR", FP/(TN+FP))
        # print("-------------------")
        # with open("result.txt", 'a') as resultFile:
        #     print("-------------------", file=resultFile)
        #     print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN, file=resultFile)
        #     print("precision:", precision, file=resultFile)
        #     print("recall:", recall, file=resultFile)
        #     print("F1 score", f1score , file=resultFile)
        #     print("TPR", TP/(TP+FN), file=resultFile)
        #     print("FPR", FP/(TN+FP), file=resultFile)
        #     print("threshold",threshold)
        #     print("-------------------", file=resultFile)
        return precision,recall ,f1score,TP,TN,FP,FN

    def anomalyScore2anomaly(self,anomalyScore, threshold):
        anomaly = [x >= threshold for x in anomalyScore]
        return anomaly

    def plotAnomalyScore(self, dataset_index, threshold):
        plt.figure(figsize=(100, 10))
        index = dataset_index
        plt.plot(index, self.y_pred, 'b', label="pred")
        plt.plot(index, self.y_True, 'r', label="ground_true")
        plt.plot(index, threshold, 'g', label="threshold")
        plt.ylabel("anomaly_score")
        plt.xlabel("time")
        plt.legend()
        plt.savefig("anomalyScore")
        plt.close()
    def threshold_base_2_threshold_list(self,threshold_base):
        print("threshold_base",threshold_base)
        print('*****threshold_base.shape',threshold_base.shape)
        y_pred_stddev = np.std(self.y_pred).item()
        print("y_pred_stddev", y_pred_stddev)
        step = y_pred_stddev/8 + 1e-4
        max_of_step = 30
        threshold_range = np.arange(y_pred_stddev/2,max_of_step*step,step)
        threshold_list = np.array([ np.full((self.y_pred.shape[0]),x) + threshold_base for x in threshold_range])
        print('*****threshold_list',threshold_list)
        
        return threshold_list


class bruteForce_threshold_select(threshold_select_class):
    def __init__(self,cfg,datasetIndex):
        super().__init__(cfg,datasetIndex)

    def threshold_select(self):
        y_pred_median = np.mean(self.y_pred).item()
        y_pred_median = np.full(self.y_pred.shape[0],y_pred_median)
        threshold_list = self.threshold_base_2_threshold_list(y_pred_median)
        
        return threshold_list
    


class normal_threshold_select(threshold_select_class):
    def __init__(self,cfg,datasetIndex):
        super().__init__(cfg,datasetIndex)

    def getThreshold(self,y_pred,y_True,TPR_start = 90):

        from collections import Counter
        from sklearn import metrics
        atk, nrm = Counter(y_True)[1], Counter(y_True)[0]

        # make figure
        sns.set(style='whitegrid', rc={"grid.linewidth": 0.5, 'grid.linestyle': '--'})
        plt.figure(dpi=80)

        # ROC curve
        fpr, tpr, threshold = metrics.roc_curve(y_True, y_pred, drop_intermediate=False)
        plt.plot(fpr, tpr, linestyle='-')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='_nolegend_')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(linestyle='--')
        plt.savefig("ROC")

        # AUC
        print("AUC:", metrics.auc(fpr, tpr))
        # performance table
        fpr_, tpr_, thr_ = fpr, tpr, threshold
        tpr, fpr, acc, rec, pre, spe, f1, thr = [list() for i in range(8)]
        for r in range(TPR_start, 100):
            r *= 0.01
            # print("tpr_",tpr_)
            # print("fpr_",fpr_)
            # print("len of tpr_",len(tpr_))
            # print("1np.where",np.where(tpr_>=r))
            # print("1np.where",np.where(tpr_>=r)[0])

            # the index of first element that greater than r :print("np.where",np.where(tpr_>=r)[0][0])

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
        df = df.sort_values(by=['F1-score'])
    
        from IPython.display import display
        display(df)
        print("df",df)
        return df.iloc[-1]['Threshold']
        

    def threshold_2_result(self,y_pred, y_True,threshold):
        # threshold = 0.058
        auc = roc_auc_score(y_True, y_pred)
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
        f1_score = (2*precision*recall / (precision+recall + 0.001))
        print("-------final result------------")
        print("threshold: ",threshold)
        print("auc:",auc)
        print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)
        print("precision:", precision)
        print("recall:", recall)
        print("F1 score", f1_score )
        print("TPR", TP/(TP+FN))
        print("FPR", FP/(TN+FP))
        print("-------------------")
        return threshold,f1_score,precision,recall,TP,TN,FP,FN

    def evaluation(self, y_pred, y_True):
        self.y_pred = y_pred
        self.y_True = y_True
        threshold = self.getThreshold(y_pred,y_True)
        threshold,f1_score,precision,recall,TP,TN,FP,FN = self.threshold_2_result(y_pred,y_True,threshold)
        threshold = self.singleThreshol2_array(threshold)
        return threshold ,f1_score,precision ,recall ,TP,TN,FP,FN
    def singleThreshol2_array(self,threshold):
        return np.full(self.y_pred.shape[0],threshold)
        

# class ML_dynamic_threshold_select(threshold_select_class):
#     def __init__(self,cfg,datasetIndex):
#         super().__init__(cfg,datasetIndex)
#         self.epoch = 50
#         self.batch_size = 7000
#         self.window_size = 1
#         self.datasetIndex = datasetIndex
#         self.datasetName = cfg.dataset.name
#         self.threshold_select_modelName = cfg.threshold_select.model
#         self.input_feature_size  = 1
#         self.saveModelPath = get_original_cwd()+"/model/threshold_select/" + self.datasetName+"/"+self.threshold_select_modelName+"/model.pth"
#     def data_preprocessing(self,loss_data):
#         loss_data = np.array(loss_data)
#         loss_data = loss_data.reshape(loss_data.shape[0],-1)
#         loss_data = pd.DataFrame(loss_data,index = self.datasetIndex)
#         print("before loss_data",loss_data)
#         loss_data = normalization().transform(loss_data)
#         print("loss_data",loss_data)
#         return loss_data

#     def train(self,loss_data):
#         print("======ML_dynamic_threshold_select train start======")
#         self.get_ML_model_class_obj = get_ML_Model_class(self.epoch,self.threshold_select_modelName,self.batch_size,self.window_size,self.datasetName,self.input_feature_size)
#         self.Detector_obj,detectorType= self.get_ML_model_class_obj.get_dectector()
#         loss_data = self.data_preprocessing(loss_data)
#         self.Detector_obj.getModel().setSaveModelPath(self.saveModelPath)
#         self.Detector_obj.fit(loss_data)
#         print("======ML_dynamic_threshold_select train end======")
#     def evaluation(self,y_pred,y_True):
#         self.test(y_pred,y_True)
#         return super().evaluation(y_pred,y_True)
#     def test(self,loss_data,y_True):
#         print("====== ML_dynamic_threshold_select test start======")
#         self.get_ML_model_class_obj = get_ML_Model_class(self.epoch,self.threshold_select_modelName,self.batch_size,self.window_size,self.datasetName,self.input_feature_size)
#         self.Detector_obj,detectorType= self.get_ML_model_class_obj.get_dectector(y_True)
#         loss_data = self.data_preprocessing(loss_data)
#         self.Detector_obj.getModel().setSaveModelPath(self.saveModelPath)
#         self.Detector_obj.detect(loss_data)
#         self.anomaly_score_reconstruct = self.Detector_obj.getModel().get_input_output_window_result()["output"]
#         print("anomaly_score_reconstruct",self.anomaly_score_reconstruct)
#         print("====== ML_dynamic_threshold_select test end======")
        

#     def threshold_select(self):
#         threshold_list = self.threshold_base_2_threshold_list(self.anomaly_score_reconstruct.values.flatten())
#         return threshold_list


# if __name__== "__main__":
#     model_dynamic_threshold_select

    
        

def get_threshold_select_obj(cfg,datasetIndex):
    # if cfg.threshold_select.name == "ML_dynamic":
    #     return ML_dynamic_threshold_select(cfg,datasetIndex)
    if cfg.threshold_select.name == "brute_force":
        return bruteForce_threshold_select(cfg,datasetIndex)
    if cfg.threshold_select.name == "normal":
        return normal_threshold_select(cfg,datasetIndex)