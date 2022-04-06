
from types import DynamicClassAttribute
import pickle
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



from torch.optim import optimizer

from utils import *
device = get_default_device()

class normal_model(nn.Module):
    def __init__(self, cfg, modelName,  input_feature_dim, window_labels=None):
        super().__init__()
        self.cfg = cfg
        self.input_feature_dim = input_feature_dim
        self.datasetName = cfg.dataset.name
        self.batch_size = cfg.hyperParam.batch_size
        self.window_size = cfg.dataPreprocessing.sliding_window.window_size
        self.input_output_windows_result = {
            "input": [], "output": [], "loss": []}
        self.input_output_sliding_windows_result = {
            "input": [], "output": [], "loss": []}
        self.window_labels = window_labels
        self.modelName = modelName
        self.epoch = 1
        self.target_epochs = cfg.hyperParam.epochs

    def clean(self):
        self.input_output_windows_result = {
            "input": [], "output": [], "loss": []}

    ################################ training ##########################

    def training_all(self, dataset) -> pd.DataFrame:
        print("============="+self.modelName+" start training==============")
        train_loader, val_loader = self.preProcessTrainingData(dataset)
        # self.train(True)
        self.train(True)
        opt_func=torch.optim.Adam
        optimizer1 = opt_func(list(self.parameters()),lr=0.001)
        # print("self paramter",list(self.encoder.parameters())+list(self.decoder.parameters()))
        self.count=0
        epoch_loss = []
        for epoch in range(self.target_epochs):
            self.epoch +=1
            for [batch] in train_loader:
                self.count+=1
                batch = to_device(batch, device)

                # Train AE1
                loss1 = self.training_step(batch)
                loss1.backward()
                optimizer1.step()
                optimizer1.zero_grad()
                if epoch == self.target_epochs-1:
                    epoch_loss.append(loss1.item())

            result = self.evaluate(val_loader, epoch+1)
            self.epoch_end(epoch, result)
        epoch_loss = np.asarray(epoch_loss)
        print("last_epoch_loss_list",epoch_loss)
        print("MSE epoch_loss:",np.mean(epoch_loss))

        self.saveModel()
        print("=================="+self.modelName+" end training==============")
        # return history

    def training_step(self, batch):
        loss = self.caculateMSE(batch)
        loss = torch.mean(loss)
        
        return loss

    def preProcessTrainingData(self, window_dataset: pd.DataFrame):
        # create sliding windows and convert dataset to dataloader
        # DataProcessingObj =  DataProcessing()
        # windows_dataset=DataProcessingObj.seq2Window(dataset,self.windows_size)
        # training_windows_dataset,validation_windows_dataset = DataProcessingObj.divideTrainingAndValidation(dataset)
        training_windows_dataset = window_dataset.iloc[:int(
            np.floor(.8 * window_dataset.shape[0]))]
        validation_windows_dataset = window_dataset.iloc[int(
            np.floor(.8 * window_dataset.shape[0])):int(np.floor(window_dataset.shape[0]))]
        print("split training to train and validation")
        print("training_windows_dataset.shape", training_windows_dataset.shape)

        train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(training_windows_dataset.values).float()
        ), batch_size=self.batch_size, shuffle=False, num_workers=self.cfg.hyperParam.worker)

        val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(validation_windows_dataset.values).float()
        ), batch_size=self.batch_size, shuffle=False, num_workers=self.cfg.hyperParam.worker)

        return train_loader, val_loader

    def getTrainingHistory(self):
        return self.history
    ######################### testing ####################################

    def preProcessTestingData(self, windows_dataset: pd.DataFrame):
        # create sliding windows and convert dataset to dataloader
        # DataProcessingObj =  DataProcessing()
        self.dataset_window_index = windows_dataset.index
        self.dataset_window_column = windows_dataset.columns
        # windows_dataset=DataProcessingObj.seq2Window(dataset,self.windows_size)
        # print("testing window_dataset.shape",windows_dataset.shape)

        dataset_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_dataset.values).float()
        ), batch_size=self.batch_size, shuffle=False, num_workers=8)
        return dataset_loader

    def get_anomaly_score(self):
        return self.anomaly_score

    def testPreprocessing(self):
        print("================="+self.modelName +
              " start testing=================")
        self.clean()
        self.loadModel()
        self.eval()

    def testing_all(self, dataset: pd.DataFrame) -> pd.DataFrame:
        self.testPreprocessing()
        test_loader = self.preProcessTestingData(dataset)
        count = 0
        results = []
        self.tensor_loss= []
        for [batch] in test_loader:
            count +=1
            print("testing batch iter:", count)
            batch = to_device(batch, device)

            with torch.no_grad():
                # w1,_ = self.encoder(batch)
                # w1=self.decoder(w1)
                loss = self.caculateMSE(batch)
                self.tensor_loss.extend(loss)
                results.extend(loss.cpu().numpy())

            # del w1
            torch.cuda.empty_cache()
        results = self.testPostProcessing(results)
        return results

    def testPostProcessing(self,results):
        results = np.asarray(results)
        self.anomaly_score = results
        # results = self.postProcessingLoss(results)
        # results = self.anomaly_score_2_boolean(results)
        results = self.result2PDSeries(results)
        print("================="+self.modelName +
              " end testing=================")
        return results


    def postProcessingLoss(self, loss):
        loss = self.result2PDSeries(loss)
        loss = ClassicSeasonalDecomposition().fit_transform(loss)
        return loss.values


    def anomaly_score_2_boolean(self, y_pred):
        if self.window_labels is not None:
            threshold = ROC(self.window_labels, y_pred,
                            self.modelName, self.datasetName, False)
            print("threshold", threshold)
        else:
            return []
        return [True if x >= threshold else False for x in y_pred]

    def result2PDSeries(self, results):
        # convert result 2 dataFrame
        return pd.Series(index=self.dataset_window_index, data=results)

    ############################# other ################################

    def validation_step(self, batch, n):
        loss1 = self.training_step(batch)
        return {'val_loss1': loss1}

    def validation_epoch_end(self, outputs):
        batch_losses1 = [x['val_loss1'] for x in outputs]
        epoch_loss1 = torch.stack(batch_losses1).mean()
        return {'val_loss1': epoch_loss1.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss1: {:.4f}".format(
            epoch, result['val_loss1']))

    def evaluate(self, val_loader, n):
        outputs = [self.validation_step(to_device(batch, device), n) for [
            batch] in val_loader]
        return self.validation_epoch_end(outputs)

    def add_result_2_input_output_window_result(self, input, output, loss):
        self.input_output_windows_result["input"].extend(
            input.detach().cpu().numpy())
        self.input_output_windows_result["output"].extend(
            output.detach().cpu().numpy())
        self.input_output_windows_result["loss"].extend(
            loss.detach().cpu().numpy())
    def add_result_2_input_output_sliding_window_result(self, input, output, loss):
        # print("------------------fsdfsfsd--sdf-s---")
        # print("input",input)
        # print("input.size",input.size())
        self.input_output_sliding_windows_result["input"].extend(
            [pd.DataFrame(data = x.detach().cpu().numpy().reshape(-1,self.input_feature_dim)) for x in input])
        self.input_output_sliding_windows_result["output"].extend(
            [pd.DataFrame(data = x.detach().cpu().numpy().reshape(-1,self.input_feature_dim)) for x in output])
        self.input_output_sliding_windows_result["loss"].extend(
            [pd.DataFrame(data = x.detach().cpu().numpy().reshape(-1,self.input_feature_dim)) for x in loss])
        # print("input_window",self.input_output_sliding_windows_result["input"])

    def get_input_output_window_result(self):
        if len(self.input_output_windows_result["input"]) == 0:
            return self.input_output_windows_result
        self.input_output_windows_result["input"] = pd.DataFrame(
            index=self.dataset_window_index, data=np.asarray(self.input_output_windows_result["input"]))
        self.input_output_windows_result["output"] = pd.DataFrame(
            index=self.dataset_window_index, data=np.asarray(self.input_output_windows_result["output"]))
        self.input_output_windows_result["loss"] = pd.DataFrame(
            index=self.dataset_window_index, data=np.asarray(self.input_output_windows_result["loss"]))
        return self.input_output_windows_result

    def get_input_output_sliding_window_result(self):
        if len(self.input_output_sliding_windows_result["input"]) == 0:
            return self.input_output_sliding_windows_result
        return self.input_output_sliding_windows_result

    def caculateMSE(self):
        raise NotImplementedError()

    def saveModel(self):
        
        torch.save(self.state_dict(), get_original_cwd()+"/model/" +
                   self.datasetName+"/"+self.modelName+"/model.pth")
        # self.Convert_ONNX()
    def forward(self,dataset):
        return self.caculateMSE(dataset)

    #Function to Convert to ONNX 
    def Convert_ONNX(self): 

        # set the model to inference mode 
        self.eval() 

        # Let's create a dummy input tensor  
        dummy_input = torch.randn(1, self.input_feature_dim, requires_grad=True).cuda()

        savePath = get_original_cwd()+"/model/" + self.datasetName+"/"+self.modelName+"/model_onnx.pth"
        # Export the model   
        torch.onnx.export(self,         # model being run 
            dummy_input,       # model input (or a tuple for multiple inputs) 
            savePath,
            opset_version=10
        )
            # export_params=True,  # store the trained parameter weights inside the model file 
            # opset_version=10,    # the ONNX version to export the model to 
            # do_constant_folding=True,  # whether to execute constant folding for optimization 
            # input_names = ['modelInput'],   # the model's input names 
            # output_names = ['modelOutput'], # the model's output names 
            # dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
            #                         'modelOutput' : {0 : 'batch_size'}}) 
        # torch.onnx.export(self,         # model being run 
        #     dummy_input,       # model input (or a tuple for multiple inputs) 
        #     savePath,
        #     export_params=True,  # store the trained parameter weights inside the model file 
        #     opset_version=10,    # the ONNX version to export the model to 
        #     do_constant_folding=True,  # whether to execute constant folding for optimization 
        #     input_names = ['modelInput'],   # the model's input names 
        #     output_names = ['modelOutput'], # the model's output names 
        #     dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
        #                             'modelOutput' : {0 : 'batch_size'}}) 

        print(" ") 
        print('Model has been converted to ONNX') 



    def loadModel(self):
        checkpoint = torch.load(
            get_original_cwd()+"/model/"+self.datasetName+"/"+self.modelName+"/model.pth")
        self.load_state_dict(checkpoint)
        checkpoint = torch.load(
            get_original_cwd()+"/model/"+self.datasetName+"/"+self.modelName+"/model.pth")
        self.load_state_dict(checkpoint)