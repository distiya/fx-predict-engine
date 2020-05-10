import wget
import os
import json
import numpy as np
from predict_pb2 import MultiPredictBatchForGranularity
from predict_pb2 import MultiPredictedCandleForGranularity
from predict_pb2 import Candle
from predict_pb2 import PredictedCandle
from tensorflow.keras.models import load_model
from sklearn.externals import joblib

class ForexPredictor:
    def __init__(self,dataPathEnv=None,appConfigEnv=None,batchSize=48,features=4):
        self.batchSize = batchSize
        self.features = features
        self.dataPath = self.loadEnvironmentVariable(dataPathEnv)
        print("DataPath environment variable loaded\n")
        self.appConfigFile = self.dataPath+"/appconfig.json"
        self.downloadFile(self.loadEnvironmentVariable(appConfigEnv),self.appConfigFile)
        print("AppConfig file downloaded\n")
        self.appConfig = self.loadJsonFile(self.appConfigFile)
        self.downloadAllModelsAndScallers(self.dataPath,self.appConfig)
        print("Dowmloading and loading all models and scallers completed\n")
        
    def downloadFile(self,link=None,destination=None,overwrite=True):
        if destination is not None and overwrite and os.path.exists(destination):
            os.remove(destination)
        if destination is not None and link is not None:
            wget.download(link,destination)
            
    def loadEnvironmentVariable(self,envName=None):
        return os.environ.get(envName)
    
    def loadJsonFile(self,file=None):
        with open(file) as f:
            data = json.load(f)
            return data
    
    def getSaveFileLocation(self,basePath=None,ticker=None,granularity=None,forModel=False,forScaler=False):
        if basePath is not None and ticker is not None and granularity is not None:
            if forModel:
                return basePath + "/model-"+ticker+"-"+granularity+".h5"
            elif forScaler:
                return basePath + "/scaller-"+ticker+"-"+granularity+".sav"
        return None
    
    def loadModel(self,modelFile):
        return load_model(modelFile)
    
    def loadScaller(self,scallerFileName):
        return joblib.load(scallerFileName)

    def getPredictionForBatch(self,request):
        predictedCandles = []
        for pb in request.batches:
            inputData = []
            for c in pb.candles:
                inputData.append(c.open)
                inputData.append(c.close)
                inputData.append(c.high)
                inputData.append(c.low)
            p = self.predictForBatch(inputData,pb.ticker,request.granularity)
            print("Prediction for {t} in granulatiy {g} is ==> O:{o},C:{c},H:{h},L:{l}\n".format(t=pb.ticker,g=request.granularity,o=p[0],c=p[1],h=p[2],l=p[3]))
            predictedCandles.append(PredictedCandle(ticker=pb.ticker,predicted=Candle(open=p[0],close=p[1],high=p[2],low=p[3])))
        return MultiPredictedCandleForGranularity(granularity=request.granularity,candles=predictedCandles)
    
    def predictForBatch(self,inputData=None,pair=None,granularity=None):
        if inputData is not None and pair is not None and granularity is not None and len(inputData) == (self.batchSize*self.features):
            return self.appConfig[granularity][pair]["predictScaller"].inverse_transform(self.appConfig[granularity][pair]["predictModel"].predict(self.appConfig[granularity][pair]["predictScaller"].transform(np.array(inputData).reshape(self.batchSize,self.features)).reshape(1,self.batchSize,self.features))).flatten()
        return None
    
    def downloadAllModelsAndScallers(self,basePath=None,appConfig=None):
        if basePath is not None and appConfig is not None:
            for granularity in appConfig:
                granularityConfig = appConfig[granularity]
                for pair in granularityConfig:
                    pairConfig = granularityConfig[pair]
                    modelPath = self.getSaveFileLocation(basePath,pair,granularity,True,False)
                    scallerPath = self.getSaveFileLocation(basePath,pair,granularity,False,True)
                    self.downloadFile(pairConfig["model"],modelPath)
                    self.downloadFile(pairConfig["scaller"],scallerPath)
                    appConfig[granularity][pair]["predictModel"] = self.loadModel(modelPath)
                    appConfig[granularity][pair]["predictScaller"] = self.loadScaller(scallerPath)