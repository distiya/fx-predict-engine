from concurrent import futures
from forex_predict import ForexPredictor
import grpc
import predict_pb2
import predict_pb2_grpc
import threading
import time
import predict_service
import os
class PredictService(predict_pb2_grpc.FxPredictServicer):

    def __init__(self,*args,**kwargs):
        self.predictor = ForexPredictor(dataPathEnv="BASE_PATH",appConfigEnv="APP_CONFIG_FILE_DOWNLOAD_URL")

    def getPredictionForBatch(self, request, context):
        try:
            return self.predictor.getPredictionForBatch(request)
        except Exception as e:
            print("Error in gRPC" + str(e))
            return None

def startServer():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    predict_pb2_grpc.add_FxPredictServicer_to_server(PredictService(),server)
    server.add_insecure_port("[::]:50051")
    server.start()
    try:
        while True:
            print("Server is active")
            time.sleep(10)
    except KeyboardInterrupt:
        print("KeyboardInterupted")
        server.stop(0)
    except Exception as e:
        print("Server stopped due to exception"+str(e))
        server.stop(0)

if __name__ == "__main__":
    # Set values for BASE_PATH and APP_CONFIG_FILE_DOWNLOAD_URL environment variables
    startServer()