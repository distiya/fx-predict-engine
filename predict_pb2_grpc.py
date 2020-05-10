# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import predict_pb2 as predict__pb2


class FxPredictStub(object):
    """Missing associated documentation comment in .proto file"""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.getPredictionForBatch = channel.unary_unary(
                '/predict.FxPredict/getPredictionForBatch',
                request_serializer=predict__pb2.MultiPredictBatchForGranularity.SerializeToString,
                response_deserializer=predict__pb2.MultiPredictedCandleForGranularity.FromString,
                )


class FxPredictServicer(object):
    """Missing associated documentation comment in .proto file"""

    def getPredictionForBatch(self, request, context):
        """Missing associated documentation comment in .proto file"""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FxPredictServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'getPredictionForBatch': grpc.unary_unary_rpc_method_handler(
                    servicer.getPredictionForBatch,
                    request_deserializer=predict__pb2.MultiPredictBatchForGranularity.FromString,
                    response_serializer=predict__pb2.MultiPredictedCandleForGranularity.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'predict.FxPredict', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class FxPredict(object):
    """Missing associated documentation comment in .proto file"""

    @staticmethod
    def getPredictionForBatch(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/predict.FxPredict/getPredictionForBatch',
            predict__pb2.MultiPredictBatchForGranularity.SerializeToString,
            predict__pb2.MultiPredictedCandleForGranularity.FromString,
            options, channel_credentials,
            call_credentials, compression, wait_for_ready, timeout, metadata)
