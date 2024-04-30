# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from proto import sft_llama2_pb2 as proto_dot_sft__llama2__pb2


class SFTServerStub(object):
    """Supervised Fine-tuning service definition

    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.fine_tune = channel.unary_unary(
                '/SFTServer/fine_tune',
                request_serializer=proto_dot_sft__llama2__pb2.FineTuneRequest.SerializeToString,
                response_deserializer=proto_dot_sft__llama2__pb2.FineTuneReply.FromString,
                )
        self.inferene = channel.unary_unary(
                '/SFTServer/inferene',
                request_serializer=proto_dot_sft__llama2__pb2.InferenceRequest.SerializeToString,
                response_deserializer=proto_dot_sft__llama2__pb2.InferenceReply.FromString,
                )


class SFTServerServicer(object):
    """Supervised Fine-tuning service definition

    """

    def fine_tune(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def inferene(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_SFTServerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'fine_tune': grpc.unary_unary_rpc_method_handler(
                    servicer.fine_tune,
                    request_deserializer=proto_dot_sft__llama2__pb2.FineTuneRequest.FromString,
                    response_serializer=proto_dot_sft__llama2__pb2.FineTuneReply.SerializeToString,
            ),
            'inferene': grpc.unary_unary_rpc_method_handler(
                    servicer.inferene,
                    request_deserializer=proto_dot_sft__llama2__pb2.InferenceRequest.FromString,
                    response_serializer=proto_dot_sft__llama2__pb2.InferenceReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'SFTServer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class SFTServer(object):
    """Supervised Fine-tuning service definition

    """

    @staticmethod
    def fine_tune(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/SFTServer/fine_tune',
            proto_dot_sft__llama2__pb2.FineTuneRequest.SerializeToString,
            proto_dot_sft__llama2__pb2.FineTuneReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def inferene(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/SFTServer/inferene',
            proto_dot_sft__llama2__pb2.InferenceRequest.SerializeToString,
            proto_dot_sft__llama2__pb2.InferenceReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
