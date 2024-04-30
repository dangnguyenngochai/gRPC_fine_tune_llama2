import asyncio
import logging

from sft.sft_lora_llama2 import run_sft, run_infer

import grpc 
from proto.sft_llama2_pb2_grpc import (
    add_SFTServerServicer_to_server,
    SFTServerServicer
)
from proto.sft_llama2_pb2 import (
    FineTuneReply,
    FineTuneRequest,
    InferenceRequest,
    InferenceReply
)

logging.basicConfig(level=logging.INFO)

class  SFTServerServicer(SFTServerServicer):
    async def inference(self, request: InferenceRequest, context) -> InferenceReply:
        logging.info("Inference request recieved")
        try:
            prompt = request.prompt
            response = run_infer(prompt)
        except Exception as ex:
            logging.info(ex)
            return InferenceReply(response="Error")
        logging.info("Inference request handled")
        return InferenceReply(response=response)

    async def fine_tune(self, request: FineTuneRequest, context) -> FineTuneReply:
        logging.info("Fine-tune request recieved")
        try:
            data_file = request.data_file
            file_name = request.file_name
            with open("data/%s"%file_name,  'wb+') as f:
                f.write(data_file)

            with_data = True
            file_path = "data/%s"%file_name
            run_sft(with_data, file_path)

        except Exception as ex:
            logging.info(ex)
            return FineTuneReply(status='Error')

        logging.info("Fine-tune request handled")
        return FineTuneReply(status='Done')

# The function define how we are going to start our server
async def serve():
    server = grpc.aio.server()
    add_SFTServerServicer_to_server(SFTServerServicer(), server)
    # using ip v6
    adddress = "[::]:50052"
    server.add_insecure_port(adddress)
    logging.info(f"Starting server on {adddress}")
    await server.start()
    await server.wait_for_termination()

if __name__ == '__main__':
    asyncio.run(serve())