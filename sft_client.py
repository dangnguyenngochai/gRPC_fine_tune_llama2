import asyncio

import grpc
from grpc import aio as grpc_aoi

from proto.sft_llama2_pb2_grpc import SFTServerStub
from proto.sft_llama2_pb2 import FineTuneReply, FineTuneRequest

import logging
logging.basicConfig(level=logging.INFO) 

from time import perf_counter

import argparse

parser = argparse.ArgumentParser(description='fine_tune_llama2')
parser.add_argument('-f', '--file')

async def fine_tune(data: bytes, file_name: str):
    print("File type", type(data))

    async with grpc_aoi.insecure_channel('localhost:50052') as channel:
        stub = SFTServerStub(channel)
        start = perf_counter()
        
        req = FineTuneRequest(data_file=data, file_name=file_name)

        res: FineTuneReply = await stub.fine_tune(
            request = req
            )
        
        status = res.status
        print(status)
        
async def inference():
    pass

if __name__ == '__main__':
    args = parser.parse_args()
    file_name = args.file.split('\\')[-1]
    print(args.file)

    with open(file_name, 'rb') as f:
        data = f.read()
        asyncio.run(fine_tune(data, file_name))

    print("Sent data in file", args.file)