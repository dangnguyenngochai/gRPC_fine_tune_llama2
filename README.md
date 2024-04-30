- Create code from `.proto` file
   
   - On window
   
      python -m grpc_tools.protoc -I . --python_out=. --pyi_out=. --grpc_python_out=. .\proto\sft_llama2.proto
   
   - On linux
      python -m grpc_tools.protoc -I . --python_out=. --pyi_out=. --grpc_python_out=. proto/sft_llama2.proto

