FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update
RUN apt-get install -y protobuf-compiler

RUN pip install pyyaml lz4

