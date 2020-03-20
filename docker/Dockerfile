from nvcr.io/nvidia/pytorch:20.02-py3

RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt && cd torch2trt && python setup.py install

# Install Python packages and pytorch
ENV NB_USER pytorch
ENV NB_UID 1000

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /workspace/pytorch_detection