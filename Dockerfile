FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 필수 패키지 설치
RUN apt update && apt-get install -y software-properties-common git libsndfile1 ffmpeg

# Python 3.9 설치 및 기본 Python으로 설정
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.9 python3.9-venv python3.9-dev && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    update-alternatives --set python3 /usr/bin/python3.9

RUN apt-get install -y python3-pip && \
    pip3 install --upgrade pip && \
    pip install networkx==2.8.8

# MeloTTS 설치
RUN python3.9 -m pip install git+https://github.com/myshell-ai/MeloTTS.git
RUN python3.9 -m unidic download

# requirements.txt 복사 및 패키지 설치
COPY ./requirements.txt /tmp/requirements.txt
RUN python3.9 -m pip install -r /tmp/requirements.txt
