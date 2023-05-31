FROM nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04

RUN apt update && apt install -y wget

# Prepare the global python environment
RUN apt install -y python3.8
RUN ln -sf /usr/bin/python3.8 /usr/bin/python
RUN apt install -y pip

ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# for opencv
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6

# Install other tools
RUN apt install -y \
    git \
    libsm6  \
    libxext-dev \
    libxrender1 \
    unzip \
    cmake \
    libxml2 libxml2-dev libxslt1-dev \
    dirmngr gnupg2 lsb-release \
    xvfb kmod swig patchelf \
    libopenmpi-dev  libcups2-dev \
    libssl-dev  libosmesa6-dev \
    mesa-utils

# Clean up to make the resulting image smaller
RUN  rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
