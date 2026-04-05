# Base image: Python 3.14 with CUDA 12.8 support
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.14 from deadsnakes PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.14 \
        python3.14-venv \
        python3.14-dev \
        curl \
        git \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.14 \
    && ln -sf /usr/bin/python3.14 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.14 /usr/bin/python \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# DataSphere requirement: jupyter user with UID 1000
RUN useradd -ms /bin/bash --uid 1000 jupyter

# Install packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        -r /tmp/requirements.txt \
    && pip install --no-cache-dir \
        ipykernel \
        jupyter \
    && python -m ipykernel install --name python3.14 --display-name "Python 3.14"

# Switch to jupyter user
USER jupyter
WORKDIR /home/jupyter

CMD ["bash"]
