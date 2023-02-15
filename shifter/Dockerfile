FROM nvcr.io/nvidia/pytorch:22.11-py3

RUN \
    echo "PIP installing tensorflow-gpu..." && \
    pip install tensorflow-gpu==2.10.0      && \
    echo "PIP installing filelock..."       && \
    pip install filelock                   

ENV HOROVOD_GPU_OPERATIONS=NCCL \
    HOROVOD_NCCL_LINK=SHARED    \
    HOROVOD_WITH_TENSORFLOW=1   \
    HOROVOD_WITH_PYTORCH=1      \
    HOROVOD_WITH_MPI=1          \
    HOROVOD_WITH_GLOO=1
    
RUN \
    echo "PIP Installing Horovod..."        && \
    pip install --no-cache-dir horovod
