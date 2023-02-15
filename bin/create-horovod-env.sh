#!/bin/bash

# make cuda available and cmake is required to compile & build horovod 
module restore
module load cudnn/8.7.0  nccl/2.15.5-ofi  evp-patch

# create the horovd virtual environment
# make sure that miniconda was installed in /scratch/userID/miniconda3
# prefix (=horovod directory): /scratch/$USER/miniconda3/envs/hvd-env
conda env create -f environment.yml --force

# activate the horovod environment
conda activate horovod


# install horovod in the horovod environment
HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_NCCL_LINK=SHARED HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 \
HOROVOD_WITH_MPI=1 HOROVOD_WITH_GLOO=1 pip install --no-cache-dir horovod==0.26.1

