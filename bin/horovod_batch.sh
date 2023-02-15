#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 2:00:00
#SBATCH -N 2
#SBATCH -c 32
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4 
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

export SLURM_CPU_BIND="cores"

module load  cudnn/8.3.2  nccl/2.15.5-ofi  evp-patch
source ~/.bashrc
conda activate craympi-hvd

srun python distributed-training-on-perlmutter-using-horovod/src/tensorflow/tf_keras_mnist.py
#srun python distributed-training-on-perlmutter-using-horovod/src/pytorch/pytorch_imagenet_resnet50.py
#srun python distributed-training-on-perlmutter-using-horovod/src/tensorflow/tf_keras_imagenet_resnet50.py
