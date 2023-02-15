#!/bin/bash
#SBATCH --image=qualis2006/tensorflow-pytorch-horovod:tf2.10_pt1.13
#SBATCH --module=gpu,nccl-2.15
#SBATCH -N 2
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

#export NCCL_DEBUG=INFO

#srun -l -u --mpi=pmi2 shifter bash -c "python horovod/examples/tensorflow2/tensorflow2_keras_mnist.py"
#srun -l -u --mpi=pmi2 shifter bash -c "python distributed-training-on-perlmutter-using-horovod/src/tensorflow/tf_keras_imagenet_resnet50.py"
srun -l -u --mpi=pmi2 shifter python distributed-training-on-perlmutter-using-horovod/src/tensorflow/tf_keras_imagenet_resnet50.py
