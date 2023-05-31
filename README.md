# distributed-training-on-perlmutter-using-horovod

This repository is intended to share large-scale distributed deep learning training practices with those who might be interested in running his/her deep learning codes accross multiple GPU nodes using [Horovod](https://github.com/horovod/horovod) on Perlmutter.

**Contents**
* [NERSC Perlmutter Supercomputer](#nersc-perlmutter-supercomputer)
* [Distributed DL training practices on supercomputer](#distributed-dl-training-practices-on-supercomputer)
* [Installing Conda](#installing-conda)
* [Why Horovod for distributed DL](#why-horovod-for-distributed-dl)
* [Building Horovod](#building-horovod)
* [Horovod Usage](#horovod-usage)
* [Running Horovod interactively](#running-horovod-interactively)
* [Submitting and Running a Horovod batch job](#submitting-and-running-a-horovod-batch-job)
* [Running Jupyter](#running-jupyter)
* [Why Shifter Container](#why-shifter-container)
* [Running Horovod interactively using Shifter](#running-horovod-interactively-using-shifter)
* [Submitting and Running a Horovod batch job using Shifter](#submitting-and-running-a-horovod-batch-job-using-shifter)
* [A glimpse of Running Pytorch Lightning](#a-glimpse-of-running-pytorch-lightning)


## NERSC Perlmutter Supercomputer
[Perlmutter](https://docs.nersc.gov/systems/perlmutter/), located at [NERSC](https://www.nersc.gov/) in [Lawrence Berkeley National Laboratory](https://www.lbl.gov/), is a HPE Cray EX supercomputer with ~1,500 AMD Milan CPU nodes and ~6000 Nvidia A100 GPUs (4 GPUs per node). It debuted as the world 5th fastest supercomputer in the Top500 list in June 2021. Please refer to [Perlmutter Architecture](https://docs.nersc.gov/systems/perlmutter/architecture/) for the architecutural details of Perlmutter including system specifications, system performance, node specifications and interconnect. [Slurm](https://slurm.schedmd.com/) is adopted for cluster/resource management and job scheduling. 

<p align="center"><img src="https://user-images.githubusercontent.com/84169368/218645916-30e920b5-b2cf-43ad-9f13-f6a2568c0e37.jpg" width=550/></p>

## Distributed DL training practices on supercomputer
We may need to set up some ditributed deep learning routines or workflows by which DL researchers and Supercomputer facilities administrators exchange and share ideas and thoughts as to how to develope and run distributed training/inferencing practices on national supercomputing facilites. It might be that distributed deep learning (DL) practices on national supercomputing facilities are not so hard as we think it is, with proper tools, flexible operation & resource management policies and reasonably easy-to-use services available in the hands of DL researchers and developers. 
<p align="center"><img src="https://user-images.githubusercontent.com/84169368/218653307-4928d27e-50e5-4bf6-a8b3-b9ab7914cd63.png" width=500/></p> 

## Installing Conda
Once logging in to Perlmutter, you will need to have either [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your scratch directory. Anaconda is distribution of the Python and R programming languages for scientific computing, aiming to simplify package management and deployment. Anaconda comes with +150 data science packages, whereas Miniconda, a small bootstrap version of Anaconda, comes with a handful of what's needed.

Note that we will assume that your NERSC account information is as follows:
```
- Username: elvis
- Project Name : ddlproj
- Account (CPU) : m1234
- Account (GPU) : m1234_g
```
so, you will have to replace it with your real NERSC account information.

1. Download Anaconda or Miniconda. Miniconda comes with python, conda (package & environment manager), and some basic packages. Miniconda is fast to install and could be sufficient for distributed deep learning training practices. 
```
# (option 1) Anaconda 
perlmutter:login15>$ cd $SCRATCH 
perlmutter:login15>$ wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
```
```
# (option 2) Miniconda 
perlmutter:login15>$ cd $SCRATCH
perlmutter:login15>$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
2. Install Miniconda. By default conda will be installed in your home directory, which has a limited disk space. Please refer to [File System Overview](https://docs.nersc.gov/filesystems/) for more details of NERSC storage systems . You will install it on the /global/common/software/&lt;myproject&gt; directory and then create a conda virtual environment on your own scratch directory. 
```
perlmutter:login15>$ chmod 755 Miniconda3-latest-Linux-x86_64.sh
perlmutter:login15>$ ./Miniconda3-latest-Linux-x86_64.sh

Welcome to Miniconda3 py39_4.12.0

In order to continue the installation process, please review the license
agreement.
Please, press ENTER to continue
>>>                               <======== press ENTER here
.
.
.
Do you accept the license terms? [yes|no]
[no] >>> yes                      <========= type yes here 

Miniconda3 will now be installed into this location:
/global/homes/s/swhwang/miniconda3        

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/global/homes/s/swhwang/miniconda3] >>> /global/common/software/ddlproj/elvis/miniconda3  <======== type /global/common/software/myproject/$USER/miniconda3 here
PREFIX=/global/common/software/dasrepo/swhwang/miniconda3
Unpacking payload ...
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /global/common/software/ddlproj/elvis/miniconda3
.
.
.
Preparing transaction: done
Executing transaction: done
installation finished.
Do you wish the installer to initialize Miniconda3
by running conda init? [yes|no]
[no] >>> yes         <========== type yes here
.
.
.
If you'd prefer that conda's base environment not be activated on startup,
   set the auto_activate_base parameter to false:

conda config --set auto_activate_base false

Thank you for installing Miniconda3!
```

3. finalize installing Miniconda with environment variables set including conda path.
```
perlmutter:login15>$ source ~/.bashrc    # set conda path and environment variables 
perlmutter:login15>$ conda config --set auto_activate_base false
perlmutter:login15>$ which conda
/global/common/software/ddlproj/elvis/miniconda3/condabin/conda
perlmutter:login15>$ conda --version
conda 23.1.0
```

## Why Horovod for distributed DL?
Horovod, developed by Uber in 2017, is a distributed deep learning training framework, aimed to make it easy and simple to take a DL code developed with different DL frameworks like Tensorflow and Pytorch and scale it to run across many GPUs. It is designed with having the followings in mind in the first place:
1. (neutral to DL frameworks to be used) Is it possible to make your DL codes run in parallel irrespective of whether you are using Tensorflow, Keras or Pytorch?    
2. (easy to use & codify) How much modification does one have to make to a existing DL code to make it distributed? and how easy is it to run it?
3. (fast to run) How much faster would it run in distributed mode and how easy is it to scale up to run?

<p align="center"><img src="https://user-images.githubusercontent.com/84169368/218693630-4943ed18-488f-41bd-af97-e14520cc5897.png" width=750/></p> 

## Building Horovod
Now you are ready to build Horovod as a conda virtual environment: 
1. load modules: 
```
perlmutter:login15>$ module load  cudnn/8.7.0  nccl/2.15.5-ofi  evp-patch
```
2. create a new conda virtual environment and activate the environment:
```
perlmutter:login15>$ conda create -n horovod
perlmutter:login15>$ conda activate horovod
```
3. install the pytorch conda package & the tensorflow pip package:
```
(horovod) perlmutter:login15>$ conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
(horovod) perlmutter:login15>$ pip install tensorflow-gpu==2.10.0
```
4. install the horovod pip package with support for tensorflow and pytorch with [NCCL](https://developer.nvidia.com/nccl), [MPI](https://www.open-mpi.org/) and [GLOO](https://pytorch.org/docs/stable/distributed.html) enabled:
```
(horovod) perlmutter:login15>$ HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_NCCL_LINK=SHARED HOROVOD_WITH_TENSORFLOW=1 \
HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MPI=1 HOROVOD_WITH_GLOO=1 pip install --no-cache-dir horovod
```
5. verify the horovod conda environment. You should see output something like the following:
```
(horovod) perlmutter:login15>$ horovodrun -cb
Horovod v0.27.0:

Available Frameworks:
    [X] TensorFlow
    [X] PyTorch
    [ ] MXNet

Available Controllers:
    [X] MPI
    [X] Gloo

Available Tensor Operations:
    [X] NCCL
    [ ] DDL
    [ ] CCL
    [X] MPI
    [X] Gloo
```
6. You have built the horovod virtual environment based on [Cray MPICH](https://docs.nersc.gov/development/programming-models/mpi/cray-mpich/) that is default and 
recommended on Cray systems like Perlmutter. You can also build your horovod environment against [Open MPI](https://docs.nersc.gov/development/programming-models/mpi/openmpi/) by loading the openmpi module. 
```
perlmutter:login15>$ module restore
perlmutter:login15>$ module load nccl/2.14.3 cudnn/8.7.0
perlmutter:login15>$ module use /global/common/software/dasrepo/swhwang/modulefiles
perlmutter:login15>$ #module use /global/common/software/m3169/perlmutter/modulefiles
perlmutter:login15>$ module load openmpi/4.1.3-ucx-1.11.1-cuda-21.11_11.5

perlmutter:login15>$ conda create -n openmpi-hvd
perlmutter:login15>$ conda activate openmpi-hvd

(openmpi-hvd) perlmutter:login15>$ conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
(openmpi-hvd) perlmutter:login15>$ pip install tensorflow-gpu==2.10.0
(openmpi-hvd) perlmutter:login15>$ HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_NCCL_LINK=SHARED HOROVOD_WITH_TENSORFLOW=1 \
HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MPI=1 HOROVOD_WITH_GLOO=1 pip install --no-cache-dir horovod

(openmpi-hvd) perlmutter:login15>$ horovodrun -cb
```

## Horovod Usage
To use horovod, five steps/lines to be added in your code:
1. Initialize Horovod.
```
# Tensorflow 
import horovod.tensorflow as hvd
hvd.init()

# Keras
import horovod.keras as hvd
hvd.init()

# Pytorch
import horovod.torch as hvd
hvd.init()
```
2. Pin GPU to each worker, making sure each worker to be allocated to each GPU available.
```
# Tensorflow/Keras
tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# Pytorch
torch.cuda.set_device(hvd.local_rank())
```
3. Adjust learning rate and wrap the optimizer
```
# Tensorflow
opt = tf.optimizers.Adam(0.01 * hvd.size())
opt = hvd.DistributedOptimizer(opt,…)

# Keras
opt = keras.optimizers.Adadelta(0.01 * hvd.size())
opt = hvd.DistributedOptimizer(opt,…)

# Pytorch
opt = optim.SGD(model.parameters(), 0.01 * hvd.size())
opt= hvd.DistributedOptimizer(opt, …)
```
4. Broadcast the initial variable states from the masker worker (rank 0)and synchroize state across workers.
```
# Tensorflow/Keras
callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

# Pytorch
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)
```
5. Checkpoint on the master worker (rank 0)
```
# Tensorflow/Keras
if hvd.rank() == 0:
  callbacks.append(keras.callbacks.ModelCheckpoint(args.checkpoint_format))

# Pytorch
if hvd.rank() == 0:
   state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), } 
   torch.save(state, filepath)
```

6. An example code using Pytorch (see the [src](https://github.com/hwang2006/distributed-training-on-perlmutter-using-horovod/tree/main/src) directory for full example codes): 
```
import torch
import horovod.torch as hvd

# Initialize Horovod
hvd.init()

# Pin GPUs to local rank
torch.cuda.set_device(hvd.local_rank())

# Build model
model = Net()
model.cuda()
opt = optim.SGD(model.parameters())

# Adjust learning rate and wrap the optimizer
opt = optim.SGD(model.parameters(), 0.01 * hvd.size())
opt = hvd.DistributedOptimizer(opt, …)

# Broadcast parameters and optimizer state from the master worker (rank 0)
hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

for epoch in range (100):
   for batch, (data, target) in enumerate(...):
       opt.zero_grad()
       output = model(data)
       loss = F.nll_loss(output, target)
       loss.backward()
       opt.step()
   # checkpoint at every epoch
   if hvd.rank() == 0:
      state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), } 
      torch.save(state, filepath)
```

## Running Horovod interactively 
Now, you are ready to run distributed training using Horovod on Perlmutter. Please refer to [Running jobs](https://docs.nersc.gov/jobs/) for more details.
1. request allocation of available GPU-nodes for interactively running and testing distributed training codes. Please refer to [Interactive Jobs](https://docs.nersc.gov/jobs/interactive/) for more details of interactive resource allocation.
```
(horovod) perlmutter:login15>$ salloc --nodes 2 --qos interactive --time 01:00:00 --constraint gpu --gpus-per-node=4 --account=m1234_g
salloc: Pending job allocation 5472214
salloc: job 5472214 queued and waiting for resources                                   
salloc: job 5472214 has been allocated resources                                         
salloc: Granted job allocation 5472214
salloc: Waiting for resource configuration
salloc: Nodes nid[001140-001141] are ready for job
```
In this example case, nid001140 and nid001141 are allocated with 4 GPUs each, and you are residing on the nid001140 node.

2. load modules again on the gpu node.
```
nid001140>$ module load  cudnn/8.3.2  nccl/2.15.5-ofi  evp-patch
```
3. activate the horovod conda environment. 
```
nid001140>$ conda activate horovod
(horovod) nid001140>$
```
4. run & test horovod-enabled distributed DL codes. Please refer to 
  - to run on the two nodes with 4 GPUs each: 
```
(horovod) nid001140>$ srun -n 8 python train_hvd.py
```
  - to run on two nodes with 2 GPUs each:
```
(horovod) nid001140>$ srun -n 4 python train_hvd.py
 or
(horovod) nid001140>$ srun -N 2 -n 4 python train_hvd.py
```
  - to run on one node with 4 GPUs:
```
(horovod) nid001140>$ srun -N 1 -n 4 python train_hvd.py
```
6. You can also run the openmpi-enabled horovod interactively   
```
perlmutter:login15>$ salloc --nodes 2 --qos interactive --time 01:00:00 --constraint gpu --gpus-per-node=4 --account=m1234_g
nid001140>$ module restore
nid001140>$ module load nccl/2.14.3 cudnn/8.7.0
nid001140>$ module use /global/common/software/dasrepo/swhwang/modulefiles
nid001140>$ module load openmpi/4.1.3-ucx-1.11.1-cuda-21.11_11.5

nid001140>$ conda activate openmpi-hvd

(openmpi-hvd) nid001140>$ horovodrun -np 8 -H nid001140:4,nid001141:4 python train_hvd.py
(openmpi-hvd) nid001140>$ mpirun -np 8 -H nid001140:4,nid001141:4 python train_hvd.py
(openmpi-hvd) nid001140>$ mpirun -np 4 -H nid001140:2,nid001141:2 python train_hvd.py
(openmpi-hvd) nid001140>$ mpirun -np 4 python train_hvd.py

** Note that for some openmpi-relating issue, if it failed to run, you should be able to run it using GLOO instead of MPI.
(openmpi-hvd) nid001140>$ horovodrun --gloo -np 8 -H nid001140:4,nid001141:4 python train_hvd.py
```

## Submitting and Running a Horovod batch job
1. edit a batch job script running on 2 nodes with 4 GPUs each. Please refer to [Running Jobs on Perlmutter](https://docs.nersc.gov/systems/perlmutter/running-jobs/) for some examples of slurm job scripts.
```
perlmutter:login15>$ cat horovod_batch.sh
#!/bin/bash
#SBATCH -A m1234_g
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
conda activate horovod

srun python distributed-training-on-perlmutter-using-horovod/src/tensorflow/tf_keras_mnist.py
#srun python distributed-training-on-perlmutter-using-horovod/src/pytorch/pytorch_imagenet_resnet50.py
#srun python distributed-training-on-perlmutter-using-horovod/src/tensorflow/tf_keras_imagenet_resnet50.py
```
2. submit and run the batch job.
```
perlmutter:login15>$ sbatch horovod_batch.sh
Submitted batch job 5473133
```
3. check the batch job status.
```
perlmutter:login15>$ squeue -u $USER
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           5473133  gpu_ss11 horovod_    elvis PD       0:00      2 (Priority)
perlmutter:login15>$ squeue -u $USER
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           5473133  gpu_ss11 horovod_    elvis  R       0:33      2 nid[002836,003928]
```
You can also submit and run openmpi-enabled horovod batch jobs
```
perlmutter:login15>$ cat openmpi_horovod_batch.sh
#!/bin/bash
#SBATCH -A m1234_g
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

module load nccl/2.14.3 cudnn/8.7.0
module use /global/common/software/dasrepo/swhwang/modulefiles
module load openmpi/4.1.3-ucx-1.11.1-cuda-21.11_11.5
source ~/.bashrc
conda activate openmpi-hvd

srun python distributed-training-on-perlmutter-using-horovod/src/tensorflow/tf_keras_mnist.py
#srun python distributed-training-on-perlmutter-using-horovod/src/pytorch/pytorch_imagenet_resnet50.py
#srun python distributed-training-on-perlmutter-using-horovod/src/tensorflow/tf_keras_imagenet_resnet50.py
```
```
perlmutter:login15>$ sbatch openmpi_horovod_batch.sh
perlmutter:login15>$ squeue -u $USER
```

## Running Jupyter
[Jupyter](https://jupyter.org/) is free software, open standards, and web services for interactive computing across all programming languages. Jupyterlab is the latest web-based interactive development environment for notebooks, code, and data. The Jupyter Notebook is the original web application for creating and sharing computational documents. NERSC provides a [JupyterHub service](https://docs.nersc.gov/services/jupyter/#jupyterhub) that allows you to run a Jupyter notebook on Perlmutter. JupyterHub is designed to serve Jupyter notebook for multiple users to work in their own individual workspaces on shared resources. Please refer to [JupyterHub](https://jupyterhub.readthedocs.io/en/stable/) for more detailed information. You can also run your own jupyter notebook server on a compute node (*not* on a login node), which will be accessed from the browser on your PC or laptop through SSH tunneling. 
<p align="center"><img src="https://github.com/hwang2006/KISTI-DL-tutorial-using-horovod/assets/84169368/34a753fc-ccb7-423e-b0f3-f973b8cd7122"/>
</p>

In order to do so, you need to add the horovod-enabled virtual envrionment that you have created as a python kernel.
1. activate the horovod-enabled virtual environment:
```
perlmutter:login15>$ conda activate horovod
```
2. install Jupyter on the virtual environment:
```
(horovod) perlmutter:login15>$ conda install jupyter
(horovod) perlmutter:login15>$ pip install jupyter-tensorboard
```
3. add the virtual environment as a jupyter kernel:
```
(horovod) perlmutter:login15>$ pip install ipykernel 
(horovod) perlmutter:login15>$ python -m ipykernel install --user --name horovod
```
4. check the list of kernels currently installed:
```
(horovod) perlmutter:login15>$ jupyter kernelspec list
Available kernels:
  pytho       /global/common/software/ddlproj/evlis/miniconda3/envs/craympi-hvd/share/jupyter/kernels/python3
  horovod     /global/u1/s/elvis/.local/share/jupyter/kernels/horovod
```
5. launch a jupyter notebook server on a compute node 
- to deactivate the virtual environment
```
(horovod) perlmutter:login15>$ conda deactivate
```
- to create a batch script for launching a jupyter notebook server: 
```
perlmutter:login15>$ cat jupyter_run.sh
#!/bin/bash
#SBATCH -A dasrepo_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 8:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -c 32

export SLURM_CPU_BIND="cores"

#removing the old port forwading
if [ -e port_forwarding_command ]
then
  rm port_forwarding_command
fi

#getting the node name and port
SERVER="`hostname`"
PORT_JU=$(($RANDOM + 10000 )) # random number greaten than 10000

echo $SERVER
echo $PORT_JU

echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@perlmutter-p1.nersc.gov" > port_forwarding_command
echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@perlmutter-p1.nersc.gov"
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@perlmutter-p1.nersc.gov" > port_forwarding_command
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@perlmutter-p1.nersc.gov"

echo "load module-environment"
module load  cudnn/8.7.0  nccl/2.15.5-ofi  evp-patch

echo "execute jupyter"
source ~/.bashrc
conda activate horovod
cd $SCRATCH/ddl-projects
jupyter lab --ip=0.0.0.0 --port=${PORT_JU} --NotebookApp.token=${USER}
#bash -c "jupyter lab --ip=0.0.0.0 --port=${PORT_JU} --NotebookApp.token='${USER}'"
echo "end of the job"
```
- to launch a jupyter notebook server 
```
perlmutter:login15>$ sbatch jupyter_run.sh
Submitted batch job 5494200
```
- to check if a jupyter notebook server is running
```
perlmutter:login15>$ squeue -u $USER
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           5494200  gpu_ss11 jupyter_  swhwan PD       0:00      1 (Priority)
perlmutter:login15>$ squeue -u $USER
             JOBID       PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
            5494200       gpu_ss11  jupyter_    evlis  RUNNING       0:02   8:00:00      1 nid001140
perlmutter:login15>$ cat slurm-XXXXXX.out
.
.
[I 2023-02-14 08:30:04.790 ServerApp] Jupyter Server 1.23.4 is running at:
[I 2023-02-14 08:30:04.790 ServerApp] http://nid######:#####/lab?token=...
.
.
```
- to check the SSH tunneling information generated by the jupyter_run.sh script 
```
perlmutter:login15>$ cat port_forwarding_command
ssh -L localhost:8888:nid######:##### $USER@neuron.ksc.re.kr
```
6. open a SSH client (e.g., Putty, PowerShell, Command Prompt, etc) on your PC or laptop and log in to Perlmutter just by copying and pasting the port_forwarding_command:
```
C:\Users\hwang>ssh -L localhost:8888:nid######:##### elvis@perlmutter-p1.nersc.gov
Password(OTP):
Password:
```
7. open a web browser on your PC or laptop to access the jupyter server
```
- URL Address: localhost:8888
- Password or token: elvis   # your username on Perlmutter
```
<p align="center"><img src="https://user-images.githubusercontent.com/84169368/218938419-f38c356b-e682-4b1c-9add-6cfc29d53425.png"/></p> 


## Why Shifter Container? 
[Shifter](https://shifter.readthedocs.io/en/latest/) is designed to enable environment containers for HPC.  It allows a user to create their own software environment, typically with Docker, then run it at a supercomputing facility. Shifter aims to increase scientific computing productivity by:
- Simplifying software deployment and management
- Allowing user code to be portable
- Enabling scientists to share HPC software directly using the Docker framework and Dockerhub community of software.
- Encouraging repeatable and reproducible science with more durable software environments.
- Providing software solutions to improve system utilization by optimizing common bottlenecks in software delivery and I/O in-general.
- Empowering the user - deploy your own software environment

Shifter also comes with improvements in performance, especially for shared libraries. NERSC has been working on harnessing the power of Shifter to increase flexibility and usability of its HPC systems including Cori and Perlmutter by enabling Docker-like Linux container technology. According to NERSC's benchmark testing, it appears that Shifter is the best performing option for python code stacks across multiple nodes because Shifter is allowed to leverage its volume mounting capabilities to provide local-disk like functionality and I/O performance. Please refer to [Using Shifter at NERSC](https://docs.nersc.gov/development/shifter/how-to-use/) for more details.
<p align="center"><img src="https://user-images.githubusercontent.com/84169368/218930936-f268ab50-c21d-411a-98d0-62ad98348010.png"/></p> 

## Running Horovod interactively using Shifter
You don't have to bother to deal with all the hassles of the Conda and Horovod, and just request an allocation of available nodes using the Slurm salloc command and run a horovod-enabled shifter container built on Perlmutter. That's it!
```
perlmutter:login15>$ salloc --nodes 2 --qos interactive --time 01:00:00 --constraint gpu --gpus-per-node=4 --account=m1234_g

# to run a tensorflow code using the Horovod-enabled shifter image on Perlmutter 
perlmutter:login15>$ NCCL_DEBUG=INFO srun --mpi=pmi2 -l -u -n 8 shifter --module=gpu,nccl-2.15 --image=qualis2006/tensorflow-pytorch-horovod:tf2.10_pt1.13 python distributed-training-using-horovod-on-perlmutter/src/tensorflow/tf_keras_cats_dogs.py

# to run a keras code using the Horovod-enabled shifter image on Perlmutter
perlmutter:login15>$ srun --mpi=pmi2 -n 8 shifter --module=gpu,nccl-2.15 --image=qualis2006/tensorflow-pytorch-horovod:tf2.10_pt1.13 python distributed-training-using-horovod-on-perlmutter/src/keras/keras_imagenet_resnet50.py

# to run a pytorch code using the Horovod-enabled shifter image on Perlmutter
perlmutter:login15>$ srun --mpi=pmi2 -n 8 shifter --module=gpu,nccl-2.15 --image=qualis2006/tensorflow-pytorch-horovod:tf2.10_pt1.13 python distributed-training-using-horovod-on-perlmutter/src/pytorch/pytorch_imagenet_resnet50.py
```

You can build your own Horovod Docker image with both Tensorflow and Pytorch enabled on your own machine (e.g., PC, laptop) as follows:   
```
# create a Dockerfile
[hwang@ubuntu20]$ cat Dockerfile
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


# build a Horovod Docker image
[hwang@ubuntu20]$ docker build -t qualis2006/tensorflow-pytorch-horovod:tf2.10_pt1.13 .

# list Docker images on your machine
REPOSITORY                              TAG             IMAGE ID       CREATED         SIZE
qualis2006/tensorflow-pytorch-horovod   tf2.10_pt1.13   cd355901ec90   2 days ago      20.1GB
.
.
```

## Submitting and Running a Horovod batch job using Shifter 
1. edit a batch job script running on 2 nodes with 4 GPUs each:
```
perlmutter:login15>$ cat ./shifter_horovod_batsh.sh
#!/bin/bash
#SBATCH --image=qualis2006/tensorflow-pytorch-horovod:tf2.10_pt1.13
#SBATCH --module=gpu,nccl-2.15
#SBATCH -N 2
#SBATCH -A ddlproj
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
```
2. submit and execute the batch job.
```
perlmutter:login15>$ sbatch shiter_horovod_batch.sh
Submitted batch job 5497322
```
3. check the batch job status.
```
perlmutter:login15>$ squeue -u $USER
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           5497322  gpu_ss11 shifter_  swhwang PD       0:00      2 (Priority)
perlmutter:login15>$ squeue -u $USER
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
           5497322  gpu_ss11 shifter_  swhwang  R       8:36      2 nid[008500-008501]
```

## A glimpse of Running Pytorch Lightning
[Pytorch Lightning](https://www.pytorchlightning.ai/index.html) is a lightweight wrapper or interface on top of PyTorch, which simplifies the implementation of complex deep learning models. It is a PyTorch extension that enables researchers and practitioners to focus more on their research and less on the engineering aspects of deep learning. PyTorch Lightning automates many of the routine tasks, such as distributing training across multiple GPUs, logging, and checkpointing, so that the users can focus on writing high-level code for their models.

We will show a glimpse of how to run a simple pytorch lightning code on multiple nodes interactively.
1. Install Pytorch Lightning:
```
perlmutter:login15>$ conda activate horovod
(horovod) perlmutter:login15>$ pip install lightning
```
2. Check the Pytorch Lightning version
```
(horovod) perlmutter:login15>$ conda list | grep lightning
lightning                 2.0.2                    pypi_0    pypi
lightning-cloud           0.5.34                   pypi_0    pypi
lightning-utilities       0.8.0                    pypi_0    pypi
pytorch-lightning         1.8.6                    pypi_0    pypi
```
3. request allocation of available GPU-nodes:
```
(horovod) perlmutter:login15>$ salloc --nodes 2 --qos interactive --time 01:00:00 --constraint gpu --gpus-per-node=4 --account=m1234_g
salloc: Pending job allocation 5472214
salloc: job 5472214 queued and waiting for resources                                   
salloc: job 5472214 has been allocated resources                                         
salloc: Granted job allocation 5472214
salloc: Waiting for resource configuration
salloc: Nodes nid[001140-001141] are ready for job
```
4. load modules and activate the horovod conda environment:
```
nid001140>$ module load cudnn/8.3.2 nccl/2.14.3  evp-patch
nid001140>$ conda activate horovod
(horovod) nid001140>$
```
4. run pytorch a lightning code:
- to run on the two nodes with 4 GPUs each
```
(horovod) nid001140>$ srun -N 2 --ntasks-per-node=4 python distributed-training-on-perlmutter-using-horovod/src/pytorch-lightning/pytorch_mnist_lightning.py --num_nodes 2
```
- to run the Bert NSMC (Naver Sentiment Movie Corpus) example in the src/pytorch-lightning directory, you need to install additional packages (i.e., emoji, soynlp, transformers and pandas) and download the nsmc datasets, for example, using git cloning
```
(horovod) nid001140>$ pip install emoji==1.7.0 soynlp transformers pandas
(horovod) nid001140>$ git clone https://github.com/e9t/nsmc  # download the nsmc datasets in the ./nsmc directory
(horovod) nid001140>$ srun -N 2 --ntasks-per-node=4 python distributed-training-on-perlmutter-using-horovod/src/pytorch-lightning/pt_bert_nsmc_lightning.py --num_nodes 2
```
- to run on the two nodes with 2 GPUs each
```
(horovod) nid001140>$ srun -N 2 --ntasks-per-node=2 python distributed-training-on-perlmutter-using-horovod/src/pytorch-lightning/pytorch_mnist_lightning.py --num_nodes 2 --devices 2
(horovod) nid001140>$ srun -N 2 --ntasks-per-node=2 python distributed-training-on-perlmutter-using-horovod/src/pytorch-lightning/pt_bert_nsmc_lightning.py --num_nodes 2 --devices 2
```
- to run on the two nodes with 1 GPU each
```
(horovod) nid001140>$ srun -N 2 --ntasks-per-node=1 python distributed-training-on-perlmutter-using-horovod/src/pytorch-lightning/pytorch_mnist_lightning.py --num_nodes 2 --devices 1
(horovod) nid001140>$ srun -N 2 --ntasks-per-node=1 python distributed-training-on-perlmutter-using-horovod/src/pytorch-lightning/pt_bert_nsmc_lightning.py --num_nodes 2 --devices 1
```
- to run one node with 4 GPUs
```
(horovod) nid001140>$ python distributed-training-on-perlmutter-using-horovod/src/pytorch-lightning/pytorch_mnist_lightning.py 
(horovod) nid001140>$ python distributed-training-on-perlmutter-using-horovod/src/pytorch-lightning/pt_bert_nsmc_lightning.py
(horovod) nid001140>$ srun -N 1 --ntasks-per-node=4 python distributed-training-on-perlmutter-using-horovod/src/pytorch-lightning/pt_bert_nsmc_lightning.py
```
- to run one node with 2 GPUs
```
(horovod) nid001140>$ python distributed-training-on-perlmutter-using-horovod/src/pytorch-lightning/pytorch_mnist_lightning.py --devices 2
(horovod) nid001140>$ python distributed-training-on-perlmutter-using-horovod/src/pytorch-lightning/pt_bert_nsmc_lightning.py --devices 2
(horovod) nid001140>$ srun -N 1 --ntasks-per-node=2 python distributed-training-on-perlmutter-using-horovod/src/pytorch-lightning/pt_bert_nsmc_lightning.py --devices 2
```
