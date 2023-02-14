# distributed-training-with-horovod-on-perlmutter

This repository is intended to share a large-scale distributed deep learning training practice for those who might be interested in running his/her deep learning codes accross multiple GPU nodes using [Horovod](https://github.com/horovod/horovod) on Perlmutter.

**Contents**
* [NERSC Perlmutter Supercomputer](#nersc-perlmutter-supercomputer)
* [Distributed DL training practices on supercomputer](#distributed-dl-training-practices-on-supercomputer)



## NERSC Perlmutter Supercomputer
[Perlmutter](https://docs.nersc.gov/systems/perlmutter/), located at [NERSC](https://www.nersc.gov/) in [Lawrence Berkeley National Laboratory](https://www.lbl.gov/), is a HPE Cray EX supercomputer with ~1,500 AMD Milan CPU nodes and ~6000 Nvidia A100 GPUs (4 GPUs per node). It debutted as the world 5th fastest supercomputer in the Top500 list in June 2021. Refer to this [link](https://docs.nersc.gov/systems/perlmutter/architecture/) for the architecutural details of Perlmutter including system specifications, system performance, node specifications and interconnect. [slurm](https://slurm.schedmd.com/) is adopted for cluster/resource management and job scheduling. 

<p align="center"><img src="https://user-images.githubusercontent.com/84169368/218645916-30e920b5-b2cf-43ad-9f13-f6a2568c0e37.jpg" width=500/></p>

## Distributed DL training practices on supercomputer
We may need to set up some ditributed deep learning routines or workflows by which DL researchers and Supercomputer facilities administrators exchange and share ideas and thoughts as to how to develope and run distributed training/inferencing practices on national supercomputing facilites. It might be that distributed deep learning (DL) practices on national supercomputing facilities are not so hard as we think it is, with proper tools, flexible operation & resource management policies and reasonably easy-to-use services available in the hands of DL researchers and developers. 
<p align="center"><img src="https://user-images.githubusercontent.com/84169368/218653307-4928d27e-50e5-4bf6-a8b3-b9ab7914cd63.png" width=500/></p> 

## Installing Conda
Once logging in to Perlmutter, you will need to have either [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your scratch directory. Anaconda is distribution of the Python and R programming languages for scientific computing, aiming to simplify package management and deployment. Anaconda comes with +150 data science packages, whereas Miniconda, a small bootstrap version of Anaconda, comes with a handful of what's needed.

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

