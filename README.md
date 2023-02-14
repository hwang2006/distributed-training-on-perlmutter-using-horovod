# distributed-training-with-horovod-on-perlmutter

This repository is intended to share a large-scale distributed deep learning training practice for those who might be interested in running his/her deep learning codes accross multiple GPU nodes using [Horovod](https://github.com/horovod/horovod) on Perlmutter.

**Contents**
* [NERSC Perlmutter Supercomputer](#nersc-perlmutter-supercomputer)
* [Distributed DL training practices on supercomputer](#distributed-dl-training-practices-on-supercomputer)
* [Installing Conda](#installing-conda)


## NERSC Perlmutter Supercomputer
[Perlmutter](https://docs.nersc.gov/systems/perlmutter/), located at [NERSC](https://www.nersc.gov/) in [Lawrence Berkeley National Laboratory](https://www.lbl.gov/), is a HPE Cray EX supercomputer with ~1,500 AMD Milan CPU nodes and ~6000 Nvidia A100 GPUs (4 GPUs per node). It debutted as the world 5th fastest supercomputer in the Top500 list in June 2021. Refer to this [link](https://docs.nersc.gov/systems/perlmutter/architecture/) for the architecutural details of Perlmutter including system specifications, system performance, node specifications and interconnect. [slurm](https://slurm.schedmd.com/) is adopted for cluster/resource management and job scheduling. 

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
2. Install Miniconda. By default conda will be installed in your home directory, which has a limited disk space. You will install it on the /global/common/software/myproject directory and then create subsequent conda virtual environments on your scratch directory. 
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
[glogin01]$ conda --version
conda 23.1.0
```

## Why Horovod for distributed DL?
Horovod, developed by Uber in 2017, is a distributed deep learning training framework, aimed to make it easy and simple to take a DL code developed with different DL frameworks like Tensorflow and Pytorch and scale it to run across many GPUs. It is designed with having the followings in mind in the first place:
1. (neutral to DL frameworks to be used) Is it possible to make your DL codes run in parallel irrespective of whether you are using Tensorflow, Keras or Pytorch?    
2. (easy to use & codify) How much modification does one have to make to a existing DL code to make it distributed? and how easy is it to run it?
3. (fast to run) How much faster would it run in distributed mode and how easy is it to scale up to run?

<p align="center"><img src=" " width=800/></p> 

