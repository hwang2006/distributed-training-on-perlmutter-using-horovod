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

