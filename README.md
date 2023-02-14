# distributed-training-with-horovod-on-perlmutter

This repository is intended to share a large-scale distributed deep learning training practice for those who might be interested in running his/her deep learning codes accross multiple GPU nodes using [Horovod](https://github.com/horovod/horovod) on Perlmutter.

**Contents**

## NERSC Perlmutter Supercomputer
[Perlmutter](https://docs.nersc.gov/systems/perlmutter/architecture/), located at [NERSC](https://www.nersc.gov/) in [Lawrence Berkeley National Laboratory](https://www.lbl.gov/), is a HPE Cray EX supercomputer with ~1,500 AMD Milan CPU nodes and ~6000 Nvidia A100 GPUs (4 GPUs per node). It debutted as the world 5th fastest supercomputer in the Top500 list in June 2021. [Slurm](https://slurm.schedmd.com/) is adopted for cluster/resource management and job scheduling.

<p align="center"><img src=" "/></p>
