#!/bin/bash 
#SBATCH -A <project>
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

#conda activate horovod
conda activate craympi-hvd
#cd $SCRATCH/ddl-projects
cd $SCRATCH

jupyter lab --ip=0.0.0.0 --port=${PORT_JU} --NotebookApp.token=${USER}
#bash -c "jupyter lab --ip=0.0.0.0 --port=${PORT_JU} --NotebookApp.token='${USER}'"
echo "end of the job"

