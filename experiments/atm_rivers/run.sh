#! /usr/bin/bash
#-----------------------------------------------------------------------------
#SBATCH --account=bk1517

#SBATCH --job-name=atm_rivers
#SBATCH --partition=compute
#SBATCH --nodes=10
#SBATCH --output=logs/LOG.atm_rivers.run.%j.out
#SBATCH --time=00:10:00
#SBATCH --qos=summerschool

#-----------------------------------------------------------------------------
#
# OpenMP environment variables
# ----------------------------
export OMP_NUM_THREADS=1
export ICON_THREADS=1
export OMP_SCHEDULE=dynamic,1
export OMP_DYNAMIC="false"
export OMP_STACKSIZE=200M

# environment variables for the experiment and the target system
# --------------------------------------------------------------
export MALLOC_TRIM_THRESHOLD_="-1"
export SLURM_DIST_PLANESIZE="32"
export OMPI_MCA_btl="self"
export OMPI_MCA_coll="^ml,hcoll"
export OMPI_MCA_io="romio321"
export OMPI_MCA_osc="ucx"
export OMPI_MCA_pml="ucx"
export UCX_HANDLE_ERRORS="bt"
export UCX_TLS="shm,dc_mlx5,dc_x,self"
export UCX_UNIFIED_MODE="y"

no_of_nodes=${SLURM_JOB_NUM_NODES:=1}
mpi_procs_pernode=128
((mpi_total_procs=no_of_nodes * mpi_procs_pernode))

export PYTHONPATH=$(cd ../..; pwd -P)

# ignore numpy longdouble incompatibility warning due to embedded python
export PYTHONWARNINGS="ignore:Signature:UserWarning"

srun -l --kill-on-bad-exit=1 \
     --nodes=${SLURM_JOB_NUM_NODES:-1} \
     --distribution=plane \
     --hint=nomultithread \
     --ntasks=$mpi_total_procs \
     --ntasks-per-node=${mpi_procs_pernode} \
     --cpus-per-task=${OMP_NUM_THREADS} \
     ./icon

