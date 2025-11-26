#!/bin/bash -l
#SBATCH --job-name=cylinder_classical_sph_job
#SBATCH --output=cylinder_classical_sph.%j.out
#SBATCH --error=cylinder_classical_sph.%j.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --partition=express3

#SBATCH --mail-type=ALL
#SBATCH --mail-user=belank@karlin.mff.cuni.cz

echo "=== JOB START $(date) on $(hostname) ==="

module load julia/1.9.1
module load spack-bin-julia/.1.9.1-gcc9.4.0-uzbijvaynhque2ch

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Running on $(hostname) with $JULIA_NUM_THREADS threads"
echo "Julia binary: $(which julia)"

# cd /path/to/your/julia/project

julia -t $JULIA_NUM_THREADS job_classical.jl

echo "=== JOB END $(date) ==="

