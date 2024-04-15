#!/bin/bash

#SBATCH --job-name=SPHcavity-test
#SBATCH --output=res-SPHcavity-test.txt
#SBATCH -N 1
#SBATCH -n 36
#SBATCH --time=30:00
#SBATCH -p express
#SBATCH --constraint="InfiniBand"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=belank@karlin.mff.cuni.cz

module add julia
cd ~/usr/users/belank/examples
julia -t 36 SPHcavity-test.jl
