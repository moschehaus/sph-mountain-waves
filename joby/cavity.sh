#!/bin/bash

#SBATCH --job-name=SPH_cav
#SBATCH --output=res_SPH_cav.txt
#SBATCH -n 48
#SBATCH --time=4:30:00
#SBATCH -p express
#SBATCH --constraint="InfiniBand"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=belank@karlin.mff.cuni.cz

module add Arch/linux-ubuntu20.04-sandybridge
module load julia
julia -t 48 cav.jl
