#!/bin/bash
#SBATCH --account=PCON0003
#SBATCH --job-name=dc1_real_v1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=8
#SBATCH --output=dc1_real_v1.out

python cocoa_dcsolver.py real 1 10
