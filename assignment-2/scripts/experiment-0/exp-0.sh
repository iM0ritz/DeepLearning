#!/bin/bash
#SBATCH --job-name=exp0
#SBATCH --output=ass2exp0/exp-0.out
#SBATCH --error=ass2exp0/exp-0.err
#SBATCH --time=04:00:00
#SBATCH --partition priority

python ass2exp0/base_code.py data/spa-eng/