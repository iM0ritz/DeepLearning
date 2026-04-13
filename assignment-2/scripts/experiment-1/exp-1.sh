#!/bin/bash
#SBATCH --job-name=exp1
#SBATCH --output=ass2exp1/exp-1.out
#SBATCH --error=ass2exp1/exp-1.err
#SBATCH --time=04:00:00
#SBATCH --partition priority

python ass2exp1/train.py data/spa-eng/