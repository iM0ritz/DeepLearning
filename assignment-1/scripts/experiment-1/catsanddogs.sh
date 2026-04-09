#!/bin/bash
#SBATCH --job-name=exp1
#SBATCH --output=ass1exp1/catsanddogs.out
#SBATCH --error=ass1exp1/catsanddogs.err
#SBATCH --time=04:00:00

python ass1exp1/catsanddogs.py data/Catsanddogs/