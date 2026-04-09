#!/bin/bash
#SBATCH --job-name=exp-2
#SBATCH --output=ass1exp2/pretrained_catsanddogs.out
#SBATCH --error=ass1exp2/pretrained_catsanddogs.err
#SBATCH --time=03:00:00

python ass1exp2/pretrained_catsanddogs.py data/Catsanddogs/