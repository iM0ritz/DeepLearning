#!/bin/bash
#SBATCH --job-name=exp-3
#SBATCH --output=ass1exp3/pretrained_catsanddogs2.out
#SBATCH --error=ass1exp3/pretrained_catsanddogs2.err
#SBATCH --time=04:00:00
#SBATCH --partition priority

python ass1exp3/pretrained_catsanddogs2.py data/Catsanddogs/