#!/bin/bash
#SBATCH --job-name=exp-4
#SBATCH --output=ass1exp4/pretrained_catsanddogs3.out
#SBATCH --error=ass1exp4/pretrained_catsanddogs3.err
#SBATCH --time=04:00:00
#SBATCH --partition priority

python ass1exp4/pretrained_catsanddogs3.py data/Catsanddogs/