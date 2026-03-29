#!/bin/bash
#SBATCH --job-name=exp1
#SBATCH --output=catsanddogs.out
#SBATCH --error=catsanddogs.err
#SBATCH --time=04:00:00

python catsanddogs.py Catsanddogs/