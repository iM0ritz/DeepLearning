#!/bin/bash
#SBATCH --job-name=exp-setup
#SBATCH --output=ass1setup/stanforddogs.out
#SBATCH --error=ass1setup/stanforddogs.err
#SBATCH --time=03:00:00

python ass1setup/stanforddogs.py data/Stanforddogs/