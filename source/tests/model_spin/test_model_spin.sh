#!/bin/sh

#SBATCH -J test
#SBATCH --gpus 1

python -u ../test_model_spin.py > test.log 2>&1

