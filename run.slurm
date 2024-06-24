#! /bin/sh

#SBATCH --account=gpu-research
#SBATCH --job-name=asaf_refrec

#SBATCH --partition=gpu-a100-killable # (see resources section)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks-per-node=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100

#SBATCH --output=/home/yandex/DL20232024a/asafavrahamy/Projects/RefRec/logs/refrec_imprv.out # redirect stdout
#SBATCH --error=/home/yandex/DL20232024a/asafavrahamy/Projects/RefRec/logs/refrec_imprv.err # redirect stderr

bash /home/yandex/DL20232024a/asafavrahamy/Projects/RefRec/run.sh

