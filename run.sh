#!/bin/bash
mamba init
bash
mamba activate deep_env
#this is to change the default path for caches
export HF_HOME=/home/yandex/DL20232024a/asafavrahamy/.cache/huggingface
export MPLCONFIGDIR=/home/yandex/DL20232024a/asafavrahamy/.cache/matplotlib

nvidia-smi

cd /home/yandex/DL20232024a/asafavrahamy/Projects/RefRec;
python main.py

# cd /home/yandex/DL20232024a/asafavrahamy/Projects/RefRec/preprocess;
# python generate_data_and_prompt.py