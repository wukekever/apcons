#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES="0,1" # use gpu 0 and 1 

TIMESTAMP="$(date --iso-8601="seconds")"
DATA_PATH=${1:-"./data/"} # DATA_PATH=${1:-"/workspaces/DeepLTE/data/"}
# RESTORE_DIR=${1:-"/workspaces/DeepLTE/ckpts/square_full_it_2023-05-13T10:29:00/models/latest/step_33600_2023-05-13T10:44:39"}

# python run_deeplte.py \
# 	--config=deeplte/config.py \
# 	--data_path="${DATA_PATH}" \
# 	--config.checkpoint_dir="./ckpts/square_full_it_${TIMESTAMP%+*}" \
# 	--jaxline_mode="train_eval_multithreaded" \
# 	--alsologtostderr="true"

python run_deeplte.py \
	--config=deeplte/config.py \
	--data_path="${DATA_PATH}" \
	--config.checkpoint_dir="./ckpts/square_full_it_${TIMESTAMP%+*}" \
	--jaxline_mode="train" \
	--alsologtostderr="true"