#!/bin/bash

seq_name=$1
gpu_id=$2
bash scripts/train_dual_forward.sh ${seq_name} ${gpu_id} & sleep 30s && bash scripts/train_dual_backward.sh ${seq_name} ${gpu_id}