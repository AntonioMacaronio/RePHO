#!/bin/bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
seq_name=$1
gpu_id=$2
out_root=tracking_intermediate_file
motion_root=data/demo_data/output
cfg_env=intermimic/data/cfg/omomo_train_917.yaml
cfg_train=intermimic/data/cfg/train/rlg/omomo.yaml

config_json="${out_root}/${seq_name}_dual/forward/config.json"
mkdir -p "${out_root}/${seq_name}_dual/forward"

cat > "${config_json}" << EOF
{
  "seq_name": "${seq_name}",
  "gpu_id": "${gpu_id}",
  "out_root": "${out_root}",
  "motion_root": "${motion_root}",
  "cfg_env": "${cfg_env}",
  "cfg_train": "${cfg_train}"
}
EOF

CUDA_VISIBLE_DEVICES=${gpu_id} python intermimic/run.py \
    --task InterMimic \
    --cfg_env ${cfg_env} \
    --cfg_train ${cfg_train} \
    --headless \
    --output_path ${out_root}/${seq_name}_dual/forward \
    --stateInit Traverse_Random \
    --no_reverse_time \
    --device_id 0 \
    --rl_device cuda:0 \
    --motion_file ${motion_root}/${seq_name} \
    --sub_file_name intermimic_vistracker
