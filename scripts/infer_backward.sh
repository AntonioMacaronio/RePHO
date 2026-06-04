#!/bin/bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
seq_name=$1
gpu_id=$2
out_root=tracking_intermediate_file
motion_root=data/demo_data/output
cfg_env=intermimic/data/cfg/omomo_train_917.yaml
cfg_train=intermimic/data/cfg/train/rlg/omomo.yaml
out_images_root=data/demo_data/output/final_result/backward
checkpoint_dir=${out_root}/${seq_name}_dual/backward/smplx/nn
current_epoch=
min_epoch=54500
current_epoch=$(
    CHECKPOINT_DIR="${checkpoint_dir}" \
    REF_TAR_DIR="${out_root}/${seq_name}_dual/backward/ref_tar" \
    MIN_EPOCH="${min_epoch}" \
    python - <<'PY'
import os
from pathlib import Path

import torch

checkpoint_dir = Path(os.environ["CHECKPOINT_DIR"])
ref_tar_dir = Path(os.environ["REF_TAR_DIR"])
min_epoch = int(os.environ["MIN_EPOCH"])

best_epoch = None
best_ones = None

for checkpoint_path in sorted(checkpoint_dir.glob("mimic_000*.pth")):
    checkpoint_name = checkpoint_path.name
    epoch_text = checkpoint_name[len("mimic_000"):-len(".pth")]
    if not epoch_text.isdigit():
        continue

    epoch = int(epoch_text)
    if epoch < min_epoch:
        continue

    intermimic_path = ref_tar_dir / f"ref_tar_{epoch}" / "intermimic.pt"
    if not intermimic_path.exists():
        continue

    tensor = torch.load(intermimic_path, map_location="cpu")
    if getattr(tensor, "ndim", 0) < 2:
        continue

    ones_count = int((tensor[:, -1] == 1).sum().item())
    if best_ones is None or ones_count > best_ones or (ones_count == best_ones and epoch > best_epoch):
        best_epoch = epoch
        best_ones = ones_count

if best_epoch is not None:
    print(best_epoch)
PY
)

if [ -z "${current_epoch}" ]; then
    echo "No eligible checkpoint found in ${checkpoint_dir} with intermimic.pt and epoch >= ${min_epoch}" >&2
    exit 1
fi

python intermimic/run.py \
    --task InterMimic \
    --cfg_env "${cfg_env}" \
    --cfg_train "${cfg_train}" \
    --headless \
    --output_path "${out_images_root}" \
    --stateInit Start \
    --init_range_left 0 \
    --reverse_time \
    --device_id "${gpu_id}" \
    --rl_device "cuda:${gpu_id}" \
    --motion_file "${motion_root}/${seq_name}" \
    --sub_file_name intermimic_vistracker \
    --checkpoint "${checkpoint_dir}/mimic_000${current_epoch}.pth" \
    --hoi_refs_path "${out_root}/${seq_name}_dual/backward/ref_hoi/ref_hoi_${current_epoch}.npz" \
    --hoi_data_path "${out_root}/${seq_name}_dual/backward/hoi_data/intermimic_${current_epoch}.pt" \
    --test --num_envs 1 --enable_camera_sensors --save_images
    
python scripts/image_to_video.py --inputdir ${out_images_root}/camera_images --outputfile ${out_images_root}/camera_images.mp4
