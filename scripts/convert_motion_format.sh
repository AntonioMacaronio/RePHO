#!/bin/bash
set -euo pipefail
name=$1
obj_name=$2
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p data/demo_data/output/${name}/intermimic_gt
ln -sfn SMPLH_MALE.pkl body_models/smplh/SMPLH_NEUTRAL.pkl #placeholder for now
cp data/demo_data/intermediate_process/${name}/cam.npz data/demo_data/output/${name}/intermimic_gt/cam.npz
cp data/demo_data/intermediate_process/${name}/contact.pt data/demo_data/output/${name}/intermimic_gt/contact.pt

python scripts/vistracker_to_interact_format.py --recon_file data/demo_data/intermediate_process/${name}/vistracker_result/recon_test-releasev2/${name}_k1.pkl --save_name intermimic_vistracker --output_path data/demo_data/output/${name} --obj_name ${obj_name} 
python scripts/interact_to_tracking_format.py --input_dir data/demo_data/output/${name} --type smplh --save_name intermimic_vistracker --obj_name ${obj_name} --load_contact_from intermimic_gt