<h2 align="center">RePHO: Recovering Physically Plausible Human-Object Interactions from Monocular Videos</h2>

<p align="center">
  <a href="https://dingbang777.github.io/">Dingbang Huang</a>,
  <a href="https://www.cs.utexas.edu/~evouga/">Etienne Vouga</a>,
  <a href="https://www.cs.utexas.edu/~huangqx/">Qixing Huang</a>,
  <a href="https://geopavlakos.github.io/">Georgios Pavlakos</a>
  <br>
  CVPR 2026 (Highlight)
</p>

<p align="center">
<a href="https://dingbang777.github.io/RePHO/"><img src="https://img.shields.io/badge/Page-GitHub%20Pages-2ea44f?logo=githubpages&logoColor=white"></a>
<a href="https://arxiv.org/abs/2606."><img src="https://img.shields.io/badge/ArXiv-2605-%23B31C1C?logo=arxiv&logoSize=auto"></a>
<a href="https://dingbang777.github.io/RePHO/assets/SupMat.pdf"><img src="https://img.shields.io/badge/SupMat-PDF-d99a24?logo=adobeacrobatreader&logoColor=white"></a>
<a href="https://www.youtube.com/watch?v=suYGJYmFbyg"><img src="https://img.shields.io/badge/YouTube-Video-EA3323?style=flat&logo=youtube&logoColor=EA3323"></a>
<a href="#citation"><img src="https://img.shields.io/badge/BibTeX-Citation-4285f4?logo=googlescholar&logoColor=white"></a>
</p>

https://github.com/user-attachments/assets/4cd97d01-85c7-49e3-a095-6a93913bfa5e

## Installation

Here, we include a demo, running our optimization on an example from BEHAVE, where we provide the initialization from VisTracker. Please follow this for a minimal demo. Later, we provide instructions for how to run our code on arbitrary videos.

### 1. Download Body Models

<details>
<summary><strong>click open</strong></summary>

- From the [SMPL website](https://smpl.is.tue.mpg.de/), register/login and download the `SMPL` body model.
- From the [MANO website](https://mano.is.tue.mpg.de/), register/login and download: `MANO v1_2` and `SMPLH with Python Package`

After extraction, organize the required files as follows:

```text
body_models
├── mano_v1_2
│   └── models
│       ├── MANO_LEFT.pkl
│       ├── MANO_RIGHT.pkl
│       ├── SMPLH_female.pkl
│       └── SMPLH_male.pkl
├── smpl
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
├── smplh
│   ├── SMPLH_FEMALE.pkl
│   └── SMPLH_MALE.pkl
```

</details>

### 2. Install Isaac Gym Environment for Motion Tracking

Download Isaac Gym from the [official website](https://developer.nvidia.com/isaac-gym). Unzip the file and place the `isaacgym` folder in the main `RePHO` folder.

```bash
export CUDA_HOME='/your_local_cuda/cuda-12.1'
conda create -n RePHO_tracking python=3.8 -y
conda activate RePHO_tracking
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirement.txt
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py38_cu121_pyt241.tar.bz2
cd isaacgym/python
pip install -e .
conda deactivate
cd ../..
```

### 3. Download demo data

```bash
pip install -q gdown
mkdir -p data/downloads
gdown --fuzzy 'https://drive.google.com/file/d/1krzRglO1OiypOcI3oZo31VGS0CTLDdJ6/view?usp=drive_link' -O data/downloads/demo_data.zip
unzip -oq data/downloads/demo_data.zip -d data
```

### 4. Convert Reconstruction to Tracking Format

```bash
conda activate RePHO_tracking
bash scripts/convert_motion_format.sh Date03_Sub05_boxmedium_demo boxmedium
```

### 5. Run Motion Tracking Training
Download InterMimic pretrained checkpoint, our training initialized from this.

```bash
gdown --fuzzy 'https://drive.google.com/file/d/1rI8qJCsQFQaMjVjXp-JRwOAN7CazE62Y/view?usp=drive_link' -O data/downloads/intermimic_pretrained_checkpoint.zip
unzip -oq data/downloads/intermimic_pretrained_checkpoint.zip -d .
```

```bash
conda activate RePHO_tracking
bash scripts/train.sh Date03_Sub05_boxmedium_demo 0 #0 is gpu_id
```
The process will exit automatically when the training is done. 
If the you want to restart the training, you need to delete the whole sub-folder under `tracking_intermediate_file`

### 6. Run Motion Tracking Inference

```bash
bash scripts/infer_forward.sh Date03_Sub05_boxmedium_demo 0
```
You can see the rendered result at `data/demo-data/final_result`

If you are interested in the backward result, run:

```bash
bash scripts/infer_backward.sh Date03_Sub05_boxmedium_demo 0
```

## Running on new videos

If you want to run the complete pipeline on new videos, we provide here details for installing VisTracker. This will help you run the code on BEHAVE videos.

```bash
export CUDA_HOME='/your_local_cuda/cuda-11.8'
mkdir preprocess && cd preprocess
git clone -b RePHO_test_release --single-branch https://github.com/dingbang777/VisTracker.git
cd VisTracker
conda create -n RePHO_vistracker python=3.8 -y
conda activate RePHO_vistracker
bash install_RePHO_vistracker.sh
conda deactivate
cd ../..
```

### Run demo
```bash
cd preprocess/VisTracker
conda activate RePHO_vistracker
CUDA_VISIBLE_DEVICES=0 bash scripts/demo_RePHO.sh Date03_Sub05_boxmedium_demo
conda deactivate
cd ../..
```

## Running on in-the-wild videos
Finally, if you want to run on in-the-wild videos, i.e., cases where VisTracker is not able to handle out of the box (e.g., because you might not have a template), you additionally need to install and run:

<details>
<summary><strong>click open</strong></summary>

More required human body downloads:

- [`SMPLX`](https://download.is.tue.mpg.de/download.php?domain=smplx&resume=1&sfile=models_smplx_v1_1.zip)
- [`SMPL SPIN regressor`](http://visiondata.cis.upenn.edu/spin/data.tar.gz)
  - required files: `J_regressor_extra.npy`, `J_regressor_h36m.npy`
- [`SMPLX_flip`](https://download.is.tue.mpg.de/download.php?domain=smplx&resume=1&sfile=smplx_flip_correspondences.zip)
- [`SMPLX_to_J14`](https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPLX_to_J14.pkl)
- [`model_transfer`](https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=model_transfer.zip)
  - required files: `smplx2smpl_deftrafo_setup.pkl`, `smpl2smplx_deftrafo_setup.pkl`
- [`smpl_kid+rename`](https://download.is.tue.mpg.de/download.php?domain=agora&resume=1&sfile=smpl_kid_template.npy)
- [`smplx_kid+rename`](https://download.is.tue.mpg.de/download.php?domain=agora&resume=1&sfile=smplx_kid_template.npy)

Complete file structure with optional files:

```text
body_models
├── mano_v1_2
│   └── models
│       ├── MANO_LEFT.pkl
│       ├── MANO_RIGHT.pkl
│       ├── SMPLH_female.pkl
│       └── SMPLH_male.pkl
├── model_transfer
│   ├── smpl2smplx_deftrafo_setup.pkl
│   └── smplx2smpl_deftrafo_setup.pkl
├── smpl
│   ├── J_regressor_extra.npy
│   ├── J_regressor_h36m.npy
│   ├── kid_template.npy
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
├── smplh
│   ├── SMPLH_FEMALE.pkl
│   └── SMPLH_MALE.pkl
└── smplx
    ├── kid_template.npy
    ├── SMPLX_FEMALE.npz
    ├── SMPLX_FEMALE.pkl
    ├── smplx_flip_correspondences.npz
    ├── SMPLX_MALE.npz
    ├── SMPLX_MALE.pkl
    ├── SMPLX_NEUTRAL.npz
    ├── SMPLX_NEUTRAL.pkl
    └── SMPLX_to_J14.pkl
```


```bash
cd preprocess
#Install SAM3, SAM3d, rtmlib
conda create -n RePHO_preprocess python=3.11 -y
conda activate RePHO_preprocess
git clone https://github.com/dingbang777/sam-3d-objects.git
export CUDA_HOME=/lusr/cuda-12.1.1
export PATH=/lusr/cuda-12.1.1/bin:$PATH
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install 'setuptools<81'
python -m pip install appdirs
python -m pip install --no-build-isolation --prefer-binary nvidia-pyindex==1.0.9
PYTHONNOUSERSITE=1 python -m pip install auto-gptq==0.7.1 --prefer-binary
PYTHONNOUSERSITE=1 python -m pip install -r sam-3d-objects/requirements.txt
PYTHONNOUSERSITE=1 python -m pip install --no-build-isolation -r sam-3d-objects/requirements.p3d.txt
PYTHONNOUSERSITE=1 PIP_FIND_LINKS=https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html python -m pip install --no-build-isolation -r sam-3d-objects/requirements.inference.txt
PYTHONNOUSERSITE=1 python -m pip install 'git+https://github.com/autonomousvision/mip-splatting.git#subdirectory=submodules/diff-gaussian-rasterization' --no-build-isolation
PYTHONNOUSERSITE=1 python -m pip install 'git+https://github.com/NVlabs/nvdiffrast.git' --no-build-isolation
PYTHONNOUSERSITE=1 python sam-3d-objects/patching/hydra
PYTHONNOUSERSITE=1 python -m pip install -e ./sam-3d-objects --no-deps
(cd sam-3d-objects && hf download --repo-type model --local-dir checkpoints/hf-download --max-workers 1 facebook/sam-3d-objects && mv checkpoints/hf-download/checkpoints checkpoints/hf && rm -rf checkpoints/hf-download)
(cd sam-3d-objects && rm -rf checkpoints/hf && mv checkpoints/checkpoints checkpoints/hf)

PYTHONNOUSERSITE=1 python -m pip install -U ultralytics
PYTHONNOUSERSITE=1 python -m pip install git+https://github.com/ultralytics/CLIP.git
PYTHONNOUSERSITE=1 python -m pip install 'imageio[ffmpeg]'
HF_HUB_ENABLE_HF_TRANSFER=1 hf download --repo-type model facebook/sam3 sam3.pt --local-dir .

PYTHONNOUSERSITE=1 python -m pip install rtmlib -i https://pypi.org/simple
PYTHONNOUSERSITE=1 python -m pip install 'numpy<2' 'opencv-contrib-python<4.13'
PYTHONNOUSERSITE=1 python -m pip uninstall -y onnxruntime
PYTHONNOUSERSITE=1 python -m pip install onnxruntime-gpu==1.24.4

#Run SAM3, SAM3d, rtmlib
bash video_preprocess_example.sh \
  --input-video ../data/demo_data/input_video/${seq_name}.color.mp4 \
  --output-root ../data/demo_data/intermediate_process \
  --gpu-id ${gpu_id}


#Install GVHMR
git clone https://github.com/dingbang777/GVHMR.git
cd GVHMR

conda create -y -n RePHO_preprocess_gvhmr python=3.10
conda activate RePHO_preprocess_gvhmr

pip install -U gdown
pip install setuptools==69.5.1
pip install -r requirements.txt
pip install --no-build-isolation chumpy
pip install -e .

#Run GVHMR
CUDA_VISIBLE_DEVICES=${gpu_id} bash basedemo_commands_custom_new.sh ../../data/demo_data/intermediate_process/${seq_name}
conda deactivate
cd ..


#Run VisTracker (or you can turn to recent SOTA like CARI4d, which might be better initialization)
cd VisTracker
conda activate RePHO_vistracker
CUDA_VISIBLE_DEVICES=${gpu_id} bash scripts/run_custom_video_demo.sh ../../data/demo_data/intermediate_process/${seq_name}

```


</details>


## Citation

    @inproceedings{huang2026physicalhoi,
      title={Recovering Physically Plausible Human-Object Interactions from Monocular Videos},
      author={Huang, Dingbang and Vouga, Etienne and Huang, Qixing and Pavlakos, Georgios},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2026}
    }

## License

This repository is released under the [MIT License](LICENSE).

The VisTracker code used in the reconstruction pipeline is forked from the
VisTracker repository and remains subject to the original
VisTracker license. Please refer to the [VisTracker repository](https://github.com/xiexh20/VisTracker)
for its license terms.
