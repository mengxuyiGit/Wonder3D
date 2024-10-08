
# stage 1
export CUDA_LAUNCH_BLOCKING=1
accelerate launch --config_file 4gpu.yaml train_mvdiffusion_image.py --config configs/train/stage1-mix-6views-lvis.yaml
# accelerate launch --config_file 8gpu.yaml train_mvdiffusion_image.py --config configs/train/stage1-mix-6views-lvis.yaml

### debug
# accelerate launch --config_file 1gpu.yaml train_mvdiffusion_image.py --config configs/train/stage1-mix-6views-lvis-pretrained_unet.yaml