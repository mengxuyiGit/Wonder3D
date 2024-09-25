
# stage 1
export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=2,3,4,5
# accelerate launch --config_file 2gpu.yaml train_mvdiffusion_image.py --config configs/train/stage1-mix-6views-lvis.yaml
# accelerate launch --config_file 8gpu.yaml train_mvdiffusion_image.py --config configs/train/stage1-mix-6views-lvis.yaml
accelerate launch --config_file 8gpu.yaml train_mvdiffusion_image.py --config configs/train/stage1-mix-6views-lvis-3dgs.yaml

### debug
# accelerate launch --config_file 1gpu.yaml train_mvdiffusion_image.py --config configs/train/stage1-mix-6views-lvis-pretrained_unet.yaml