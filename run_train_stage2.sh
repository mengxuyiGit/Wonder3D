# stage 2
# export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=0
accelerate launch --config_file 4gpu.yaml train_mvdiffusion_joint.py --config configs/train/stage2-joint-6views-lvis.yaml