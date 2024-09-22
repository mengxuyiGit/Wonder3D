# stage 2
# export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=6,7
# accelerate launch --config_file 1gpu.yaml train_mvdiffusion_joint.py --config configs/train/stage2-joint-6views-lvis.yaml
# accelerate launch --config_file 1gpu.yaml train_mvdiffusion_joint_splatter.py --config configs/train/stage2-joint-6views-lvis.yaml
accelerate launch --config_file 2gpu.yaml train_mvdiffusion_joint_splatter.py --config configs/train/stage2-joint-6views-lvis-render_2dgs.yaml