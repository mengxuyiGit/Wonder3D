# stage 2
# export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=1
# export CUDA_VISIBLE_DEVICES=6,7
# accelerate launch --config_file 1gpu.yaml train_mvdiffusion_joint.py --config configs/train/stage2-joint-6views-lvis.yaml
# accelerate launch --config_file 1gpu.yaml train_mvdiffusion_joint_splatter.py --config configs/train/stage2-joint-6views-lvis.yaml
# accelerate launch --config_file 1gpu.yaml train_mvdiffusion_joint_splatter.py --config configs/train/stage2-joint-6views-lvis-render_2dgs.yaml
# accelerate launch --config_file 8gpu.yaml train_mvdiffusion_joint_splatter.py --config configs/train/stage2-joint-6views-lvis-render_3dgs.yaml # LVIS
# accelerate launch --config_file 1gpu.yaml train_mvdiffusion_joint_splatter.py --config configs/train/stage2-joint-6views-lara-overfit.yaml # LARA
accelerate launch --config_file 8gpu.yaml train_mvdiffusion_joint_splatter.py --config configs/train/stage2-joint-6views-lara.yaml # LARA