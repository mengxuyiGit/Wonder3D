# stage 2
accelerate launch --config_file 8gpu.yaml train_mvdiffusion_joint_splatter.py --config configs/train/stage2-joint-6views-lara.yaml # LARA