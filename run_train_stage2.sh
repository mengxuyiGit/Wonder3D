# stage 2
# export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=4,5
accelerate launch --config_file 1gpu.yaml train_mvdiffusion_joint_splatter.py --config configs/train/stage2-joint-6views-lara_camNormFalse_cat3d.yaml # Cat3D, LARA, splatter with camNorm=False
