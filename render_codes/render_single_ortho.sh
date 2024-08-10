CUDA_VISIBLE_DEVICES=2 \
 blenderproc run --blender-install-path /mnt/kostas-graid/sw/envs/xuyimeng/software \
 blenderProc_ortho.py \
 --object_path /home/chenwang/data/objaverse_lvis_glbs/zebra/000-005/64350a72ad11412b9725721d8b27d225.glb --view 0 \
 --output_folder ./out_renderings/ \
 --object_uid c70e8817b5a945aca8bb37e02ddbc6f9 \
 --ortho_scale 1.35 \
 --resolution 512 \
#  --reset_object_euler