#!/bin/bash

# pre_process_poses.sh /workspace/datasets/tangerine
# pre_process_poses.sh /workspace/datasets/cctv 
# pre_process_poses.sh /workspace/datasets/cube_2
# convert_to_tnt.sh /workspace/datasets/tangerine
# convert_to_tnt.sh /workspace/datasets/cctv
# convert_to_tnt.sh /workspace/datasets/cube_2

# cd /workspace/svox2/opt


# python autotune.py -g "0 1" "/workspace/svox2/opt/tasks/datasets.json" 


# datasets=("cactus" "cctv" "tangerine" "cube_2" "pen_cup_2" "dog")

datasets=("cactus"  "cctv"  "dog" "tangerine" "pen_cup_2")
grid_dims=("256" "128" "64")


for dataset in ${datasets[@]}; do
  COPY_DIR=/workspace/data/mask_$dataset
  mkdir $COPY_DIR
  cp /workspace/datasets/$dataset/result/voxel/vox_mask_debug.vox $COPY_DIR
  cp /workspace/datasets/$dataset/result/vox_masked.vox $COPY_DIR
  # python get_voxel_data_palette_debug.py --checkpoint /workspace/datasets/$dataset/ckpt/std/ckpt.npz --data_dir /workspace/datasets/$dataset
  # python ../voxel_mask.py --checkpoint /workspace/datasets/$dataset/ckpt/std/ckpt.npz --data_dir /workspace/datasets/$dataset
  # python /workspace/aseeo-research/RLResearch/workflow/mask.py \
  # --image_data_path="/workspace/datasets/$dataset/source/images_undistorted" \
  # --output_folder="/workspace/datasets/$dataset/source/masks"

  # for grid_dim in ${grid_dims[@]}; do
  #   python get_voxel_data_palette_debug.py --checkpoint /workspace/datasets/$dataset/ckpt/std/ckpt.npz --grid_dim $grid_dim --data_folder /workspace/datasets/$dataset 
  # done
done

#https://opensource.com/article/18/5/you-dont-know-bash-intro-bash-arrays