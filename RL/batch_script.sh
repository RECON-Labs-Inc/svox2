#!/bin/bash

# pre_process_poses.sh /workspace/datasets/tangerine
# pre_process_poses.sh /workspace/datasets/cctv 
# pre_process_poses.sh /workspace/datasets/cube_2
# convert_to_tnt.sh /workspace/datasets/tangerine
# convert_to_tnt.sh /workspace/datasets/cctv
# convert_to_tnt.sh /workspace/datasets/cube_2

# cd /workspace/svox2/opt

# python autotune.py -g "0 1" "/workspace/svox2/opt/tasks/datasets.json" 


datasets=("cactus" "cctv" "tangerine" "cube_2" "pen_cup_2")


for dataset in ${datasets[@]}; do
  python get_voxel_data_palette_debug.py --checkpoint /workspace/datasets/$dataset/ckpt/std/ckpt.npz
done

#https://opensource.com/article/18/5/you-dont-know-bash-intro-bash-arrays