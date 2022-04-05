#!/bin/bash

datasets=("_t_cactus_color" "_t_shoe_color" "_t_cctv_color")
# datasets=("_t_cctv_color")
# grid_dims=("256" "128" "64")
dataset_folder=/workspace/datasets
n_clusters=8
for dataset in ${datasets[@]}; do
  # python voxel_colorize.py --vox_file $dataset_folder/$dataset/result/voxel/vox_pushed.vox --data_dir $dataset_folder/$dataset --debug_folder /workspace/data/$dataset --color_mode classifier 
  # voxelizing_pipeline.sh sdfsdf $dataset_folder/$dataset
  python ../voxel_push.py --vox_file $dataset_folder/$dataset/result/vox_masked.vox --ref_vox_file /workspace/datasets/$dataset/result/vox_masked.vox --data_dir /workspace/datasets/$dataset --debug_folder /workspace/data/$dataset --use_block 
  python voxel_colorize.py --vox_file $dataset_folder/$dataset/result/voxel/vox_pushed.vox --data_dir /workspace/datasets/$dataset --debug_folder /workspace/data/$dataset
done

#https://opensource.com/article/18/5/you-dont-know-bash-intro-bash-arrays