current_dir=$(pwd)
dataset_folder=/workspace/datasets


# ------- BENCHMARK --------------------
video=${1:-/workspace/data/cctv.mov}
# video=/workspace/data/cctv.mov
# video="/workspace/datasets/nike_shoe.mov"
dataset=${2:-'cctv_30mar_130'}
# dataset='cctv_30mar_130'
num_frames=130
ms_downscale=2

echo $video
echo $dataset

make_dirs.sh $dataset_folder/$dataset
cp /workspace/svox2/RL/voxelizing_pipeline.sh $dataset_folder/$dataset/project_files/voxelizing_pipeline_frozen.sh

# Tick total time
tic.sh $dataset_folder/$dataset/logs/total_time_log.log

echo
echo $video
echo $num_frames
echo $dataset

echo 
echo "Extract frames"
log_file=$dataset_folder/$dataset/logs/extract_frames_time.log
tic.sh $log_file
extract_frames.sh $video $num_frames $dataset_folder/$dataset
toc.sh $log_file


echo
echo "Preprocess_poses"
log_file=$dataset_folder/$dataset/logs/pre_process_poses_time.log
tic.sh $log_file
#pre_process_poses.sh /workspace/datasets/$dataset /workspace/datasets/$dataset/logs/pipeline/pre_process_poses.log
cd $RL_RUNPATH
python -m workflow.compute_poses --image_data_path $dataset_folder/$dataset/source/images --chunk_type raw_poses --remove_unaligned --out_dir $dataset_folder/$dataset/project_files --ms_downscale $ms_downscale
python -m workflow.export_cam_poses --ms_filename ms_poses.psz --chunk_type raw_poses --cam_json_filename cam_data.json --out_dir $dataset_folder/$dataset/project_files
python -m workflow.write_calibration --ms_filename ms_poses.psz --chunk_type raw_poses --cam_json_filename cam_data.json --out_dir $dataset_folder/$dataset/project_files
toc.sh $log_file

cd $current_dir
echo
echo "Convert to tnt"
time convert_to_tnt.sh /workspace/datasets/$dataset /workspace/datasets/$dataset "images"


echo
echo "Train"
cd /workspace/svox2/opt
experiment=std

# # # TRAIN


CKPT_DIR=$dataset_folder/$dataset/ckpt/$experiment
mkdir -p $CKPT_DIR
NOHUP_FILE=$CKPT_DIR/log
echo CKPT $CKPT_DIR
echo LOGFILE $NOHUP_FILE
config=fastest.json

log_file=$dataset_folder/$dataset/logs/train_time.log
tic.sh $log_file
# --data_dir $dataset_folder/$dataset -c configs/fastest.json
time CUDA_VISIBLE_DEVICES=0 nohup python -u opt.py -t $CKPT_DIR $dataset_folder/$dataset -c configs/fastest.json --log_depth_map > $NOHUP_FILE
toc.sh $log_file

log_file=$dataset_folder/$dataset/logs/post_train_time.log
tic.sh $log_file

cd $current_dir
echo
echo "make voxels"
log_file=$dataset_folder/$dataset/logs/make_voxel.log
tic.sh $log_file
time python voxel_make.py --checkpoint /workspace/datasets/$dataset/ckpt/std/ckpt.npz --data_dir /workspace/datasets/$dataset --debug_folder /workspace/data/$dataset
toc.sh $log_file


cd $current_dir

echo
echo "Mask voxels"
log_file=$dataset_folder/$dataset/logs/mask_voxels_time.log

tic.sh $log_file
time python ../voxel_mask.py --checkpoint /workspace/datasets/$dataset/ckpt/std/ckpt.npz --data_dir /workspace/datasets/$dataset --source images --debug_folder /workspace/data/$dataset
toc.sh $log_file


echo
echo "Fill voxels"
log_file=$dataset_folder/$dataset/logs/fill_voxels_time.log

tic.sh $log_file
time python ../voxel_mask.py --vox_file /workspace/datasets/$dataset/result/vox_masked.vox --checkpoint /workspace/datasets/$dataset/ckpt/std/ckpt.npz --data_dir /workspace/datasets/$dataset --source images --use_block --debug_folder /workspace/data/$dataset
toc.sh $log_file

echo
echo "Push voxels"
log_file=$dataset_folder/$dataset/logs/push_voxels_time.log

tic.sh $log_file
# time python ../voxel_push.py --vox_file /workspace/datasets/$dataset/result/vox_masked_block.vox --ref_vox_file /workspace/datasets/$dataset/result/vox_masked.vox --data_dir /workspace/datasets/$dataset --debug_folder /workspace/data/$dataset
time python ../voxel_push.py --vox_file /workspace/datasets/$dataset/result/vox_masked.vox --ref_vox_file /workspace/datasets/$dataset/result/vox_masked.vox --data_dir /workspace/datasets/$dataset --debug_folder /workspace/data/$dataset --use_block
toc.sh $log_file

echo
echo "Colorize voxels"
log_file=$dataset_folder/$dataset/logs/colorize_voxels_time.log

tic.sh $log_file
time python voxel_colorize.py --vox_file /workspace/datasets/$dataset/result/voxel/vox_pushed.vox --data_dir /workspace/datasets/$dataset --debug_folder /workspace/data/$dataset
toc.sh $log_file

# Post train time]
log_file=$dataset_folder/$dataset/logs/post_train_time.log
toc.sh $log_file

# Toc total time
toc.sh $dataset_folder/$dataset/logs/total_time_log.log