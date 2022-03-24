current_dir=$(pwd)
$dataset_folder=/workspace/datasets


## ---------------- PREPARE DATA ---------

# video="/workspace/data/cctv.mov"
# dataset='/workspace/datasets/_t_shoe_100'


# make_dirs.sh $dataset

# echo $video
# echo $num_frames
# echo $dataset

# time extract_frames.sh $video $num_frames $dataset


## ----------- EXTRACT FRAMES -----------

# num_frames=(30 60)
# prefix="cctv"
# video="/workspace/data/cctv.mov"

# for nf in ${num_frames[@]}; do
#     dataset="/workspace/datasets/_t_"$prefix"_"$nf
#     make_dirs.sh $dataset
#     echo "Extracting "$nf" frames"
#     time extract_frames.sh $video $nf /workspace/datasets/"_t_"$prefix"_"$nf
# done

## ----------- PREPROCESS POSES ----------



# datasets=("_t_shoe_30" "_t_shoe_100" ) 
# datasets=("_t_cctv_60" ) 


# for dataset in ${datasets[@]}; do
    
#     echo $dataset
#     time pre_process_poses.sh /workspace/datasets/$dataset /workspace/datasets/$dataset/logs/pipeline/pre_process_poses.log > /workspace/datasets/$dataset/logs/pipeline/pre_process_time.log
 
# done

# ------- BENCH MARK --------------------

video="/workspace/data/cctv.mov"
dataset='/workspace/datasets/_t_shoe_100'
num_frames=200

make_dirs.sh $dataset

echo $video
echo $num_frames
echo $dataset

echo "Extract frames"
time extract_frames.sh $video $num_frames $dataset

echo "Preprocess_poses"
time pre_process_poses.sh /workspace/datasets/$dataset /workspace/datasets/$dataset/logs/pipeline/pre_process_poses.log > /workspace/datasets/$dataset/logs/pipeline/pre_process_time.log

echo "Convert to tnt"
time convert_to_tnt.sh /workspace/datasets/$dataset

echo "Train"
cd /workspace/svox2/opt
time ./launch.sh std 0 /workspace/datasets/$dataset -c configs/fastest.json
cd $current_dir

echo "Make masks"
cd $RL_RUNPATH
python -m workflow.mask --image_data_path /workspace/datasets/$dataset
cd $current_dir

echo "Get voxels"
python get_voxel_data_palette_debug.py --checkpoint /workspace/datasets/$dataset/ckpt/std/ckpt.npz --data_dir /workspace/datasets/$dataset
python ../voxel_mask.py --checkpoint /workspace/datasets/$dataset/ckpt/std/ckpt.npz --data_dir /workspace/datasets/$dataset