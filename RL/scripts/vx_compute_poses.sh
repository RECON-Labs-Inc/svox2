#####################################
#  COMPUTE POSES FOR VOXELIZATION   #
#####################################

# Computes poses for voxelization. Single pose computation (no undistortion). In the end

# USAGE

# ./vx_compute_poses.sh <PROJECT_FOLDER> <[OPTIONAL] LOG_FILE> <MS_DOWNSAMPLING>

## Check if env vars exist
if ! [ -z $PROJECT_FOLDER ]
then
    echo
    echo "PROJECT_FOLDER env var exists, using it."
    echo $PROJECT_FOLDER
fi

PROJECT_FOLDER=${PROJECT_FOLDER:-$1}
LOG_FILE=${2:-$PROJECT_FOLDER"/logs/pipeline/pre_process_poses.log"}

echo
echo $LOG_FILE
echo

cd $RL_RUNPATH
MS_DOWNSAMPLING=${3:-2}
echo "Pre-proccessing logging to "$LOG_FILE
echo "Using MS downsampling: "$MS_DOWNSAMPLING

python -m workflow.compute_poses --image_data_path $PROJECT_FOLDER/source/images --chunk_type raw_poses --remove_unaligned --out_dir $PROJECT_FOLDER/project_files --ms_downscale $MS_DOWNSAMPLING
python -m workflow.export_cam_poses --ms_filename ms_poses.psz --chunk_type raw_poses --cam_json_filename cam_data.json --out_dir $PROJECT_FOLDER/project_files
python -m workflow.write_calibration --ms_filename ms_poses.psz --chunk_type raw_poses --cam_json_filename cam_data.json --out_dir $PROJECT_FOLDER/project_files

EXIT_STATUS=${PIPESTATUS[0]:-1}

if [ $EXIT_STATUS -eq 0 ]
then
    echo "Preprocess poses was successful"
else
    echo "Preprocess poses failed"
fi


exit $EXIT_STATUS

