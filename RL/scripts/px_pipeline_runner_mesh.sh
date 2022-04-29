########################################
#  PLENOXEL VOXELIZE PIPELINE RUNNER   #
########################################
# DESCRIPTION
# Will make a voxelized object from a video file, in a single command

# USAGE:
# px_pipeline_runner.sh <VIDEO_FILE> <PROJECT_FOLDER> <TRAIN_CONFIG_FILE_N> <GRID_DIM> <EULER_ANGLES> <COL_METHOD> <[OPT] CUDA_DEVICE>

## Check if env vars exist
if ! [ -z $PROJECT_FOLDER ]
then
    echo
    echo "PROJECT_FOLDER env var exists, using it."
    echo $PROJECT_FOLDER
    echo
fi

VIDEO_FILE=$1
PROJECT_FOLDER=${PROJECT_FOLDER:-$2} # Useless???
TRAIN_CONFIG_FILE_N=$3
GRID_DIM=$4
EULER_ANGLES=$5
COL_METHOD=$6
CUDA_DEVICE=$7
CUDA_DEVICE=${CUDA_DEVICE:-0}

NUM_FRAMES=130

echo "Video file $PROJECT_FOLDER"
echo "Train config file $TRAIN_CONFIG_FILE_N"
echo "Grid dim $GRID_DIM"
echo "Euler angles $EULER_ANGLES"
echo "Color method $COL_METHOD"


# prepare_data.sh $PROJECT_FOLDER $VIDEO_FILE $NUM_FRAMES


# if ! [ "$?" -eq 0 ]
# then
#     echo "Prepare data failed"
#     exit 1
# fi

# vx_pre_process_poses.sh $PROJECT_FOLDER


# if ! [ "$?" -eq 0 ]
# then
#     echo "Pose pre-processing failed"
#     exit 1
# fi

CHUNK_TYPE="raw_poses"
build_export.sh $PROJECT_FOLDER $PROJECT_FOLDER/logs/build_export.log $CHUNK_TYPE "3 3 3"


if ! [ "$?" -eq 0 ]
then
    echo "Training failed"
    exit 1
fi
echo "$PROJECT_FOLDER" "$GRID_DIM" "$EULER_ANGLES" "$COL_METHOD"
vx_post_process.sh "$PROJECT_FOLDER" "$GRID_DIM" "$EULER_ANGLES" "$COL_METHOD" "$CUDA_DEVICE"


if ! [ "$?" -eq 0 ]
then
    echo "Post processing failed"
    exit 1
fi
