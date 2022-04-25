########################################
#  PLENOXEL VOXELIZE PIPELINE RUNNER   #
########################################
# DESCRIPTION
# Will make a voxelized object from a video file, in a single command

# USAGE:
# px3d_pipeline_runner.sh <VIDEO_FILE> <PROJECT_FOLDER> <TRAIN_CONFIG_FILE_N> <[OPT] CUDA_DEVICE>

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
CUDA_DEVICE=$4
CUDA_DEVICE=${CUDA_DEVICE:-0}

NUM_FRAMES=130

echo "Video file $PROJECT_FOLDER"
echo "Train config file $TRAIN_CONFIG_FILE_N"
echo "Grid dim $GRID_DIM"
echo "Euler angles $EULER_ANGLES"
echo "Color method $COL_METHOD"


prepare_data.sh $PROJECT_FOLDER $VIDEO_FILE $NUM_FRAMES


if ! [ "$?" -eq 0 ]
then
    echo "Prepare data failed"
    exit 1
fi

pre_process_poses.sh $PROJECT_FOLDER


if ! [ "$?" -eq 0 ]
then
    echo "Pose pre-processing failed"
    exit 1
fi

convert_to_tnt.sh $PROJECT_FOLDER $PROJECT_FOLDER images_undistorted

if ! [ "$?" -eq 0 ]
then
    echo "Convert poses failed"
    exit 1
fi

vx_train.sh -f $PROJECT_FOLDER -n $TRAIN_CONFIG_FILE_N -u $CUDA_DEVICE


if ! [ "$?" -eq 0 ]
then
    echo "Training failed"
    exit 1
fi
echo "$PROJECT_FOLDER" "$GRID_DIM" "$EULER_ANGLES" "$COL_METHOD"

vx_render_depth.sh $PROJECT_FOLDER $CUDA_DEVICE

if ! [ "$?" -eq 0 ]
then
    echo "Render depth failed"
    exit 1
fi

depthmap_replace.sh $PROJECT_FOLDER 1

if ! [ "$?" -eq 0 ]
then
    echo "Depthmap replacement failed"
    exit 1
fi

build_export.sh $PROJECT_FOLDER


if ! [ "$?" -eq 0 ]
then
    echo "Build export failed"
    exit 1
else
    echo "Build and export succesful."
fi
