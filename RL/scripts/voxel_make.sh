##################
#  MAKE VOXEL    #
##################
#  DESCRIPTION
#  Makes a voxel from a training checkpoint of plenoxel, by sampling its grid.

# USAGE:
# voxel_make.sh <PROJECT_FOLDER> <GRID_DIM> <EULER_ANGLES>
# voxel_make.sh /workspace/datasets/cactus 128 "30 30 30"


## Check if env vars exist
if ! [ -z $PROJECT_FOLDER ]
then
    echo
    echo "PROJECT_FOLDER env var exists, using it."
    echo $PROJECT_FOLDER
    echo
fi
echo "In voxel make.sh"
PROJECT_FOLDER=${PROJECT_FOLDER:-$1}
EULER_ANGLES=${3:-"0 0 0"}

GRID_DIM=${2:-128}
echo
echo "Grid dimension: "$GRID_DIM

cd $PX_RUNPATH
echo
echo "Make voxels"
LOG_FILE=$PROJECT_FOLDER"/logs/voxel_make.log"
time_log_file=$PROJECT_FOLDER/logs/time_make_voxel.log
total_log=$PROJECT_FOLDER/logs/time_total.log

echo $PROJECT_FOLDER
echo $time_log_file
echo $EULER_ANGLES
tic.sh $time_log_file

time python voxel_make.py --checkpoint $PROJECT_FOLDER/ckpt/std/ckpt.npz \
                            --euler_angles $EULER_ANGLES --data_dir $PROJECT_FOLDER --grid_dim $GRID_DIM 2>&1 | tee "$LOG_FILE"
EXIT_STATUS=${PIPESTATUS[0]:-1}

toc.sh $time_log_file $total_log

if [ $EXIT_STATUS -eq 0 ]
then
    echo "Voxel making was was successful"
else
    echo "Voxel making failed"
fi

exit $EXIT_STATUS
