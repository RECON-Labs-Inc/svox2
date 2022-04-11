##################
#  MASK VOXEL    #
##################
# DESCRIPTION
# Removes artifacts from a voxelized object by computing masks
# USAGE:
# voxel_mask.sh <PROJECT_FOLDER> <NUM_MASKS> <MASK_THRES>


## Check if env vars exist
if ! [ -z $PROJECT_FOLDER ]
then
    echo
    echo "PROJECT_FOLDER env var exists, using it."
    echo $PROJECT_FOLDER
    echo
fi

PROJECT_FOLDER=${PROJECT_FOLDER:-$1}
LOG_FILE=$PROJECT_FOLDER"/logs/voxel_mask.log"

# Using 0.5 as default threshold
MASK_THRES=${3:-0.5}

# Using 10 masking images by default
NUM_MASKS=${2:-10}

echo "Mask threshold "$MASK_THRES
echo "Number of masks "$NUM_MASKS


# ----- MASK --------
cd $PX_RUNPATH

echo
echo "Mask voxels"
time_log_file=$PROJECT_FOLDER/logs/time_mask_voxels.log
total_log=$PROJECT_FOLDER/logs/time_total.log

tic.sh $time_log_file
# TODO: Change location of file?
time python voxel_mask.py --checkpoint $PROJECT_FOLDER/ckpt/std/ckpt.npz --data_dir $PROJECT_FOLDER --mask_thres $MASK_THRES --num_masks $NUM_MASKS --source images 2>&1 | tee $LOG_FILE 
EXIT_STATUS=${PIPESTATUS[0]:-1}

toc.sh $time_log_file $total_log


if [ $EXIT_STATUS -eq 0 ]
then
    echo "Voxel masking was was successful"
else
    echo "Voxel masking failed"
fi

exit $EXIT_STATUS