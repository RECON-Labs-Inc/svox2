##################
#  PUSH VOXEL    #
##################
# DESCRIPTION
# Will fill a voxel by pushing inwards an unsculpted block

# USAGE:
# voxel_push.sh <PROJECT_FOLDER> <KEEP_FLOOR>
# voxel_push.sh /workspace/datasets/cactus keep_floor  --> use like this

# DETAILED DESCRIPTION
# If you use the keep floor option, the voxels will not be pushed upwards.



## Check if env vars exist
if ! [ -z $PROJECT_FOLDER ]
then
    echo
    echo "PROJECT_FOLDER env var exists, using it."
    echo $PROJECT_FOLDER
    echo
fi

PROJECT_FOLDER=${PROJECT_FOLDER:-$1}
LOG_FILE=$PROJECT_FOLDER"/logs/voxel_push.log"

# ----- PUSH --------
cd $PX_RUNPATH

echo
echo "Push voxels"
time_log_file=$PROJECT_FOLDER/logs/time_push_voxels.log
total_log=$PROJECT_FOLDER/logs/time_total.log

tic.sh $time_log_file

if [ "$2" = "keep_floor" ]
then
    echo "Keeping floor"
    time python ../voxel_push.py --vox_file $PROJECT_FOLDER/result/vox_masked.vox --ref_vox_file $PROJECT_FOLDER/result/vox_masked.vox --data_dir $PROJECT_FOLDER --use_block --keep_floor 2>&1 | tee $LOG_FILE 
else
    time python ../voxel_push.py --vox_file $PROJECT_FOLDER/result/vox_masked.vox --ref_vox_file $PROJECT_FOLDER/result/vox_masked.vox --data_dir $PROJECT_FOLDER --use_block 2>&1 | tee $LOG_FILE 
fi

EXIT_STATUS=${PIPESTATUS[0]:-1}
toc.sh $time_log_file $total_log


if [ $EXIT_STATUS -eq 0 ]
then
    echo "Voxel pushing was successful"
else
    echo "Voxel pushing failed"
fi

exit $EXIT_STATUS