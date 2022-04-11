##################
#  POST-PROCESS  #
##################

# USAGE:
# vx_post_process.sh <PROJECT_FOLDER> <GRID_DIM> <EULER_ANGLES> <COL_METHOD> 
#

# Chains the following stages in one: 
# 1. VOXEL MAKE
# 2. VOXEL MASK
# 3. VOXEL FILL
# 4. VOXEL COLORIZE

# NOTES:
# - This script will log by default to PROJECT_FOLDER/logs/ There will be an error if the folder does not exist

## Check if env vars exist
if ! [ -z $PROJECT_FOLDER ]
then
    echo
    echo "PROJECT_FOLDER env var exists, using it."
    echo $PROJECT_FOLDER
    echo
fi

PROJECT_FOLDER=${PROJECT_FOLDER:-$1} # Useless???
GRID_DIM=$2
EULER_ANGLES=$3
COL_METHOD=$4

echo "Project folder "$PROJECT_FOLDER
echo "Grid dim "$GRID_DIM
echo "Euler angles "$EULER_ANGLES
echo "Coloroxing method "$COL_METHOD

echo "$PROJECT_FOLDER" "$GRID_DIM" "$EULER_ANGLES"


voxel_make.sh "$PROJECT_FOLDER" "$GRID_DIM" "$EULER_ANGLES"

if ! [ "$?" -eq 0 ]
then
    echo "Post processing failed"
    exit 1
fi

voxel_mask.sh $PROJECT_FOLDER

if ! [ $? -eq 0 ]
then
    echo "Masking failed"
    exit 1
fi

voxel_push.sh $PROJECT_FOLDER

if ! [ $? -eq 0 ]
then
    echo "Pushing failed"
    exit 1
fi


voxel_colorize.sh -f $PROJECT_FOLDER -m $COL_METHOD -p true -c 10

if ! [ $? -eq 0 ]
then
    echo "Colorizing failed"
    exit 1
else
    echo "Colorizing succesful"
fi
