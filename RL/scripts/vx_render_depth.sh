##################
#  RENDER-DEPTH  #
##################

# USAGE:
# vx_render_depth.sh <PROJECT_FOLDER> <CUDA_DEVICE>
#

# DESCRIPTION
# Renders the depth from a trained Plenoxel 


## Check if env vars exist
if ! [ -z $PROJECT_FOLDER ]
then
    echo
    echo "PROJECT_FOLDER env var exists, using it."
    echo $PROJECT_FOLDER
    echo
fi

PROJECT_FOLDER=${PROJECT_FOLDER:-$1} # Useless???
echo "Project folder "$PROJECT_FOLDER
echo "Cuda device "$CUDA_DEVICE 
 
echo "$PROJECT_FOLDER" "$GRID_DIM" "$EULER_ANGLES"

if [ -z "$CUDA_DEVICE" ]
then
    echo "No cuda device specified, using first available."
    CUDA_VISIBLE_DEVICES=0
else
    # Using 10 and 0.5 for masking as default values
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
fi

cd $PX_RUNPATH/../opt
python render_depth.py $PROJECT_FOLDER/ckpt/std/ckpt.npz $PROJECT_FOLDER

if ! [ $? -eq 0 ]
then
    echo "Depth rendering failed"
    exit 1
else
    echo "Depth rendering succesful"
    exit 0
fi
