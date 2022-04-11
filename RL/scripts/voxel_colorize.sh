######################
#  COLORIZE VOXEL    #
######################
# DESCRIPTION
# Will colorize a voxel using a palette or a classifier, or both.

# USAGE:
# voxel_colorize.sh -f <PROJECT_FOLDER> -m <METHOD[classifier,palette]> -c <N_CLUSTERS> -p <PALETIZE [true, false]>

# DETAILED DESCRIPTION
# Colorizes a voxel using by computing the minimum distance to a palette, or by using kmeans clustering.
# If METHOD is classifier, and you also input a palette filename PAL_FILENAME, the algorithm will
# use a kmeans classifier first, and then min the colors to a palette.



## Check if env vars exist
if ! [ -z $PROJECT_FOLDER ]
then
    echo
    echo "PROJECT_FOLDER env var exists, using it."
    echo $PROJECT_FOLDER
    echo
fi

# Parse args
optstring="f:m:c:p:"

while getopts ${optstring} arg; do
  case "${arg}" in
    f)
        FOLDER=${OPTARG}
        PROJECT_FOLDER=${PROJECT_FOLDER:-$FOLDER}
        ;;
    m)  
        METHOD=${OPTARG}
        echo
        echo "METHOD is $METHOD" 
        ;;
    c) 
        N_CLUSTERS=${OPTARG}
        echo
        echo "N_CLUSTERS "$N_CLUSTERS ;;
    p) 
        PALETIZE=true
        echo
        echo "PAL_FILENAME "$PALETIZE ;;
    ?)
      echo "Invalid option: -${OPTARG}."
      echo
    #   usage
      ;;
  esac
done



echo "Method "$METHOD

LOG_FILE=$PROJECT_FOLDER"/logs/voxel_colorize.log"



# ----- PUSH --------
cd $PX_RUNPATH

echo
echo "Colorize voxels"
time_log_file=$PROJECT_FOLDER/logs/time_colorize.log
total_log=$PROJECT_FOLDER/logs/time_total.log

tic.sh $time_log_file

if [ "$METHOD" = "classifier" ]
then
    echo "Classifierrrr"
    if [ "$PALETIZE" = "true" ] # If there is no palette
    then
        echo "Paletize"
        time python voxel_colorize.py --vox_file $PROJECT_FOLDER/result/voxel/vox_pushed.vox --data_dir $PROJECT_FOLDER \
                    --color_mode classifier --paletize --palette_filename $PX_RUNPATH/../vox_palette.png --n_clusters $N_CLUSTERS 2>&1 | tee $LOG_FILE 
    else
        echo "Dont paletize"
        time python voxel_colorize.py --vox_file $PROJECT_FOLDER/result/voxel/vox_pushed.vox --data_dir $PROJECT_FOLDER \
                    --color_mode classifier 2>&1 | tee $LOG_FILE 
    fi
   
elif [ "$METHOD" = "palette" ]
then
    echo "Palette"
    time python voxel_colorize.py --vox_file $PROJECT_FOLDER/result/voxel/vox_pushed.vox --data_dir $PROJECT_FOLDER \
                    --color_mode palette --palette_filename $PX_RUNPATH/../vox_palette.png 2>&1 | tee $LOG_FILE 
else
    echo "No colorizing method or wrong one. Use -m option with either \"classifier\" or \"palette\". See docs."
    exit 1
fi

EXIT_STATUS=${PIPESTATUS[0]:-1}
toc.sh $time_log_file $total_log


if [ $EXIT_STATUS -eq 0 ]
then
    echo "Voxel colorizing was successful"
else
    echo "Voxel colorizing failed"
fi

exit $EXIT_STATUS