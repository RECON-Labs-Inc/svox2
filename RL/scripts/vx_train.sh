#########################
#           TRAIN       #
#########################

# Train using neurecon.

# USAGE
# train.sh -f <PROJECT_FOLDER> -l <[OPTIONAL] LOG_FILE> -n <[OPTIONAL] NUMBER OF CONFIG FILE> -c <[OPTIONAL] PATH OF CONFIG FILE>

# NOTE:
# With the -n option its possible to choose a config file named like config_0.json, config_1.json, 
# etc, placed in /workspace/svox2/opt/configs ($PX_RUNPATH/../opt/configs) Otherwise, the -c option will load a custom config json file from
# a specific path. -c option will override -l.


optstring="f:l:n:c:u"

while getopts ${optstring} arg; do
  case "${arg}" in
    f)
        # Current behaviour is that pre-existing env var overrides the project folder. This could change later TODO
        
        ## Check if env vars exist
        if ! [ -z $PROJECT_FOLDER ]
        then
            echo
            echo "PROJECT_FOLDER env var exists, using it."
            echo $PROJECT_FOLDER
        else
            PROJECT_FOLDER=${OPTARG} # if it doesn't exist, it takes the argument
        fi
        
        ;;
    l)  
        LOG_FILE=${OPTARG}
        echo
        echo "log file is $LOG_FILE" 
        ;;
    n) 
        N_CONFIG=${OPTARG}
        echo
        echo "Requested config number "$N_CONFIG ;;
    
    c) 
        CUSTOM_CONFIG=${OPTARG}
        echo
        echo "Request to use custom config file "$CUSTOM_CONFIG ;;
    u)
        CUDA_DEVICE=${OPTARG}
        echo
        echo "Using cuda device "$CUDA_DEVICE;;
    ?)
      echo "Invalid option: -${OPTARG}."
      echo
    #   usage
      ;;
  esac
done
# Cuda device 0 if unset
CUDA_DEVICE=${CUDA_DEVICE:-0}

# Using a set of numbered config files config_N.json that should be under $PX_RUNPATH/../opt/configs
N_CONFIG=${N_CONFIG:-0}

OPT_DIR=$PX_RUNPATH/../opt/
cd $OPT_DIR

CONFIG=$OPT_DIR"/configs/config_"$N_CONFIG".json"

# Use custom config file if requested. Else use config_N.json
CONFIG=${CUSTOM_CONFIG:-$CONFIG}

if test -f "$CONFIG"; then
    echo 
    echo "using config file "$CONFIG
else
    echo "Requested file $CONFIG does not exist. Aborting"
    exit 1
fi

# If the LOG_FILE args exists, pipe
LOG_FILE=${LOG_FILE:-$PROJECT_FOLDER"/logs/train.log"}


echo
echo "Train"
cd $OPT_DIR
experiment=std

CKPT_DIR=$PROJECT_FOLDER/ckpt/$experiment
mkdir -p $CKPT_DIR
echo CKPT $CKPT_DIR
echo "LOGFILE "$LOG_FILE
time_log_file=$PROJECT_FOLDER/logs/time_train.log
total_log=$PROJECT_FOLDER/logs/time_total.log

tic.sh $time_log_file

time CUDA_VISIBLE_DEVICES=$CUDA_DEVICE nohup python -u opt.py -t $CKPT_DIR $PROJECT_FOLDER -c $CONFIG --log_depth_map > $LOG_FILE
EXIT_STATUS=${PIPESTATUS[0]:-1}

toc.sh $time_log_file $total_log


if [ $EXIT_STATUS -eq 0 ]
then
    echo "Training was successful"
else
    echo "Training failed"
fi


exit $EXIT_STATUS