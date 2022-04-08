#########################
#  AI-PIPELINE-RUNNER   #
#########################

# Will run the whole AI pipeline from preparing data, pre-processing, training, to making  model

# USAGE
# ai_pipeline_runner.sh <PROJECT_FOLDER> <VIDEO_FILE> <[OPTIONAL] DOWNSCALING>


## -----     CHECK FOR ENV VARS -----
if ! [ -z $PROJECT_FOLDER ]
then
    echo
    echo "PROJECT_FOLDER env var exists, using it."
    echo $PROJECT_FOLDER
    echo
fi

PROJECT_FOLDER=${PROJECT_FOLDER:-$1}


if ! [ -z $VIDEO_FILE ]
then
    echo
    echo "VIDEO_FILE env var exists, using it."
    echo $VIDEO_FILE
    echo
fi

# NOT using this yet
VIDEO_FILE=${VIDEO_FILE:-$2}

if ! [ -z $DOWNSCALING ]
then
    echo
    echo "DOWNSCALING env var exists, using it."
    echo $DOWNSCALING
    echo
fi

DOWNSCALING=${DOWNSCALING:-$3}

DEFAULT_DOWNSCALING=4
if [ -z "$DOWNSCALING" ]
    then
    echo "No downscaling supplied, using default"
    DOWNSCALING=$DEFAULT_DOWNSCALING
fi

## ----- ACTIVATE METASHAPE --------

activate_ms_license.sh  -a check -f $RL_RUNPATH

if ! [ $? -eq 0 ] # If not activated
then
    echo "No metashape license activated. Trying to activate..."
    if [ -z "$MS_LICENSE" ]
    then
        echo "No metashape license key. set the MS_LICENSE environment variable. Aborting"
        exit 1
    else
        activate_ms_license.sh  -a activate -f $RL_RUNPATH -l $MS_LICENSE
    fi
fi

## -----    RUN THE PIPELINE   -----

echo "Downscaling "$DOWNSCALING

make_dirs.sh $PROJECT_FOLDER

if [ $? -eq 0 ]
then
    # Check if file is link
    if [[ $VIDEO_FILE == *"http"* ]]
    then
        VIDEO_URL=$VIDEO_FILE
        download_video.sh $VIDEO_URL $PROJECT_FOLDER"/source/" $PROJECT_FOLDER"/logs/pipeline/download_video.log"
        # EXIT_STATUS=${PIPESTATUS[0]:-1}

        if [ $? -eq 0 ]
        then
            VIDEO_FILE=$PROJECT_FOLDER"/source/source.video" 
            echo "Video downloaded correctly to "$VIDEO_FILE

        else
            echo "---> Video downloading error. Aborting."
            exit 1
        fi

    fi
else
    echo "---> Make dir error. Aborting."
    exit 1
fi



if [ $? -eq 0 ]
then
    extract_frames.sh $VIDEO_FILE 130 $PROJECT_FOLDER $PROJECT_FOLDER"/logs/pipeline/extract_frames.log"
else
    echo "---> Make dir or download error. Aborting."
    exit 1
fi

if [ $? -eq 0 ]
then
    pre_process_poses.sh $PROJECT_FOLDER $PROJECT_FOLDER"/logs/pipeline/pre_process_poses.log"
else
    echo "---> Extract frames error. Aborting."
    exit 1
fi

if [ $? -eq 0 ]
then
    train.sh -f $PROJECT_FOLDER -l $PROJECT_FOLDER"/logs/pipeline/train.log"
else
    echo "---> Preprocessing error. Aborting."
    exit 1
fi


if [ $? -eq 0 ]
then
    # Post-process will log by default
    post_process.sh $PROJECT_FOLDER $DOWNSCALING 
else
    echo "Training error. Aborting."
    exit 1
fi

EXIT_STATUS=$?

if [ $EXIT_STATUS -eq 0 ]
then
    echo "Pipeline result was succesful"
else
    echo "Pipeline failed"
fi


activate_ms_license.sh  -a deactivate -f $RL_RUNPATH -l $MS_LICENSE


exit $EXIT_STATUS


