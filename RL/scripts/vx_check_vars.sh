
################################
#   CHECK ENVIRONMENT VARIABLES      
################################

# Check the necessary environment variables for the pipeline

# USAGE
# check_vars.sh 
# 
#
# Probably want to run from host like
#
# docker exec -it <DOCKER ID> --env-file <ENV_FILE> bash -c "/workspace/aseeo-research/RLResearch/workflow/bash_scripts/check_vars.sh"
#
# If the vars were set up correctly, it probably is enough to run like
#
# docker exec -it <DOCKER ID> --env-file <ENV_FILE> bash -c "check_vars.sh"
#
# Note that <ENV_FILE> may or not be present depending on how you decided to create the environment variables

echo 
echo "-- ENVIRONMENT VARIABLES --"
echo
echo "RL_RUNPATH"
echo
echo $RL_RUNPATH
echo
echo "PX_RUNPATH"
echo
echo $PX_RUNPATH
echo
echo "PX_SCRIPTS"
echo
echo $PX_SCRIPTS
echo
echo "PATH"
echo
echo $PATH
echo
echo "PROJECT FOLDER"
echo 
echo $PROJECT_FOLDER
echo
echo "DOWNSCALING"
echo 
echo $DOWNSCALING
echo 




