#########################
#  UPDATE REPOSITORY    # 
#########################

# Will update the repositories for 3D Reconstruction

# USAGE
# update_repository.sh <RL_BRANCH> <PX_BRANCH>

RL_BRANCH=$1
PX_BRANCH=$2

echo "Will try to pull $RL_BRANCH from asseo-research"
echo "Will try to pull $PX_BRANCH from RL svox2"

cd $RL_RUNPATH
git pull origin $RL_BRANCH

cd $PX_RUNPATH
git pull origin $PX_BRANCH


