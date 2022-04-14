cd $PX_RUNPATH/..

echo "CUDA_HOME"
echo $CUDA_HOME
echo "CUB_HOME"
echo $CUB_HOME

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
echo $CUDA_HOME


conda env update --file environment.yml --prune
nvcc --version
nvidia-smi


conda list


conda install cudatoolkit=11.1
conda install -c bottler nvidiacub
conda list
python3 -m pip install opencv-python
python3 -m pip install open3d
#  python3 -m pip install torch torchvision tqdm
python3 -m pip install tqdm
python3 -m pip install scikit-image
python3 -m pip install ffmpeg-python hydra addict omegaconf imageio plyfile tensorboard scipy rembg py-vox-io

cd /workspace/svox2
pip install .