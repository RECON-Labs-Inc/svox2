import sys
from pathlib import Path
from datetime import datetime
import argparse
import json
import pickle
from PIL import Image
from rembg.bg import remove as remove_backround


import torch
from torchvision.utils import save_image
import torchvision

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle

import open3d as o3d
import cv2 as cv


from svox2 import *
from pyvox.models import Vox, Color
from pyvox.writer import VoxWriter
from pyvox.parser import VoxParser

from importlib import reload as reload

reload(svox2)
from svox2 import *

#TODO> modify this:
sys.path.append("/workspace/svox2/opt")
from util.dataset import datasets
from util import config_util

# Our nice tools
sys.path.append("/workspace/aseeo-research")
import RLResearch.utils.depth_utils as du
import RLResearch.utils.gen_utils as gu


# # ## ARGPARSE
# parser = argparse.ArgumentParser()
# parser.add_argument("--checkpoint", type=str,default=None, help=".npz checkpoint file")
# parser.add_argument("--data_dir", type=str,default=None, help="Project folder")
# parser.add_argument("--grid_dim", type=int, default = 256, help = "grid_dimension")
# parser.add_argument("--vox_file", type=str, default = None, help = "Voxel file to be masked")
# parser.add_argument("--source", type=str, default = "images_undistorted", help = "subfolder where images are located")
# args = parser.parse_args()
# checkpoint_path = Path(args.checkpoint)
# data_dir = args.data_dir
# grid_dim = args.grid_dim
# vox_file = Path(data_dir)/"result"/"voxel"/"vox.vox"
# vox_file = str(vox_file.resolve())
# source = args.source

#----
print("***** Running the no argument version *****")
data_dir = "/workspace/datasets/cctv"
exp_name = "std"
checkpoint_path = Path(data_dir)/"ckpt"/exp_name/"ckpt.npz"
vox_file = "/workspace/datasets/cctv/result/voxel/vox.vox"
grid_dim = 256 # My sampling
orig_grid_dim = 640 # Actual grid size of grid from args.json
source = "images"
# #-----

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"

# Load arguments from json
# json_config_path = Path(data_dir)/"ckpt"/exp_name/"args.json"


# ## CHANGE: to test_train
# dataset = datasets["nsvf"](
#             data_dir,
#             split="test_train",
#             device=device,
#             factor=1,
#             n_images=None)

# grid = SparseGrid.load(str(checkpoint_path.resolve()))
# config_util.setup_render_opts(grid.opt, args)
# print('Render options', grid.opt)

# ---- Load vox file

m = VoxParser(vox_file).parse()
voxel_data = m.to_dense()
orig_voxel_data = voxel_data
original_palette = m.palette

### --- LOAD points from npy file
voxel_npy_path = Path(data_dir)/"project_files"/"voxel_points.npy"
occupied_points_centered = np.load(str(voxel_npy_path.resolve()))
occupied_points_centered = torch.tensor(occupied_points_centered, device = device, dtype=torch.float64)



a = 5
### ======
masked_indices = ( (scores - 1.0) < mask_thres_int).nonzero()
mask_result[masked_indices] = 0

voxel_data_flat[voxel_indices] = mask_result

vox = Vox.from_dense(voxel_data_flat.reshape(voxel_data.shape))
vox.palette = original_palette

output_path = Path(data_dir)/"result"/"vox_masked.vox"
VoxWriter(str(output_path.resolve()), vox).write()
print('The Vox file created in ', str(output_path))

a = 5