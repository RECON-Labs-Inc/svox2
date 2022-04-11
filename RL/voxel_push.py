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

sys.path.append("/workspace/svox2")
import RL.utils


# ## ARGPARSE
parser = argparse.ArgumentParser()
parser.add_argument("--vox_file", type = str, default=None,  help="Vox file to be masked")
parser.add_argument("--ref_vox_file", type = str, default=None,  help="Vox file to be masked")
# parser.add_argument("--checkpoint", type=str,default=None, help=".npz checkpoint file")
parser.add_argument("--data_dir", type=str,default=None, help="Project folder")
# parser.add_argument("--num_masks", type=int, default = 20, help = "grid_dimension")
# parser.add_argument("--source", type=str, default = "images_undistorted", help = "subfolder where images are located")
parser.add_argument("--use_block", action="store_true" ,  help = "Use block")
# parser.add_argument("--mask_thres", type=float ,  help = "Values less than mask_thres will be masked")
parser.add_argument("--debug_folder", type=str,default=None, help="debug folder for saving stuff")
parser.add_argument("--keep_floor", action="store_true" ,  help = "Don't remove voxels upwards.")

args = parser.parse_args()
# checkpoint_path = Path(args.checkpoint)
data_dir = args.data_dir

if args.vox_file is None:
        vox_file = Path(data_dir)/"result"/"voxel"/"vox.vox"
        vox_file = str(vox_file.resolve())
else:
        vox_file = args.vox_file

use_block = args.use_block
ref_vox_file = args.ref_vox_file
debug_folder = args.debug_folder


print("INPUT VOXEL", vox_file )
print("REF VOXEL", ref_vox_file)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# device = "cpu"

# Load arguments from json
# json_config_path = Path(data_dir)/"ckpt"/exp_name/"args.json"

# # Doesn't seem necessary to actually load the dataset.
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
original_palette = m.palette

# Get bounds
print(voxel_data.shape)
indices = np.array(voxel_data.nonzero())
mins = np.amin(indices, axis=1)
maxs = np.amax(indices, axis=1)
print(mins)
print(maxs)

if use_block:
    print("SCULPTING BLOCK")
    block = np.zeros_like(voxel_data)
    block[mins[0]:maxs[0],mins[1]:maxs[1],mins[2]:maxs[2] ] = 1
    voxel_data  = block

# ---- Load reference file
m = VoxParser(ref_vox_file).parse()
ref_voxel_data = m.to_dense()

# X push
for y in range (mins[1], maxs[1]):
    for z in range (mins[2], maxs[2]):
        
        #Pushing from boundary (mins[0] to maxs[0]) until we find an occupied voxel
        k = 0
        while (mins[0] + k) < maxs[0] and  ref_voxel_data[mins[0] + k, y, z]==0 :
            voxel_data[mins[0] + k, y, z] = ref_voxel_data[mins[0] + k, y, z]
            k += 1
        
        #Push backwards, from (maxs[0] to mins[0])
        l = 0
        while (maxs[0] - l) >= mins[0] and  ref_voxel_data[maxs[0] - l, y, z]==0 :
            voxel_data[maxs[0] - l, y, z] = ref_voxel_data[maxs[0] - l, y, z]
            l += 1
        

# Y push
for x in range (mins[0], maxs[0]):
    for z in range (mins[2], maxs[2]):
        k = 0
        while (mins[1] + k) < maxs[1] and  ref_voxel_data[x, mins[1] + k, z]==0 :
            voxel_data[x, mins[1] + k, z] = ref_voxel_data[x, mins[1] + k, z]
            k += 1
        
        if args.keep_floor is False:
            # print("pushiiiiiiing")
            #Push backwards, from (maxs[1] to mins[1])
            l = 0
            while (maxs[1] - l) >= mins[0] and  ref_voxel_data[x, maxs[1] - l, z]==0 :
                voxel_data[x, maxs[1] - l, z] = ref_voxel_data[x, maxs[1] - l, z]
                l += 1
        
# z push
for x in range (mins[0], maxs[0]):
    for y in range (mins[1], maxs[1]):
        k = 0
        while (mins[1] + k) < maxs[1] and  ref_voxel_data[x,y, mins[1] + k]==0 :
            voxel_data[x,y, mins[1] + k] = ref_voxel_data[x,y, mins[1] + k]
            k += 1
         
        #Push backwards, from (maxs[2] to mins[2])
        l = 0
        while (maxs[2] - l) >= mins[0] and  ref_voxel_data[x, y, maxs[2] - l]==0 :
            voxel_data[x, y, maxs[2] - l] = ref_voxel_data[x, y, maxs[2] - l]
            l += 1

# Rewrite voxel data
# color_labels = voxel_data
vox = Vox.from_dense(voxel_data)
vox.palette = original_palette
# p_projected = p_projected

# output_path = "/workspace/data/vox_mask_summary.vox"
output_path = Path(data_dir)/"result"/"voxel"/"vox_pushed.vox"
VoxWriter(str(output_path.resolve()), vox).write()
print('The Vox file created in ', str(output_path))

if debug_folder is not None:
        Path(debug_folder).mkdir(exist_ok=True, parents=True)
        VoxWriter( str(Path(debug_folder)/output_path.name), vox).write()



