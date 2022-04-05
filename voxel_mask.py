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

import RL.utils


# ## ARGPARSE
parser = argparse.ArgumentParser()
parser.add_argument("--vox_file", type = str, default=None,  help="Vox file to be masked")
parser.add_argument("--checkpoint", type=str,default=None, help=".npz checkpoint file")
parser.add_argument("--data_dir", type=str,default=None, help="Project folder")
parser.add_argument("--num_masks", type=int, default = 20, help = "number of masks used to mask/sculpt the object")
# parser.add_argument("--vox_file", type=str, default = None, help = "Voxel file to be masked")
parser.add_argument("--source", type=str, default = "images_undistorted", help = "subfolder where images are located")
parser.add_argument("--use_block", action="store_true" ,  help = "Use block")
parser.add_argument("--mask_thres", type=float , default=0.5,  help = "Values less than mask_thres will be masked")
parser.add_argument("--debug_folder", type=str,default=None, help="debug folder for saving stuff")
args = parser.parse_args()
checkpoint_path = Path(args.checkpoint)
data_dir = args.data_dir

if args.vox_file is None:
        vox_file = Path(data_dir)/"result"/"voxel"/"vox.vox"
        vox_file = str(vox_file.resolve())
else:
        vox_file = args.vox_file

print("INPUT VOXEL", vox_file )
source = args.source
use_block = args.use_block
num_masks = args.num_masks
mask_thres = args.mask_thres
debug_folder = args.debug_folder

print("MASK THRES" , mask_thres)

# #----
# print("Running the no argument version")
# data_dir = "/workspace/datasets/_p_shoe_200_single_pose_dwn_8"
# exp_name = "std"
# checkpoint_path = Path(data_dir)/"ckpt"/exp_name/"ckpt.npz"
# vox_file = "/workspace/datasets/_p_shoe_200_single_pose_dwn_8/result/voxel/vox.vox"
# grid_dim = 256 # My sampling
# orig_grid_dim = 640 # Actual grid size of grid from args.json
# source = "images"
# # #-----

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
orig_voxel_data = voxel_data
original_palette = m.palette

if use_block:
        print("SCULPTING BLOCK")
        # Make bounding block to be sculpted
        indices = np.array(voxel_data.nonzero())
        mins = np.amin(indices, axis=1)
        maxs = np.amax(indices, axis=1)
        print(mins)
        print(maxs)

        block = np.zeros_like(voxel_data)
        block[mins[0]:maxs[0],mins[1]:maxs[1],mins[2]:maxs[2] ] = 1
        block_flat = block.flatten()
        voxel_data  = block
        orig_voxel_data = block

# ----- Load camera data from json file: recenter matrix, poses, scale_factor
cam_filename = Path(data_dir)/"project_files"/"cam_data.json"
camera_dict = json.load(open(str(cam_filename.resolve())))
nsvf_recenter_matrix = torch.tensor(camera_dict["nsvf_recenter_matrix"], device = device)
inv_recenter_matrix = torch.linalg.inv(nsvf_recenter_matrix)
scale_factor = camera_dict["scale_factor"]
poses = np.array(camera_dict["ms_poses"])

# ---  LOAD images from dataset
# mask_path = Path(data_dir)/"source"/"mask"
# masks = []
# if mask_path.exists() is False:
#         # raise FileNotFoundError(mask_path, "does not exist")
#         # masks_not_computed = True
#         print("No masks present. Will compute on the fly")
# else:
#         for mask_image in sorted(mask_path.iterdir()):
#                 print(mask_image)
#                 pili = Image.open(str(mask_image.resolve()))
#                 masks.append(np.asarray(pili))

image_path = Path(data_dir)/"source"/source

print("Number of images ", poses.shape[0])
print("num_masks", num_masks)

subset = range(0, poses.shape[0], int(poses.shape[0]/num_masks))

if image_path.exists() is False:
        raise FileNotFoundError(image_path, "does not exist")
else:
        im_glob = Path(image_path).glob('*.*')
        image_path_list = sorted([x for x in im_glob if x.is_file()])

        # for mask_image in sorted(image_path.iterdir()):
        #         print(mask_image)
        #         pili = Image.open(str(mask_image.resolve()))
                
        #         # downsampling_factor = 8
        #         # pili = Image.resize()
        #         mask = remove_backround(pili)
        #         mask_subset.append(np.asarray(mask))

subset = range(0, poses.shape[0], int(poses.shape[0]/num_masks))
source_image_filenames = [image_path_list[x] for x in subset]

mask_subset = RL.utils.make_masks(source_image_filenames, save_folder = Path(data_dir)/"source"/"mask")

c2w_subset = []
for ind in subset:
        c2w = poses[ind, ...]
        c2w_subset.append(c2w)

### --- LOAD points from npy file
if use_block:
        voxel_npy_path = Path(data_dir)/"project_files"/"grid_points.npy"
        flat_world_points = np.load(str(voxel_npy_path.resolve()))
        occupied_points_centered = flat_world_points[voxel_data.flatten().nonzero()]
        occupied_points_centered = torch.tensor(occupied_points_centered, device = device, dtype=torch.float64)
else:
        voxel_npy_path = Path(data_dir)/"project_files"/"voxel_points.npy"
        occupied_points_centered = np.load(str(voxel_npy_path.resolve()))
        occupied_points_centered = torch.tensor(occupied_points_centered, device = device, dtype=torch.float64)

# Rescale position
occupied_points_centered *= (1.0/scale_factor)
ones = torch.ones((1, occupied_points_centered.shape[0]), device=device)
occupied_points_centered = torch.cat((occupied_points_centered.T, ones))

# Convert to world
occupied_points_world = inv_recenter_matrix.double() @ occupied_points_centered


num_voxels = occupied_points_world.shape[1]
print(num_voxels)

# --- Make camera matrix
height = camera_dict["calibration"]["h"]
width = camera_dict["calibration"]["w"]

fx = camera_dict["calibration"]["f"]
fy = fx
cx = camera_dict["calibration"]["w"]/2
cy = camera_dict["calibration"]["h"]/2

cam_matrix = torch.tensor([ [fx, 0, cx],
                        [0, fy, cy],
                        [0,  0, 1 ]], device = device)

debug_path = Path(data_dir)/"result"/"voxel_block" if use_block else Path(data_dir)/"result"/"voxel"
debug_path.mkdir(exist_ok=True, parents=True)
scores, mask_result = RL.utils.mask_points(voxel_data, 
        occupied_points_world, 
        mask_subset, 
        c2w_subset,
        cam_matrix,
        device = device,
        debug_dir= None if use_block else str( debug_path.resolve() ) # Blocks are too big
        )
# Make a palette based on the score
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

viridis = cm.get_cmap('viridis', 256)
print(viridis)
vox_pal = []
for i in range(0, 255):
        col = np.array(viridis(i / 255.0))
        col *= 255
        col = col.astype(np.uint8)
        vox_pal.append(Color(col[0], col[1], col[2], col[3]))

voxel_data_flat = voxel_data.flatten()
voxel_indices = voxel_data_flat.nonzero()
voxel_indices = voxel_indices[0]
voxel_data_flat[voxel_indices] = mask_result.cpu().numpy()

# scores *= 255
scores = scores.cpu().numpy().astype(np.uint8)
scores = scores + 1

color_labels = np.zeros_like(voxel_data_flat, dtype=np.uint8)
color_labels[voxel_indices] = scores 

# Rewrite voxel data
color_labels = color_labels.reshape(voxel_data.shape)
vox = Vox.from_dense(color_labels)
vox.palette = vox_pal
# p_projected = p_projected

# output_path = "/workspace/data/vox_mask_summary.vox"
output_path = Path(data_dir)/"result"/"voxel"/"vox_mask_debug.vox"
VoxWriter(str(output_path.resolve()), vox).write()
print('The Vox file created in ', str(output_path))

if debug_folder is not None:
        VoxWriter( str(Path(debug_folder)/"vox_mask_debug.vox"), vox).write()
        



# Now filter based on threshold

# mask_thres = 0.5
print("mask_thres", mask_thres)
mask_thres_int = mask_thres * 255
mask_thres_int = int(mask_thres_int)

# masked_values = torch.
voxel_data_flat = orig_voxel_data.flatten()
voxel_indices = voxel_data_flat.nonzero()
mask_result = voxel_data_flat[voxel_indices] # Init

masked_indices = ( (scores - 1.0) < mask_thres_int).nonzero() #Don't remember why this -1.0 was necessary. Sorry. 
mask_result[masked_indices] = 0

voxel_data_flat[voxel_indices] = mask_result

vox = Vox.from_dense(voxel_data_flat.reshape(voxel_data.shape))
vox.palette = original_palette

if use_block:
        output_path = Path(data_dir)/"result"/"vox_masked_block.vox"
else:
        output_path = Path(data_dir)/"result"/"vox_masked.vox"



VoxWriter(str(output_path.resolve()), vox).write()
print('The Vox file created in ', str(output_path))

if debug_folder is not None:
        Path(debug_folder).mkdir(exist_ok=True, parents=True)
        VoxWriter( str(Path(debug_folder)/output_path.name), vox).write()

