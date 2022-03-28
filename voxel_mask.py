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
parser.add_argument("--checkpoint", type=str,default=None, help=".npz checkpoint file")
parser.add_argument("--data_dir", type=str,default=None, help="Project folder")
parser.add_argument("--grid_dim", type=int, default = 256, help = "grid_dimension")
# parser.add_argument("--vox_file", type=str, default = None, help = "Voxel file to be masked")
parser.add_argument("--source", type=str, default = "images_undistorted", help = "subfolder where images are located")
args = parser.parse_args()
checkpoint_path = Path(args.checkpoint)
data_dir = args.data_dir
grid_dim = args.grid_dim
vox_file = Path(data_dir)/"result"/"voxel"/"vox.vox"
vox_file = str(vox_file.resolve())
source = args.source

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

num_masks = 2
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

dilate_image = True
dilatation_size = 25
downsample = True


for ind in subset:
        #Compute mask here based on file list.
        # print(str(ind), image_path_list[ind])
        # pili = Image.open(str(image_path_list[ind].resolve()))
        # max_dimension = max((pili.width, pili.height))
        # if downsample is True:
        #         if max_dimension >= 3840:
        #                 mask_downsample = 4
        #         elif max_dimension >= 1920:
        #                 mask_downsample = 2
        #         else:
        #                 mask_downsample = 1
        
        #         pili = pili.resize( ( int(pili.width/mask_downsample), int(pili.height/mask_downsample) ))
        #         mask = remove_backround(pili)
        #         mask = mask.resize( (mask.width * mask_downsample, mask.height * mask_downsample) )
        # else:
        #         mask = remove_backround(pili)

        # mask = mask.split()[-1]
        # mask = np.asarray(mask)
        # # mask_subset.append(np.asarray(mask))

        # if dilate_image:
        #         mask = gu.dilate_image( mask, dilatation_size= dilatation_size)
        #         mask_subset.append(mask)
        # else:       
        #         mask_subset.append(masks[ind])

        c2w = poses[ind, ...]
        c2w_subset.append(c2w)

### --- LOAD points from npy file
voxel_npy_path = Path(data_dir)/"project_files"/"voxel_points.npy"
occupied_points_centered = np.load(str(voxel_npy_path.resolve()))
occupied_points_centered_orig = occupied_points_centered
occupied_points_centered = torch.tensor(occupied_points_centered, device = device, dtype=torch.float64)

# Rescale position
occupied_points_centered *= (1.0/scale_factor)
ones = torch.ones((1, occupied_points_centered.shape[0]), device=device)
occupied_points_centered = torch.cat((occupied_points_centered.T, ones))

# Convert to world
occupied_points_world = inv_recenter_matrix.double() @ occupied_points_centered


num_voxels = occupied_points_world.shape[1]
print(num_voxels)

fill = True

if fill is True:
        grid_path = Path(data_dir)/"project_files"/"grid_points_world.npy"
        full_block = np.load(str(grid_path.resolve()))
        # grid_world = 


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

scores, mask_result = RL.utils.mask_points(voxel_data, 
        occupied_points_world, 
        mask_subset, 
        c2w_subset,
        cam_matrix,
        device = device,
        debug_dir= str( ( Path(data_dir)/"result"/"voxel" ).resolve() ) 
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



# Now filter based on threshold

mask_thres = 0.5
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

output_path = Path(data_dir)/"result"/"vox_masked.vox"
VoxWriter(str(output_path.resolve()), vox).write()
print('The Vox file created in ', str(output_path))

# Now export the points of the new voxels
output_path_points = Path(data_dir)/"result"/"vox_masked_points.npy"
np.save( str(output_path_points.resolve()), occupied_points_centered_orig[voxel_indices] )




