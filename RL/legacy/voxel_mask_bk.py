import sys
from pathlib import Path
from datetime import datetime
import argparse
import json
import pickle
from PIL import Image


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

data_dir = "/workspace/datasets/cctv"
exp_name = "scale_test"
checkpoint_path = Path(data_dir)/"ckpt"/exp_name/"ckpt.npz"
vox_file = "/workspace/datasets/cctv/result/voxel/vox.vox"
grid_dim = 256 # My sampling
orig_grid_dim = 640 # Actual grid size of grid from args.json

# # ## ARGPARSE
# parser = argparse.ArgumentParser()
# parser.add_argument("--checkpoint", type=str,default=None, help=".npz checkpoint file")
# parser.add_argument("--data_folder", type=str,default=None, help="Project folder")
# parser.add_argument("--grid_dim", type=int, default = 256, help = "grid_dimension")
# parser.add_argument("--vox_file", type=str, default = None, help = "Voxel file to be masked")
# args = parser.parse_args()
# checkpoint = args.checkpoint
# data_folder = args.data_folder
# grid_dim = args.grid_dim

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"

# Load arguments from json
# json_config_path = Path(data_dir)/"ckpt"/exp_name/"args.json"

dataset = datasets["nsvf"](
            data_dir,
            split="test_train",
            device=device,
            factor=1,
            n_images=None)

grid = SparseGrid.load(str(checkpoint_path.resolve()))
# config_util.setup_render_opts(grid.opt, args)
# print('Render options', grid.opt)


## Single camera position
img_id = 0
c2w = dataset.c2w[img_id].to(device = device)
print("Rendering pose:", img_id)
print(c2w)
print("ndc")
print(dataset.ndc_coeffs)

cam = svox2.Camera(c2w,
                    dataset.intrins.get('fx', img_id),
                    dataset.intrins.get('fy', img_id),
                    dataset.intrins.get('cx', img_id),
                    dataset.intrins.get('cy', img_id),
                    width=dataset.get_image_size(img_id)[1],
                    height=dataset.get_image_size(img_id)[0],
                    ndc_coeffs=dataset.ndc_coeffs)

print("Cam is cuda", cam.is_cuda)

# print("Using thres", args.log_depth_map_use_thresh)


### ---  LOAD images from dataset
mask_path = Path(data_dir)/"source"/"masks"
masks = []
if mask_path.exists() is False:
        raise FileNotFoundError(mask_path, "does not exist")
else:
        for mask_image in sorted(mask_path.iterdir()):
                print(mask_image)
                pili = Image.open(str(mask_image.resolve()))
                masks.append(np.asarray(pili))

num_masks = 10
subset = range(0, len(masks), 10)

mask_subset = []
c2w_subset = []
for ind in subset:
        mask_subset.append(masks[ind])
        c2w = dataset.c2w[ind].to(device = device)
        c2w_subset.append(c2w)

### --- LOAD points from npy file
voxel_npy_path = Path(data_dir)/"project_files"/"voxel_points.npy"
occupied_points_centered = np.load(str(voxel_npy_path.resolve()))

m = VoxParser(vox_file).parse()
voxel_data = m.to_dense()

### --- Make grid points
grid_res = torch.tensor([grid_dim, grid_dim, grid_dim])

xx = torch.linspace(0, grid.links.shape[0] - 1 , grid_res[0] )
yy = torch.linspace(0, grid.links.shape[1] - 1 , grid_res[1] )
zz = torch.linspace(0, grid.links.shape[2] - 1 , grid_res[2] )

grid_mesh = torch.meshgrid(xx, yy, zz)
grid_points = torch.cat((grid_mesh[0][...,None], grid_mesh[1][...,None], grid_mesh[2][...,None]), 3)
num_voxels = grid_points.shape[0] * grid_points.shape[1] * grid_points.shape[2]
grid_points = grid_points.reshape((num_voxels, 3))
grid_points = torch.tensor(grid_points, device=device, dtype=torch.float32)

voxel_indices = voxel_data.flatten().nonzero()
voxel_indices = voxel_indices[0]
occupied_points = grid_points[voxel_indices,:]

# Convert points to world coordinates
occupied_points =  2 * (occupied_points / (grid_dim - 1)) * grid_dim / orig_grid_dim
occupied_points -= torch.tensor([0.5, 0.5, 0.5], device=device) # Recenter


# Load camera transformation
cam_filename = Path(data_dir)/"project_files"/"cam_data.json"
camera_dict = json.load(open(str(cam_filename.resolve())))
nsvf_recenter_matrix = torch.tensor(camera_dict["nsvf_recenter_matrix"], device = device)
inv_recenter_matrix = torch.linalg.inv(nsvf_recenter_matrix)
scale_factor = camera_dict["scale_factor"]

# Rescale position (invert)
occupied_points *= (1.0/scale_factor)


a = 5
ind = 5
mask_image = torch.tensor(mask_subset[ind], device = device)
c2w =  c2w_subset[ind]

# project to image
# World to camera
w2c = torch.linalg.inv(c2w)
# dset_h, dset_w = dset.get_image_size(img_id)
height, width = dataset.get_image_size(ind)
# fx = (1.0/scale_factor) * dataset.intrins.get('fx', ind)
# fy = (1.0/scale_factor) * dataset.intrins.get('fy', ind)
fx = dataset.intrins.get('fx', ind)
fy = dataset.intrins.get('fy', ind)
cx = dataset.intrins.get('cx', ind)
cy = dataset.intrins.get('cy', ind)

cam_matrix = torch.tensor([ [fx, 0, cx],
                        [0, fy, cy],
                        [0,  0, 1 ]], device = device)

num_voxels = occupied_points.shape[0]
print(num_voxels)

ones = torch.ones((1, occupied_points.shape[0]), device=device)
occupied_points = torch.cat((occupied_points.T, ones))
occupied_points = inv_recenter_matrix @ occupied_points # Points in original world coordinate system
cam_orig = c2w # Original camera position
cam_orig[:3, 3] = (1.0/scale_factor) * cam_orig[:3, 3]  #scale position
cam_orig = inv_recenter_matrix @ cam_orig # TODO: I suspect there is loss in precision here. check this out. Maybe matrices should be double
w2c_orig = torch.linalg.inv(cam_orig)
p_cam = w2c_orig @ occupied_points
p_projected = cam_matrix.float() @ p_cam[:3, :]
p_projected = p_projected[:2, :]/p_projected[2, :] # Divide by z to get coords.
projected_indices = p_projected.type(torch.long)

projected_indices_x = projected_indices[0,:]
projected_indices_y = projected_indices[1,:]

projected_indices_x_clamped = projected_indices[0,:].clamp(0, width-1) # This is not the proper way to deal with out of bounds this
projected_indices_y_clamped = projected_indices[1,:].clamp(0, height-1)


masked_results = mask_image[projected_indices_y_clamped,  projected_indices_x_clamped] # Swapped x y attention!


mask_thres = 0.5
mask_thres_int = mask_thres * 255
mask_thres_int = int(mask_thres_int)

# masked_values = torch.
masked_indices = (masked_results < mask_thres_int).nonzero()
mask_result = torch.ones(occupied_points.shape[1]) * 8 # Just using 8 as a random color
mask_result[masked_indices] = 99


voxel_data_flat = voxel_data.flatten()
voxel_data_flat[voxel_indices] = mask_result.cpu().numpy()

# Rewrite voxel data
voxel_data = voxel_data_flat.reshape(voxel_data.shape)
vox = Vox.from_dense(voxel_data)
# vox.palette = 
# p_projected = p_projected

output_path = "/workspace/data/vox_mask.vox"
VoxWriter(output_path, vox).write()
print('The Vox file created in ', str(output_path))

## Save pose info
# Occupied voxels
debug_data = {}
debug_data["c_voxels"] = grid_points[voxel_indices,:]

# Centered poses
c_poses = np.array([ c2w.to(device = "cpu").numpy() for c2w in dataset.c2w ])
debug_data["c_poses"] = c_poses

# Mask pose subset
debug_data["c_mask_subset"] = np.array([c2w.cpu().numpy() for c2w in c2w_subset])
a = 5

# Transformed voxels
debug_data["w_voxels"] = occupied_points

# Projected voxels
debug_data["p_voxels"] = p_projected

# Projected indices clamped
projected_indices_x_clamped
projected_indices_y_clamped
debug_data["projected_indices"]=(projected_indices_x_clamped, projected_indices_y_clamped)


with open(str((Path(data_dir)/'debug_data.pickle').resolve()), 'wb') as f:
    pickle.dump(debug_data,f)

projected_debug = torch.zeros((width, height)) ## Check if I should transpose this
a = 5

pointcloud = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsics, stride = 8)

o3d.io.write_point_cloud("/workspace/data/pointcloud.ply", pointcloud)

a = 5

#  -------- Export voxels -------
##### Saving vox file #####
print("Saving vox file...")
vox = Vox.from_dense(color_labels.astype(np.uint8).reshape(grid_dim, grid_dim, grid_dim))
vox.palette = vox_pal

fn = 'test-%s.vox'%datetime.now().isoformat().replace(':', '_')
# result_path = Path(checkpoint).parent/"results"
# result_path.mkdir(exist_ok=True)
# output_path = result_path / fn

print('The Vox file created in ', str(output_path))
VoxWriter(output_path, vox).write()