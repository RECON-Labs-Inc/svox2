from pathlib import Path
import sys
from datetime import datetime
import argparse
import pickle
from importlib import reload as reload
from PIL import Image
from math import pi as PI

import torch
import numpy as np
from scipy.spatial.transform import Rotation as Rotation
import matplotlib.pyplot as plt
from pyvox.models import Vox, Color
from pyvox.writer import VoxWriter

from svox2 import *
from utils import colorize_using_palette, filter_sphere, palette_from_file

sys.path.append("..")
sys.path.append("/workspace/aseeo-research")

import RLResearch.utils.gen_utils as gu
import RLResearch.utils.pose_utils as pu


## ARGPARSE
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str,default=None, help=".npz checkpoint file")
parser.add_argument("--data_dir", type=str,default=None, help="Project folder")
parser.add_argument("--grid_dim", type=int, default = 256, help = "grid_dimension")
parser.add_argument("--debug_folder", type=str,default=None, help="debug folder for saving stuff")
parser.add_argument("--euler_angles", type=float, nargs=3 ,default=None, help="Euler angles for rotation")
parser.add_argument("--euler_mode", type=str, default="zyx", help="Euler angle rotation order")
parser.add_argument("--dummy", action="store_true", help="Make a dummy axis voxel")
args = parser.parse_args()
checkpoint = args.checkpoint
data_dir = args.data_dir
grid_dim = args.grid_dim
debug_folder = args.debug_folder
euler_mode = args.euler_mode
dummy = args.dummy

if args.euler_angles is None:
    euler_angles = None
else:
    euler_angles = np.array(args.euler_angles)

if debug_folder is not None:
    print("Debugging to ", debug_folder)

print("Euler angles:", euler_angles)
print("Grid", grid_dim)

## -----

device = "cpu"

grid = SparseGrid.load(checkpoint, device)

# # Single point
# data_points = torch.tensor([[0, 0, 0]], device=torch.device('cuda:1'), dtype=torch.float32)
# density, color = grid.sample(data_point)

# Grid
# It would be ideal if grid_dim is the same or multiple of training grid. But not strictly necessary?
grid_res = torch.tensor([grid_dim, grid_dim, grid_dim])


orig_grid_dim = grid.links.shape[0] # Assuming square grid.

xx = torch.linspace(0, grid.links.shape[0] - 1 , grid_res[0] )
yy = torch.linspace(0, grid.links.shape[1] - 1 , grid_res[1] )
zz = torch.linspace(0, grid.links.shape[2] - 1 , grid_res[2] )

grid_mesh = torch.meshgrid(xx, yy, zz)
grid_points = torch.cat((grid_mesh[0][...,None], grid_mesh[1][...,None], grid_mesh[2][...,None]), 3)
num_voxels = grid_points.shape[0] * grid_points.shape[1] * grid_points.shape[2]
grid_points = grid_points.reshape((num_voxels, 3))
grid_points = torch.tensor(grid_points, device=device, dtype=torch.float32)

if euler_angles is not None:
    c = PI/180.0
    r = Rotation.from_euler(euler_mode,[c * euler_angles[0], c * euler_angles[1], c * euler_angles[2]])
    rot_mat = torch.tensor(r.as_matrix(), device = device, dtype=torch.float32)
    print(rot_mat)
    offset = torch.tensor([orig_grid_dim/2, orig_grid_dim/2, orig_grid_dim/2 ])
    grid_points = torch.matmul(rot_mat, (grid_points - offset).T).T + offset


print("GP", grid_points.shape)

sample_max_density = True
sample_max_colors = True

if sample_max_colors:
    if device != "cpu":
        ValueError("Sample max only works in CPU mode")
    else:
        # Outputs a list of 8 values (cube)
        _, color = grid.sample_max(grid_points, grid_coords=True, use_kernel = False)

        # Stack tuples to make an array
        color_cube = torch.stack( color, dim = 2)

        if grid.basis_dim > 1:
            color_cube = color_cube.reshape(-1, 3, grid.basis_dim, 8) # Split component wise
            color_cube = color_cube[:, :, 0, :] # Take only first SH for each color
        # Take the most intense color
        color_intensity = torch.linalg.norm(color_cube, dim = 1)

        # dens # Caution, overrides previous max_indices 
        _, max_indices  = torch.max(color_intensity, dim=1)

        # Don't remember how this works.......
        r = torch.gather(color_cube[:,0,:], 1, max_indices[..., None])
        g = torch.gather(color_cube[:,1,:], 1, max_indices[..., None])
        b = torch.gather(color_cube[:,2,:], 1, max_indices[..., None])

        color = torch.cat((r, g, b),dim =1)
else:
    _, color = grid.sample(grid_points, grid_coords=True)
    color = color.reshape(-1, 3, grid.basis_dim)
    color = color[:, :, 0] # Take only 1st spherical harmonic

if sample_max_density:
    if device != "cpu":
        ValueError("Sample max only works in CPU mode")
    else:
        # Outputs a list of 8 values (cube)
        density, _ = grid.sample_max(grid_points, grid_coords=True, use_kernel = False)
        density_cube = torch.cat( density, dim=1)
        density_max, max_indices = torch.max(density_cube, dim=1)
        density = density_max
        density = density[..., None]
else:
    density, _ = grid.sample(grid_points, grid_coords=True)
     
print(density.shape)
print(color.shape)

color = utils.SH_C0 * color # Take only albedo component

# For some mysterious reason, color is from -0.5 to 0.5, we need to add 0.5
color = color + 0.5

## ------ MAKE VOXEL --------
# Occupied voxels have positive densities
thres = 0
positive_dens_bool = density[:,0] > thres
negative_dens_bool = torch.logical_not(positive_dens_bool)
pos_dens_ind = positive_dens_bool.nonzero()

filtered_indices = positive_dens_bool.nonzero()
filtered_indices = filtered_indices[:,0]

filtered_colors = color[filtered_indices, :]
filtered_density = density[pos_dens_ind,0]

# Color labels are irrelevant at this stage. We will colorize later. Here, we only care about occupancy.
color_labels = np.zeros((grid_dim*grid_dim*grid_dim))
color_labels[filtered_indices.cpu().numpy()] = 2 # first index of palette, else is zero



sphere_filter_factor = 0.5 # Percentage of bounding box
color_labels = filter_sphere(color_labels, 
                                    grid_points / (grid.shape[0] - 1 ), # grid_points are in grid_coord. I'm normalizing here (assuming square grid)
                                    center = torch.tensor([0, 0, 0]),
                                    radius = sphere_filter_factor)

# Filter with spherical boundary
if dummy is True:
    color_labels = np.zeros((grid_dim, grid_dim, grid_dim))
    half = int(grid_dim/2)
    l = 20
    color_labels[half:half + l,  half,   half] = 201 + 2 # RED
    color_labels[half ,half:half + l,   half] = 121 + 2# GREEN
    color_labels[half  ,  half, half:half + l] = 73 + 2# BLUE
    color_labels = color_labels.flatten()
    print("Writing dummy vox file")
    print(half)

vox_pal = palette_from_file("/workspace/data/vox_palette.png")

vox = Vox.from_dense(color_labels.astype(np.uint8).reshape(grid_dim, grid_dim, grid_dim))
vox.palette = vox_pal

result_folder = Path(data_dir)/"result"/"voxel"
result_folder.mkdir(exist_ok=True, parents=True)
output_path = result_folder/"vox.vox"

print('The Vox file created in ', str(output_path))
VoxWriter(output_path, vox).write()
if debug_folder is not None:
    Path(debug_folder).mkdir(exist_ok=True, parents=True)
    VoxWriter( str(Path(debug_folder)/"vox.vox"), vox).write()

## ------- SAVE DENSITY, COLOR AND POINTS (COORDS) ----
# Convert grid to world coordinates and save them in a file
occupied_points_indices = color_labels.nonzero()[0]
occupied_grid_points  = grid_points[occupied_points_indices, :]
occupied_grid_points_centered = grid.grid2world(occupied_grid_points)
occupied_grid_points_centered = occupied_grid_points_centered.cpu().numpy()

voxel_point_path = Path(data_dir)/"project_files"/"voxel_points.npy"
np.save(str(voxel_point_path.resolve()), occupied_grid_points_centered )
print("Saved voxel points to ", voxel_point_path)

grid_points_world = grid.grid2world(grid_points)
grid_points_world = grid_points_world.cpu().numpy()
grid_points_path = Path(data_dir)/"project_files"/"grid_points.npy"
# print(grid_points_world[:10,...])
np.save(str(grid_points_path.resolve()), grid_points_world )
print("Saved grid points to ", grid_points_path)

grid_data = {}
grid_data["color"] = color
grid_data["density"] = density

print("Writing color and density to grid_data.pkl")
grid_data_path = result_folder/"grid_data.pkl"
with open(str(grid_data_path.resolve()), 'wb') as handle:
    pickle.dump(grid_data, handle)


# Debug this file with the following configuration in visual studio code. (Excerpt from launch.json)

        # {
        #     "name": "Make",
        #     "type": "python",
        #     "request": "launch",
        #     "program": "${file}",
        #     "console": "integratedTerminal",
        #     "justMyCode": true,
        #     "args" : [
        #         "--checkpoint", "/workspace/datasets/cactus_prep_d/ckpt/std/ckpt.npz",
        #         "--data_dir", "/workspace/datasets/cactus_prep_d",
        #         "--debug_folder", "/workspace/data/cactus_prep_d",
        #         "--grid_dim", "128"
            
        #     ]
        #     // --checkpoint /workspace/datasets/$dataset/ckpt/std/ckpt.npz --data_dir /workspace/datasets/$dataset --debug_folder /workspace/data/$dataset --grid_dim $grid_dim
        # }