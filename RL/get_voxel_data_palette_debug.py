from pathlib import Path
import sys
from datetime import datetime
import argparse
import pickle

sys.path.append("..")
sys.path.append("/workspace/aseeo-research")
# sys.path.append(.)

import torch
from svox2 import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from importlib import reload as reload

import RLResearch.utils.gen_utils as gu
import RLResearch.utils.pose_utils as pu

from utils import colorize_using_palette, filter_sphere, palette_from_file

reload(svox2)
from svox2 import *

from pyvox.models import Vox, Color
from pyvox.writer import VoxWriter

# checkpoint = "/workspace/datasets/TanksAndTempleBG/Truck/ckpt/tt_test/ckpt.npz"
# checkpoint = "/workspace/datasets/cube_2/ckpt/std/ckpt.npz"
# checkpoint = "/workspace/datasets/cctv/ckpt/scale_test/ckpt.npz"
# checkpoint = "/workspace/datasets/cactus/ckpt/std/ckpt.npz"

# checkpoint = "/workspace/datasets/cactus/ckpt/std/ckpt.npz"
# data_dir = "/workspace/datasets/cactus"

# checkpoint = "/workspace/datasets/_b_shoe_200/ckpt/std/ckpt.npz"
# data_dir  = "/workspace/datasets/_b_shoe_200"

saturate = True
saturation_factor = 2.2
palette_filename = "/workspace/data/vox_palette.png"
grid_dim = 256
# Careful with this: SparseGrid wants values in args.json grid range (which could be different from grid_dim)


## ARGPARSE
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str,default=None, help=".npz checkpoint file")
parser.add_argument("--data_dir", type=str,default=None, help="Project folder")
parser.add_argument("--grid_dim", type=int, default = 256, help = "grid_dimension")
parser.add_argument("--saturate", action="store_true", help="Boost saturation of voxel colors")
args = parser.parse_args()
checkpoint = args.checkpoint
data_dir = args.data_dir
grid_dim = args.grid_dim
saturate = args.saturate
## -----

device = "cpu"

grid = SparseGrid.load(checkpoint, device)

# # Single point
# data_points = torch.tensor([[0, 0, 0]], device=torch.device('cuda:1'), dtype=torch.float32)
# density, color = grid.sample(data_point)

# Grid
# It would be ideal if grid_dim is the same or multiple of training grid. But not strictly necessary?
grid_res = torch.tensor([grid_dim, grid_dim, grid_dim])

xx = torch.linspace(0, grid.links.shape[0] - 1 , grid_res[0] )
yy = torch.linspace(0, grid.links.shape[1] - 1 , grid_res[1] )
zz = torch.linspace(0, grid.links.shape[2] - 1 , grid_res[2] )

grid_mesh = torch.meshgrid(xx, yy, zz)
grid_points = torch.cat((grid_mesh[0][...,None], grid_mesh[1][...,None], grid_mesh[2][...,None]), 3)
num_voxels = grid_points.shape[0] * grid_points.shape[1] * grid_points.shape[2]
grid_points = grid_points.reshape((num_voxels, 3))
grid_points = torch.tensor(grid_points, device=device, dtype=torch.float32)

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

# Filter with spherical boundary
sphere_filter_factor = 0.35 # Percentage of bounding box
color_labels = filter_sphere(color_labels, 
                                    grid_points / (grid.shape[0] - 1 ), # grid_points are in grid_coord. I'm normalizing here (assuming square grid)
                                    center = torch.tensor([0, 0, 0]),
                                    radius = sphere_filter_factor)

vox_pal = palette_from_file("/workspace/data/vox_palette.png")
# vox_pal.append(Color(0, 0, 0, 255))
# vox_pal.append(Color(128, 128, 128, 255))
# vox_pal.append(Color(255, 0, 0, 255))

vox = Vox.from_dense(color_labels.astype(np.uint8).reshape(grid_dim, grid_dim, grid_dim))
vox.palette = vox_pal

result_folder = Path(data_dir)/"result"/"voxel"
result_folder.mkdir(exist_ok=True, parents=True)
output_path = result_folder/"vox.vox"

print('The Vox file created in ', str(output_path))
VoxWriter(output_path, vox).write()

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
np.save(str(grid_points_path.resolve()), grid_points_world )
print("Saved grid points to ", grid_points_path)


# # Save bounding block (sculpting block)
# mins = np.amin(occupied_grid_points.cpu().numpy(), axis=0)
# maxs = np.amax(occupied_grid_points.cpu().numpy(), axis = 0)

# bounding_block = grid_

grid_data = {}
grid_data["color"] = color
grid_data["density"] = density

print("Writing color and density to grid_data.pkl")
grid_data_path = result_folder/"grid_data.pkl"
with open(str(grid_data_path.resolve()), 'wb') as handle:
    pickle.dump(grid_data, handle)

