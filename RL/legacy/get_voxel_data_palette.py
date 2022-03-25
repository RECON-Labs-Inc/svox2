from pathlib import Path
import sys
from datetime import datetime
import argparse

sys.path.append("..")
sys.path.append("/workspace/aseeo-research")

import torch
from svox2 import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from importlib import reload as reload

import RLResearch.utils.gen_utils as gu


reload(svox2)
from svox2 import *

from pyvox.models import Vox, Color
from pyvox.writer import VoxWriter

def data_hist(colors,filename, num_bins = 30):
    fig = plt.figure()
    ax = fig.gca()
    ax.hist(colors, num_bins)
    print(filename)
    plt.savefig(filename, format='png')

def load_palette(filename):
    pili = Image.open(filename)
    return np.asarray(pili)



checkpoint = "/workspace/datasets/dog/ckpt/std/ckpt.npz"
palette_filename = "/workspace/data/vox_palette.png"

# TODO: Change this for any available gpu
device = "cpu"
# device = "cuda:0" # Getting ooms
grid = SparseGrid.load(checkpoint, device)

# # Single point
# data_points = torch.tensor([[0, 0, 0]], device=torch.device('cuda:1'), dtype=torch.float32)
# density, color = grid.sample(data_point)

# Grid
grid_dim = 250
grid_res = torch.tensor([grid_dim, grid_dim, grid_dim])

xx = torch.linspace(0, grid.links.shape[0],grid_res[0] )
yy = torch.linspace(0, grid.links.shape[1],grid_res[1] )
zz = torch.linspace(0, grid.links.shape[2],grid_res[2] )

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

#  ------- Filtering ------
# Density thresholding
thres = 0
positive_dens_bool = density[:,0] > thres

# Positive color indices
# Reshape 27 into 3 times 9
color = utils.SH_C0 * color # Take only albedo component
## Test this:
color = color + 0.5
# rgb_color_factor = 10
# rgb_colors = rgb_color_factor * rgb_colors 

positive_components = color > 0
positive_color_bool = positive_components.all(axis=1)

# Join (AND) the positive densities with the positive colors (why are they negative?)

filtered_indices = torch.logical_and(positive_dens_bool, positive_color_bool).nonzero() 
# Filtering either negative colors or negative densities.
# filtered_indices = positive_dens_bool.nonzero()
# filtered_indices = positive_color_bool.nonzero()
filtered_indices = filtered_indices[:,0]

filtered_colors = color[filtered_indices, :]
# filtered_density = density[positive_dens_bool,0]
# filtered_points = grid_points[positive_dens_bool[:,0], ...]

# Scale colors
filtered_colors_scaled = filtered_colors * 255
print("Grid shape", grid.links.shape)

# ------ Compute color distance --------
palette = torch.tensor(load_palette(palette_filename), dtype=torch.float32, device = device)

# Remove alpha
palette = palette[0,:,:3]

# Compute distance
color_dists = torch.cdist(filtered_colors_scaled, palette)

# Get min distance (closest color)
mins = torch.min(color_dists, dim = 1)
color_indices = mins.indices

# paletted_colors = palette[color_indices].cpu().numpy().astype(np.uint8)

# ------- Make palette ----------
vox_pal = []
np_palette = palette.cpu().numpy().astype(np.uint8)

for c in np_palette:
    vox_pal.append(Color(c[0], c[1], c[2], 255))

# TODO: Revise this
# vox_pal = [Color(0, 0, 0, 0)] + vox_pal
## Voxel of size 
color_labels = np.zeros((grid_dim*grid_dim*grid_dim))
color_labels[filtered_indices.cpu().numpy()] = color_indices.cpu().numpy()
# print(vox_pal)

#  -------- Export voxels -------
##### Saving vox file #####
print("Saving vox file...")
vox = Vox.from_dense(color_labels.astype(np.uint8).reshape(grid_dim, grid_dim, grid_dim))
vox.palette = vox_pal

fn = 'test-%s.vox'%datetime.now().isoformat().replace(':', '_')
# result_path = Path(checkpoint).parent/"results"
# result_path.mkdir(exist_ok=True)
# output_path = result_path / fn
output_path  = "/workspace/data/test_color_%s.vox"%datetime.now().isoformat().replace(':', '_')
print('The Vox file created in ', str(output_path))
VoxWriter(output_path, vox).write()


# fn = 'test-%s.vox'%datetime.now().isoformat().replace(':', '_') + "_" + str(grid.links.shape[0])
# exp_name = Path(checkpoint).parent.parent
# result_path = exp_name/"result"
# result_path.mkdir(exist_ok=True)
# output_path = result_path / ( exp_name.name + str(grid.links.shape[0]))
# data_file = output_path.with_suffix(".ply")


# # Will encode the density as a color
# filtered_density_col = torch.broadcast_to(filtered_density, (filtered_density.shape[0], 3))
# filtered_density_col_np = np.array(torch.tensor(filtered_density_col.cpu()))

# gu.write_point_cloud(np.array(filtered_points.cpu()), colors=np.array(filtered_colors.cpu()), filename=data_file, color_range=1)
# gu.write_point_cloud(np.array(filtered_points.cpu()), colors = filtered_density_col_np , filename = gu.path_add_suffix(data_file, "_dens"), color_range=1000)

# data_hist(np.array(filtered_colors.cpu()), str(output_path.with_suffix(".png"))  )
# data_hist(np.array(filtered_density.detach().cpu()),
#     gu.path_add_suffix( output_path.with_suffix(".png"), "_density") )