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

# checkpoint = "/workspace/datasets/TanksAndTempleBG/Truck/ckpt/tt_test/ckpt.npz"
# checkpoint = "/workspace/datasets/cube_2/ckpt/std/ckpt.npz"
checkpoint = "/workspace/datasets/dog/ckpt/std/ckpt.npz"
palette_filename = "/workspace/data/vox_palette.png"

## ARGPARSE
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str,default=None, help=".npz checkpoint file")
args = parser.parse_args()
checkpoint = args.checkpoint
def data_hist(colors,filename, num_bins = 30, custom_palette = None):
    fig = plt.figure()
    ax = fig.gca()
    
    N, bins, patches = ax.hist(colors, num_bins) # From https://stackoverflow.com/questions/49290266/python-matplotlib-histogram-specify-different-colours-for-different-bars

    # Change colors if palette exists, else do nothing. 
    if custom_palette is not None:
        if len(custom_palette) != num_bins:
            raise ValueError("Palette (", len(custom_palette), ") has different size than bins (", num_bins, ").")
        else:
            # Color each bin by palette
            for custom_color, patch in zip(custom_palette, patches):
                print(custom_color)
                patch.set_facecolor(custom_color/255.0)

    print(filename)
    plt.savefig(filename, format='png')



def load_palette(filename):
    pili = Image.open(filename)
    return np.asarray(pili)

def colorize_using_palette(density, color, add_half = False, add_offset = True, thres = 0, color_factor = None):
    #  ------- Filtering ------
    # Density thresholding
    # thres = 0
    positive_dens_bool = density[:,0] > thres
    negative_dens_bool = torch.logical_not(positive_dens_bool)
    pos_dens_ind = positive_dens_bool.nonzero()

    ## Test this:
    if add_half:
        color = color + 0.5

    if color_factor is not None:
        color = color * color_factor
    # rgb_color_factor = 10
    # rgb_colors = rgb_color_factor * rgb_colors 

    positive_components = color > 0
    positive_color_bool = positive_components.all(axis=1)
    negative_color_bool = torch.logical_not(positive_color_bool)

    # Join (AND) the positive densities with the positive colors (why are they negative?)

    # filtered_indices = torch.logical_and(positive_dens_bool, positive_color_bool).nonzero() 
    # Filtering either negative colors or negative densities.
    filtered_indices = positive_dens_bool.nonzero()
    # filtered_indices = positive_color_bool.nonzero()
    neg_color_indices = negative_color_bool.nonzero()
    anti_filtered_indices = negative_color_bool.nonzero()
    filtered_indices = filtered_indices[:,0]

    pos_d_neg_c_ind = torch.logical_and(positive_dens_bool, negative_color_bool).nonzero()
    pos_d_pos_c_ind = torch.logical_and(positive_dens_bool, positive_color_bool).nonzero()
    filtered_colors = color[filtered_indices, :]

    filtered_density = density[pos_dens_ind,0]
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

    # for i in range(0, 255):
    #     vox_pal.append(Color(i, i, i, 255))

    # TODO: Revise this
    # vox_pal = [Color(0, 0, 0, 0)] + vox_pal
    ## Voxel of size 
    color_labels = np.zeros((grid_dim*grid_dim*grid_dim))
    color_labels[filtered_indices.cpu().numpy()] = color_indices.cpu().numpy() + [1 if add_offset else 0]
    

    # # Anti-filter debug
    # pos_d_pos_col_color = Color(37, 245, 5, 255)
    # pos_d_neg_col_color = Color(252,2,44, 255)
    # anti_filter_debug = True
    # if anti_filter_debug is True:
    #     vox_pal[1] = pos_d_pos_col_color
    #     vox_pal[2] = pos_d_neg_col_color
    #     color_labels[pos_dens_ind] = 2 # There is an offset in the palette!
    #     color_labels[pos_d_pos_c_ind] = 3

    return color_labels, vox_pal
def colorize_pos_neg(density, color, thres = 0, add_offset = True):
    
    #  ------- Filtering ------
    # Density thresholding
    # thres = 0
    positive_dens_bool = density[:,0] > thres
    negative_dens_bool = torch.logical_not(positive_dens_bool)
    pos_dens_ind = positive_dens_bool.nonzero()

    positive_components = color > 0
    positive_color_bool = positive_components.all(axis=1)
    negative_color_bool = torch.logical_not(positive_color_bool)

    # Will plot positive density voxels only
    filtered_indices = positive_dens_bool.nonzero()
    filtered_indices = filtered_indices[:,0]

    pos_d_neg_c_ind = torch.logical_and(positive_dens_bool, negative_color_bool).nonzero()
    pos_d_pos_c_ind = torch.logical_and(positive_dens_bool, positive_color_bool).nonzero()
    filtered_colors = color[filtered_indices, :]

    filtered_density = density[pos_dens_ind,0]
    # filtered_points = grid_points[positive_dens_bool[:,0], ...]

    # Scale colors
    filtered_colors_scaled = filtered_colors * 255
    print("Grid shape", grid.links.shape)

    # ------- Make palette ----------
    # TODO: Revise this
    # vox_pal = [Color(0, 0, 0, 0)] + vox_pal
    ## Voxel of size 
    color_labels = np.zeros((grid_dim*grid_dim*grid_dim))
    # color_labels[filtered_indices.cpu().numpy()] = color_indices.cpu().numpy() + [1 if add_offset else 0]
    
    vox_pal = [None] * 3
    vox_pal[0] = Color(0, 0, 0, 255)
    # Green for pos, red for neg
    pos_d_pos_col_color = Color(37, 245, 5, 255)
    pos_d_neg_col_color = Color(252,2,44, 255)
    vox_pal[1] = pos_d_pos_col_color
    vox_pal[2] = pos_d_neg_col_color
    color_labels[pos_d_pos_c_ind] = 2 # There is an offset in the palette!
    color_labels[pos_d_neg_c_ind] = 3

    return color_labels, vox_pal

def colorize_using_classifier(density, color):
    #  ------- Filtering ------
    # Density thresholding
    thres = 0
    positive_dens_bool = density[:,0] > thres
    negative_dens_bool = torch.logical_not(positive_dens_bool)
    pos_dens_ind = positive_dens_bool.nonzero()

    ## Test this:
    # if add_half:
    #     color = color + 0.5
    # rgb_color_factor = 10
    # rgb_colors = rgb_color_factor * rgb_colors 

    positive_components = color > 0
    positive_color_bool = positive_components.all(axis=1)
    negative_color_bool = torch.logical_not(positive_color_bool)

    # Join (AND) the positive densities with the positive colors (why are they negative?)

    filtered_indices = torch.logical_and(positive_dens_bool, positive_color_bool).nonzero() 
    # Filtering either negative colors or negative densities.
    # filtered_indices = positive_dens_bool.nonzero()
    # filtered_indices = positive_color_bool.nonzero()
    neg_color_indices = negative_color_bool.nonzero()
    anti_filtered_indices = negative_color_bool.nonzero()
    filtered_indices = filtered_indices[:,0]

    pos_d_neg_c_ind = torch.logical_and(positive_dens_bool, negative_color_bool).nonzero()
    pos_d_pos_c_ind = torch.logical_and(positive_dens_bool, positive_color_bool).nonzero()
    filtered_colors = color[filtered_indices, :]

    filtered_density = density[pos_dens_ind,0]
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

    # for i in range(0, 255):
    #     vox_pal.append(Color(i, i, i, 255))

    # TODO: Revise this
    # vox_pal = [Color(0, 0, 0, 0)] + vox_pal
    ## Voxel of size 
    color_labels = np.zeros((grid_dim*grid_dim*grid_dim))
    color_labels[filtered_indices.cpu().numpy()] = color_indices.cpu().numpy() + [1 if add_offset else 0]
    

    # # Anti-filter debug
    # pos_d_pos_col_color = Color(37, 245, 5, 255)
    # pos_d_neg_col_color = Color(252,2,44, 255)
    # anti_filter_debug = True
    # if anti_filter_debug is True:
    #     vox_pal[1] = pos_d_pos_col_color
    #     vox_pal[2] = pos_d_neg_col_color
    #     color_labels[pos_dens_ind] = 2 # There is an offset in the palette!
    #     color_labels[pos_d_pos_c_ind] = 3

    return color_labels, vox_pal

def colorize_using_density(density, color, thres = 0, max_density = 1000, debug_hist_filename = None, add_offset=True):

    #  ------- Filtering ------
    # Density thresholding
    # thres = 0

    positive_dens_bool = density[:,0] > thres
    negative_dens_bool = torch.logical_not(positive_dens_bool)
    pos_dens_ind = positive_dens_bool.nonzero()

    # Will plot positive density voxels only
    filtered_indices = positive_dens_bool.nonzero()
    filtered_indices = filtered_indices[:,0]

    filtered_density = density[pos_dens_ind,0]

    # Scale colors
    print("Grid shape", grid.links.shape)

    # ------- Make palette ----------
    # TODO: Revise this
    # vox_pal = [Color(0, 0, 0, 0)] + vox_pal
    ## Voxel of size 
    color_labels = np.zeros((grid_dim*grid_dim*grid_dim))
    color_labels[filtered_indices.cpu().numpy()] = (255*filtered_density.detach().numpy()[:,0]/max_density).astype(np.uint8) + [1 if add_offset else 0]
    
    vox_pal = []
    color_bar = []
    for i in range(0, 255):
        vox_pal.append(Color(i, i, i, 255))
        color_bar.append(np.array([i, i, i]))

    print("Length VOX_pal", len(vox_pal))

    # Write the histogram to file
    if debug_hist_filename is not None:
        data_hist(np.array(filtered_density.detach().cpu()), 
                                                debug_hist_filename , 
                                                num_bins=255,
                                                custom_palette=color_bar
                                                )

    return color_labels, vox_pal, color_bar



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

color = utils.SH_C0 * color # Take only albedo component

color_labels, vox_pal = colorize_using_palette(density, color, thres = 0,  add_half = True, color_factor=0.9)
# color_labels, vox_pal = colorize_pos_neg(density, color, thres = 0)

output_path  = "/workspace/data/test_color_%s.vox"%datetime.now().isoformat().replace(':', '_')

# color_labels, vox_pal, color_bars = colorize_using_density(
#                                                     density, 
#                                                     color,
#                                                     thres=150,
#                                                     max_density=600,
#                                                     debug_hist_filename=
#                                                     gu.path_add_suffix( Path(output_path).with_suffix(".png"), "_density"))

# data_hist(np.array(filtered_colors.detach().cpu()), str(Path(output_path).with_suffix(".png"), custom_palette= color_bars)  )
# data_hist(np.array(filtered_density.detach().cpu()),
#     gu.path_add_suffix( Path(output_path).with_suffix(".png"), "_density") )

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

