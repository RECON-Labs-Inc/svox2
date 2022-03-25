from pathlib import Path
import sys
from datetime import datetime
import argparse

sys.path.append("../..")
sys.path.append("/workspace/aseeo-research")

import torch
from svox2 import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from importlib import reload as reload

import RLResearch.utils.gen_utils as gu
import RLResearch.utils.pose_utils as pu


reload(svox2)
from svox2 import *

from pyvox.models import Vox, Color
from pyvox.writer import VoxWriter

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

def filter_sphere(color_labels, grid_points, center = torch.tensor([0, 0, 0]), radius = 5):
    """Removes all voxels outside of a spherical boundary"""
    distance = torch.linalg.norm( ( grid_points - torch.tensor([0.5, 0.5, 0.5]))  - center, axis = 1) # grid is centered in the corner, so subtracting (0.5, 0.5, 0.5)
    outside_indices = (distance > radius).nonzero()
    outside_indices = outside_indices[:,0]
    
    color_labels[outside_indices] = 0
    return color_labels


    return filtered_color_labels

def load_palette(filename):
    pili = Image.open(filename)
    return np.asarray(pili)

def colorize_using_palette(density, color, palette_filename, grid_dim, add_half = True, add_offset = True, thres = 0, color_factor = None, saturate=False, saturation_factor = 1.6, device = "cuda:0"):
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

    if saturate:
        def saturate_color(color, saturation_factor = 1.3):
            """color is (3,) numpy array"""
            import colorsys
            hsv = np.array(colorsys.rgb_to_hsv(color[0], color[1], color[2]))
            hsv = hsv * np.array([1, saturation_factor, 1]) # Multiply saturation by value. Leave the rest untouched
            color = colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])
            # return np.clip(np.array(color), 0, 1) # Probably done inside the library (didn't see any difference)
            return np.array(color)
        
        processed_colors = np.zeros((filtered_colors.shape[0], filtered_colors.shape[1]))
        filtered_colors_np = filtered_colors.detach().numpy()

        for i in range(0, filtered_colors.shape[0]):           
            processed_colors[i, :] = saturate_color(filtered_colors[i,:].detach().numpy(), saturation_factor = saturation_factor)

        filtered_colors = torch.tensor(processed_colors, device=device, dtype=torch.float32)

    filtered_density = density[pos_dens_ind,0]
    # filtered_points = grid_points[positive_dens_bool[:,0], ...]

    # Scale colors
    filtered_colors_scaled = filtered_colors * 255
    # print("Grid shape", grid.links.shape)

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

def colorize_using_classifier(density, color, thres = 0):
    #  ------- Filtering ------
    # Density thresholding
    # thres = 0
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

    # Join (logical AND) the positive densities with the positive colors (why are they negative?)

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

def palette_from_file(filename):
    """Make the simplest possible palette, with 1 colors. Empty, and gray"""
    
    np_palette = load_palette(filename).astype(np.uint8)
    np_palette = np_palette[0]
    vox_pal = []

    for c in np_palette:
        vox_pal.append(Color(c[0], c[1], c[2], 255))
    
    return vox_pal
    