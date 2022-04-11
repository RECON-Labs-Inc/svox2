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
from rembg.bg import remove as remove_backround

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle

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

def colorize_using_palette(voxel_data, color, palette_filename, grid_dim, add_offset = True, color_factor = None, saturate=False, saturation_factor = 1.6, device = "cuda:0"):
    
    # color = color.reshape((grid_dim, grid_dim, grid_dim, 3))
    color = torch.tensor(color, device = device,dtype=torch.float32 )
    voxel_data_flat = voxel_data.flatten()
    filtered_indices = torch.tensor(np.array(voxel_data_flat.nonzero()), device = device)
    filtered_indices = filtered_indices[0, :]
    filtered_colors = color[filtered_indices, :]

    if saturate:
        processed_colors = np.zeros((filtered_colors.shape[0], filtered_colors.shape[1]))
        filtered_colors_np = filtered_colors.detach().numpy()

        for i in range(0, filtered_colors.shape[0]):           
            processed_colors[i, :] = saturate_color(filtered_colors[i,:].detach().numpy(), saturation_factor = saturation_factor)

        filtered_colors = torch.tensor(processed_colors, device=device, dtype=torch.float32)

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

    # TODO: Revise this
    # vox_pal = [Color(0, 0, 0, 0)] + vox_pal
    ## Voxel of size 
    color_labels = np.zeros((grid_dim*grid_dim*grid_dim))
    color_labels[filtered_indices.cpu().numpy()] = color_indices.cpu().numpy() + [1 if add_offset else 0]
    
    return color_labels, vox_pal

# def colorize_using_classifier():
#     pass

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

def colorize_using_classifier( voxel_data, color, grid_dim = None, thres = 0, n_clusters=6, saturate=False, saturation_factor = 2.2, 
                                paletize = False, palette_filename = None, device = "cuda:0"):
    """Colorize using classifier"""
    
    device = "cuda:0"
    # color = torch.tensor(color, device = device,dtype=torch.float32 )
    voxel_data_flat = voxel_data.flatten()
    filtered_indices = voxel_data_flat.nonzero()
    filtered_indices = filtered_indices[0]
    filtered_colors = color[filtered_indices, :]

    if saturate:
        processed_colors = np.zeros((filtered_colors.shape[0], filtered_colors.shape[1]))
        filtered_colors_np = filtered_colors.detach().numpy()

        for i in range(0, filtered_colors.shape[0]):           
            processed_colors[i, :] = saturate_color(filtered_colors_np[i], saturation_factor = saturation_factor)

        filtered_colors = torch.tensor(processed_colors, device=device, dtype=torch.float32)

    # Scale colors
    filtered_colors_scaled = filtered_colors * 255

    image_array_sample = shuffle(filtered_colors.detach().numpy(), random_state=0, n_samples=min(len(filtered_colors), 10000))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image_array_sample)

    # filtered_labels = kmeans.predict(filtered_colors.detach().numpy()) + 2 ## IMPORTANT!!!!
    filtered_labels = kmeans.predict(filtered_colors.detach().numpy())
    np_palette = (kmeans.cluster_centers_ * 255).astype('B')
    
    np_color_labels = np.zeros_like(voxel_data_flat)

    if paletize is True: # Paletize the colors using a palette
        if palette_filename is not None:
            input_colors = np_palette[filtered_labels]
            np_color_labels[filtered_indices], pal = paletize_colors(input_colors, palette_filename)

        else:
            raise ValueError("You should input a palette filename")
    else:
        # Just output the colors normally
        
        pal = []
        for c in np_palette:
            pal.append(Color(c[0], c[1], c[2], 255))
            
        pal = [Color(0, 0, 0, 0)] + pal # Add 'palette for EMPTY voxel'. This is why the n_clusters should be less than 255
            
        np_color_labels = np.zeros_like(voxel_data_flat)
        np_color_labels[filtered_indices] = filtered_labels + 2 # Weird indexing issue for voxels

    return np_color_labels.reshape(grid_dim, grid_dim, grid_dim), pal

def paletize_colors(input_colors, palette_filename, device = "cuda:0", add_offset = True):
    """Find closest colors to a given palette
    
    Args:
        input_colors: tensor of shape (n_colors x 3)
    Returns:
        color_labels: np.array of shape (n_colors,), with indices to the palette
        pal: palette
    """
    
    filtered_colors_scaled = torch.tensor(input_colors, device = device, dtype=torch.float32)
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

    # TODO: Revise this
    # vox_pal = [Color(0, 0, 0, 0)] + vox_pal
    ## Voxel of size 
    
    # Note: these color labels are NOT (grid_dim, grid_dim, grid_dim). They are the color labels for the INPUT colors (which should correspond to only occupied voxels)
    color_labels = color_indices.cpu().numpy() + [1 if add_offset else 0]
    
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
    
    np_palette = load_palette(filename).astype(np.uint8)
    np_palette = np_palette[0]
    vox_pal = []

    for c in np_palette:
        vox_pal.append(Color(c[0], c[1], c[2], 255))
    
    return vox_pal

def make_grid(grid_dim):

    grid_res = torch.tensor([grid_dim, grid_dim, grid_dim])

    xx = torch.linspace(0, grid.links.shape[0] - 1 , grid_res[0] )
    yy = torch.linspace(0, grid.links.shape[1] - 1 , grid_res[1] )
    zz = torch.linspace(0, grid.links.shape[2] - 1 , grid_res[2] )

    grid_mesh = torch.meshgrid(xx, yy, zz)
    grid_points = torch.cat((grid_mesh[0][...,None], grid_mesh[1][...,None], grid_mesh[2][...,None]), 3)
    num_voxels = grid_points.shape[0] * grid_points.shape[1] * grid_points.shape[2]
    grid_points = grid_points.reshape((num_voxels, 3))
    grid_points = torch.tensor(grid_points, device=device, dtype=torch.float32)

    return grid_points

def prepare_points(tensor, scale_factor, inv_recenter_matrix, device = "cuda:0"):
    # Untested
    tensor = torch.tensor(tensor, device = device, dtype=torch.float64)

    # Rescale position
    tensor *= (1.0/scale_factor)
    ones = torch.ones((1, tensor.shape[0]), device=device)
    tensor = torch.cat((tensor.T, ones))

    # Convert to world
    tensor_world = inv_recenter_matrix.double() @ tensor

    return tensor_world


def mask_points(voxel_data, occupied_points_world, mask_subset, c2w_subset, cam_matrix, device = "cuda:0", debug_dir = None):
        """ Masks voxels (voxel_data) according to their position (occuped_points_world) 
        using masks (mask_subset) at view_points(c2w_subset). Projects the points into each mask 
        and checks if the point should be masked. Return a score by averaging the mask value from all masks"""

        num_voxels = occupied_points_world.shape[1]
        i = 0
        scores = torch.zeros(num_voxels, device=device) # init score matrix
        for mask_image, c2w in zip(mask_subset, c2w_subset):

                # Convert to tensor
                mask_image = torch.tensor(mask_image, device = device)
                c2w =  torch.tensor(c2w, device = device, dtype=torch.float64) 

                # World to camera
                w2c = torch.linalg.inv(c2w)

                
                # --- Project to mask: (1) World to cam, then (2) Project into viewport with cam matrix
                occupied_points_cam = w2c @ occupied_points_world
                occupied_points_projected = cam_matrix.double() @ occupied_points_cam[:3,:]
                occupied_points_projected = occupied_points_projected[:2, :]/occupied_points_projected[2, :] # Divide by z to get coords.
                projected_indices = occupied_points_projected.type(torch.long)

                projected_indices_x = projected_indices[0,:]
                projected_indices_y = projected_indices[1,:]

                height, width = mask_image.shape

                projected_indices_x_clamped = projected_indices[0,:].clamp(0, width-1) # This is not the proper way to deal with out of bounds this
                projected_indices_y_clamped = projected_indices[1,:].clamp(0, height-1)


                masked_results = mask_image[projected_indices_y_clamped,  projected_indices_x_clamped] # Swapped x y attention!
                scores += masked_results

                mask_thres = 0.5
                mask_thres_int = mask_thres * 255
                mask_thres_int = int(mask_thres_int)

                # masked_values = torch.
                masked_indices = (masked_results < mask_thres_int).nonzero()
                mask_result = torch.ones(occupied_points_projected.shape[1]) * 8 # Just using 8 as a random color
                mask_result[masked_indices] = 99

                voxel_data_flat = voxel_data.flatten()
                voxel_indices = voxel_data_flat.nonzero()
                voxel_indices = voxel_indices[0]
                voxel_data_flat[voxel_indices] = mask_result.cpu().numpy()

                # Rewrite voxel data
                voxel_data = voxel_data_flat.reshape(voxel_data.shape)
                vox = Vox.from_dense(voxel_data)
                # vox.palette = 
                # p_projected = p_projected

                # output_path = "/workspace/data/vox_mask_"+str(i)+".vox"

                if debug_dir is not None:
                        output_path=Path(debug_dir)/("vox_mask_"+str(i)+".vox")
                        VoxWriter(output_path, vox).write()
                        print('The Vox file created in ', str(output_path))
                
                i+=1

        # Divede by number of masks to normalize to range 255
        scores = scores / len(mask_subset)
        
        return scores, mask_result


def make_masks(source_image_filenames, downsample = False, save_folder = None):
    """Makes masks from filenames. Returns a list of masks"""
    
    dilate_image = True
    dilatation_size = 25
    downsample = True
    masks = []
    # i = 0
    for source_image, ind  in zip(source_image_filenames, range(0, len(source_image_filenames))):
    # for ind in subset:
        #Compute mask here based on file list.
        print(str(ind), str(source_image))
        pili = Image.open(source_image)
        max_dimension = max((pili.width, pili.height))
        if downsample is True:
                if max_dimension >= 3840:
                        mask_downsample = 4
                elif max_dimension >= 1920:
                        mask_downsample = 2
                else:
                        mask_downsample = 1
        
                pili = pili.resize( ( int(pili.width/mask_downsample), int(pili.height/mask_downsample) ))
                mask = remove_backround(pili)
                mask = mask.resize( (mask.width * mask_downsample, mask.height * mask_downsample) )
        else:
                mask = remove_backround(pili)

        mask = mask.split()[-1]
        mask = np.asarray(mask)
        # mask_subset.append(np.asarray(mask))

        if dilate_image:
                mask = gu.dilate_image( mask, dilatation_size= dilatation_size)
                
        masks.append(mask)

        if save_folder is not None:
            im = Image.fromarray(mask)
            im.save(source_image)

    return masks

def saturate_color(color, saturation_factor = 1.3):
    """color is (3,) numpy array"""
    import colorsys
    hsv = np.array(colorsys.rgb_to_hsv(color[0], color[1], color[2]))
    hsv = hsv * np.array([1, saturation_factor, 1]) # Multiply saturation by value. Leave the rest untouched
    color = colorsys.hsv_to_rgb(hsv[0], hsv[1], hsv[2])
    # return np.clip(np.array(color), 0, 1) # Probably done inside the library (didn't see any difference)
    return np.array(color)