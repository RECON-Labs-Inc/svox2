import sys
from pathlib import Path
from datetime import datetime
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle

from svox2 import *
from pyvox.models import Vox, Color
from pyvox.writer import VoxWriter

from importlib import reload as reload

reload(svox2)
from svox2 import *



def make_voxels(args):
    # checkpoint = str(Path("/workspace/svox2/opt/ckpt/conv_test/ckpt.npz"))
    checkpoint = "/workspace/datasets/TanksAndTempleBG/Truck/ckpt/tt_test/ckpt.npz"
    # checkpoint = args.checkpoint
    # checkpoint = "/workspace/leo_test/pear/ckpt/ckpt.npz"
    cuda_device = 1
    grid = SparseGrid.load(checkpoint, cuda_device)

    ##### grid sampling #####
    print("Sampling grids...")
    grid_dim = 250
    grid_res = torch.tensor([grid_dim, grid_dim, grid_dim])

    xx = torch.linspace(0, grid.links.shape[0],grid_res[0] )
    yy = torch.linspace(0, grid.links.shape[1],grid_res[1] )
    zz = torch.linspace(0, grid.links.shape[2],grid_res[2] )

    grid_mesh = torch.meshgrid(xx, yy, zz)
    grid_points = torch.cat((grid_mesh[0][...,None], grid_mesh[1][...,None], grid_mesh[2][...,None]), 3)
    num_voxels = grid_points.shape[0] * grid_points.shape[1] * grid_points.shape[2]
    grid_points = grid_points.reshape((num_voxels, 3))
    grid_points = torch.tensor(grid_points, device = torch.device('cuda:1'), dtype=torch.float32)

    print("GP", grid_points.shape)
    # color, density = grid.sample(grid_points, grid_coords=True)
    color, density = grid.sample_max(grid_points, grid_coords=True)

    density_cube = torch.cat( density, dim=1)
    color_cube = torch.stack( color, dim = 2)
    color_intensity = torch.linalg.norm(color_cube, dim = 1)

    density_max, max_indices = torch.max(density_cube, dim=1)

    # dens # Caution, overrides previous max_indices 
    _, max_indices  = torch.max(color_intensity, dim=1)

    density = density_max
    r = torch.gather(color_cube[:,0,:], 1, max_indices[..., None])
    g = torch.gather(color_cube[:,1,:], 1, max_indices[..., None])
    b = torch.gather(color_cube[:,2,:], 1, max_indices[..., None])

    color = torch.cat((r, g, b),dim =1)



    # color = color.detach.numpy()
    print(density.shape)
    print(color.shape)
    ##### density thresholding #####
    print("Density Thresholding...")
    np_density = np.abs(density.detach().numpy())

    h = np.histogram(np_density)
    print(h)

    thres = 30
    density_idx = np_density < thres
    np_density[density_idx] = 0
    nonzero_idxs = np_density.nonzero()

    h = np.histogram(np_density)
    print(h)

    # rescale
    np_density = (np_density / np_density.max() * 255).astype('B') # mapped to 0 ~ 255


    ##### color filtering #####
    print("Color Filtering...")
    # np_color = color.detach().numpy()[:, [0, 9, 18]] # for sh_dim=9 trained data
    np_color = color.detach().numpy() # for sh_dim = 1 trained data

    # Ignore Negative Colors. 
    # DISCUSSION: is this 'right' thing to do?
    idx = np_color < 0 
    np_color[idx] = 0

    # Rescale color range to 0~1
    print(f"MAX COLOR: {np_color.max()}")
    # np_color = np_color / np_color.max() 
    SH_C0 = 0.28209479177387814
    SH_C0 = 1.5
    np_color = np_color * SH_C0 # mult SH first order coeff
    idx_gt_one = np_color > 1 
    np_color[idx_gt_one] = 1
    nonzero_colors = np_color[nonzero_idxs[0], :]

    h = np.histogram(np_color[:, 0])
    print("Color histogram or R channel")
    print(h)


    ##### Making Color palette for vox file by K-means clustering #####
    print("Making the color palette...")
    # This should be less than 254, NOT 255
    # - since we gonna add 0 to our vox file palette. (+1)
    # - and (MAYBE) K-means preserve first index for outliers?
    n_clusters=254  

    image_array_sample = shuffle(nonzero_colors, random_state=0, n_samples=min(len(nonzero_colors), 10000))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image_array_sample)

    labels = kmeans.predict(np_color) + 2 ## IMPORTANT!!!!

    np_palette = (kmeans.cluster_centers_ * 255).astype('B')

    # pal = []
    # for c in np_palette:
    #     if np.linalg.norm(c) == 0:
    #         # Among k-means clusters, there might be colors that has close to 0, which should be considers as blank. maybe not blank but black?!
    #         pal.append(Color(0, 0, 0, 0))  
    #     else:
    #         pal.append(Color(c[0], c[1], c[2], 255))

    pal = []
    for c in np_palette:
        pal.append(Color(c[0], c[1], c[2], 255))
        
    pal = [Color(0, 0, 0, 0)] + pal # Add 'palette for EMPTY voxel'. This is why the n_clusters should be less than 255

    print(pal)

    print("Labeling empty voxels...")
    # give EMTPY color label to density == 0 voxels which we previously filtered with 'thres'
    for i, (lab, den) in enumerate(zip(labels, np_density)):
        if den == 0:
            labels[i] = 0
            
    np_color_labels = labels.reshape(grid_dim, grid_dim, grid_dim)

    ##### Saving vox file #####
    print("Saving vox file...")
    vox = Vox.from_dense(np_color_labels)
    vox.palette = pal

    fn = 'test-%s.vox'%datetime.now().isoformat().replace(':', '_')
    result_path = Path(checkpoint).parent/"results"
    result_path.mkdir(exist_ok=True)
    output_path = result_path / fn
    print('The Vox file created in ', str(output_path))
    VoxWriter(output_path, vox).write()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help="Checkpoint folder")
    # parser.add_argument('--output_folder', type=str, help="output folder")

    args = parser.parse_args()

    make_voxels(args)
    
    
