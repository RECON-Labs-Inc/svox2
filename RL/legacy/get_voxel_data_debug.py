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

import RLResearch.utils.gen_utils as gu

from importlib import reload as reload

reload(svox2)
from svox2 import *

def data_hist(colors,filename, num_bins = 30):
    fig = plt.figure()
    ax = fig.gca()
    ax.hist(colors, num_bins)
    print(filename)
    plt.savefig(filename, format='png')
    

checkpoint = "/workspace/datasets/cactus/ckpt/std/ckpt.npz"
cuda_device = 1
grid = SparseGrid.load(checkpoint, cuda_device)

# # Single point
# data_points = torch.tensor([[0, 0, 0]], device=torch.device('cuda:1'), dtype=torch.float32)
# density, color = grid.sample(data_point)

# Grid

grid_res = torch.tensor([250, 250, 250])

xx = torch.linspace(0, grid.links.shape[0],grid_res[0] )
yy = torch.linspace(0, grid.links.shape[1],grid_res[1] )
zz = torch.linspace(0, grid.links.shape[2],grid_res[2] )

grid_mesh = torch.meshgrid(xx, yy, zz)
grid_points = torch.cat((grid_mesh[0][...,None], grid_mesh[1][...,None], grid_mesh[2][...,None]), 3)
num_voxels = grid_points.shape[0] * grid_points.shape[1] * grid_points.shape[2]
grid_points = grid_points.reshape((num_voxels, 3))
grid_points = torch.tensor(grid_points, device=torch.device('cuda:1'), dtype=torch.float32)

print("GP", grid_points.shape)
density, color = grid.sample_max(grid_points, grid_coords=True)
print(density.shape)
print(color.shape)
# Filter
thres = 0
indices = density[:,0] > thres
indices = indices.nonzero()

filtered_density = density[indices,0]
filtered_points = grid_points[indices[:,0], ...]


#Reshape 27 into 3 times 9
color_rs = color.reshape(-1, 3, grid.basis_dim)
filtered_colors = color_rs[indices[:,0], ...]
rgb_colors = utils.SH_C0 * filtered_colors[:,:,0]
# rgb_color_factor = 10
# rgb_colors = rgb_color_factor * rgb_colors 


# Filter negative colors
positive_components = rgb_colors > 0
positive_indices = positive_components.all(axis=1)
rgb_colors = rgb_colors[positive_indices, :]
filtered_density = filtered_density[positive_indices, :]
filtered_points = filtered_points[positive_indices, :]



rgb_colors = torch.tensor(rgb_colors)
print("Grid shape", grid.links.shape)
# fn = 'test-%s.vox'%datetime.now().isoformat().replace(':', '_') + "_" + str(grid.links.shape[0])
exp_name = Path(checkpoint).parent.parent
result_path = Path(checkpoint).parent.parent/"result"
result_path.mkdir(exist_ok=True)
output_path = result_path / ( exp_name.name + str(grid.links.shape[0]))
data_file = output_path.with_suffix(".ply")

# Will encode the density as a color
filtered_density_col = torch.broadcast_to(filtered_density, (filtered_density.shape[0], 3))
filtered_density_col_np = np.array(torch.tensor(filtered_density_col.cpu()))

gu.write_point_cloud(np.array(filtered_points.cpu()), colors=np.array(rgb_colors.cpu()), filename=data_file, color_range=1)
gu.write_point_cloud(np.array(filtered_points.cpu()), colors = filtered_density_col_np , filename = gu.path_add_suffix(data_file, "_dens"), color_range=1000)

data_hist(np.array(rgb_colors.cpu()), str(output_path.with_suffix(".png"))  )
data_hist(np.array(filtered_density.detach().cpu()),
    gu.path_add_suffix( output_path.with_suffix(".png"), "_density") )