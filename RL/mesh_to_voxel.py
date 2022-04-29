import sys
from pathlib import Path
from datetime import datetime
import argparse
import json
from PIL import Image
import time

import open3d as o3d
import numpy as np

from pyvox.models import Vox, Color
from pyvox.writer import VoxWriter
from pyvox.parser import VoxParser

from svox2 import *

#TODO> modify this:
sys.path.append("/workspace/svox2/opt")

# Our nice tools
sys.path.append("/workspace/aseeo-research")
import RLResearch.utils.depth_utils as du
import RLResearch.utils.gen_utils as gu

sys.path.append("/workspace/svox2")
import RL.utils


# ## ARGPARSE
parser = argparse.ArgumentParser()
parser.add_argument("--obj_file", type = str, default=None,  help="OBJ file to be loaded")
# parser.add_argument("--checkpoint", type=str,default=None, help=".npz checkpoint file")
parser.add_argument("--data_dir", type=str,default=None, help="Project folder")
parser.add_argument("--grid_size", type=int, default=20, help = "Grid... size...")
# parser.add_argument("--num_masks", type=int, default = 20, help = "number of masks used to mask/sculpt the object")
# parser.add_argument("--source", type=str, default = "images_undistorted", help = "subfolder where images are located")
# parser.add_argument("--use_block", action="store_true" ,  help = "Use block")
# parser.add_argument("--mask_thres", type=float , default=0.5,  help = "Values less than mask_thres will be masked")
# parser.add_argument("--debug_folder", type=str,default=None, help="debug folder for saving stuff")
args = parser.parse_args()

obj_file = args.obj_file
data_dir = args.data_dir
grid_size = args.grid_size

mesh = o3d.io.read_triangle_mesh(obj_file)
print("Will compute voxelization on mesh", mesh)
mesh = o3d.geometry.TriangleMesh.simplify_quadric_decimation(mesh, 8000)
print("Will compute voxelization on downsampled mesh", mesh)

# fit to unit cube
mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
           center=mesh.get_center())
voxel_size = 1/grid_size
start = time.time()
voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh,
                                                              voxel_size=voxel_size)
computation_time = time.time() - start
print("voxel computed in ", computation_time , "seconds")
# Voxels to numpy array
voxels = voxel_grid.get_voxels()
num_voxels = len(voxels)
voxel_indices = np.zeros((num_voxels,3), dtype=np.uint)

for i in range(0, len(voxels)):
    voxel_indices[i,:] = voxels[i].grid_index

# voxel_indices = np.array(voxel_indices)
print(voxel_indices.shape)

# grid_size = max(64, np.max(voxel_indices))
# grid_size = int(1/voxel_size)
print("Grid size", grid_size)
voxel_labels = np.zeros((grid_size, grid_size, grid_size))
voxel_indices = voxel_indices - 1
voxel_labels[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1

# Shift axis
voxel_labels = np.moveaxis(voxel_labels, 0, 1)

# Save
vox = Vox.from_dense(voxel_labels.astype(np.uint))
# vox.palette = original_palette


output_path = Path(data_dir)/"result"/"voxel"/"vox_mesh.vox"


VoxWriter(str(output_path.resolve()), vox).write()
print('The Vox file created in ', str(output_path))
