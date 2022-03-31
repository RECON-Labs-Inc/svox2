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
from pyvox.parser import VoxParser

saturate = True
saturation_factor = 2.2
palette_filename = "/workspace/data/vox_palette.png"
grid_dim = 256
# Careful with this: SparseGrid wants values in args.json grid range (which could be different from grid_dim)


## ARGPARSE
parser = argparse.ArgumentParser()
parser.add_argument("--vox_file", type = str, default=None,  help="Vox file to be colorized")
parser.add_argument("--data_dir", type=str,default=None, help="Project folder")
parser.add_argument("--grid_dim", type=int, default = 256, help = "grid_dimension")
parser.add_argument("--saturate", action="store_true", help="Boost saturation of voxel colors")
parser.add_argument("--debug_folder", type=str,default=None, help="debug folder for saving stuff")
args = parser.parse_args()
data_dir = args.data_dir
grid_dim = args.grid_dim
saturate = args.saturate
vox_file = args.vox_file
debug_folder = args.debug_folder
## -----

device = "cpu"

# ---- Load vox file

m = VoxParser(vox_file).parse()
voxel_data = m.to_dense()
orig_voxel_data = voxel_data
original_palette = m.palette

vox_pal = palette_from_file("/workspace/data/vox_palette.png")
# vox_pal.append(Color(0, 0, 0, 255))
# vox_pal.append(Color(128, 128, 128, 255))
# vox_pal.append(Color(255, 0, 0, 255))

result_folder = Path(data_dir)/"result"/"voxel"
grid_data_path = Path(result_folder)/"grid_data.pkl"

with open(str(grid_data_path.resolve()), 'rb') as f:
        grid_data = pickle.load(f)

# grid_data = pickle.load(str(grid_data_path.resolve()))
color = grid_data["color"]

color_labels, vox_pal = colorize_using_palette(voxel_data, color, palette_filename, grid_dim, add_offset = True, color_factor = None, saturate=False, saturation_factor = 1.6, device = "cuda:0")

result_folder = Path(data_dir)/"result"/"voxel"
result_folder.mkdir(exist_ok=True, parents=True)
output_path = result_folder/"vox_colorized.vox"

vox = Vox.from_dense(color_labels.astype(np.uint8).reshape(grid_dim, grid_dim, grid_dim))
vox.palette = vox_pal

VoxWriter(output_path, vox).write()
print('The Vox file created in ', str(output_path))

if debug_folder is not None:
        Path(debug_folder).mkdir(exist_ok=True, parents=True)
        VoxWriter( str(Path(debug_folder)/output_path.name), vox).write()
