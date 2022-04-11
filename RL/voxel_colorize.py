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

from utils import colorize_using_palette
from utils import filter_sphere
from utils import palette_from_file
from utils import colorize_using_classifier

reload(svox2)
from svox2 import *

from pyvox.models import Vox, Color
from pyvox.writer import VoxWriter
from pyvox.parser import VoxParser


## ARGPARSE
parser = argparse.ArgumentParser()
parser.add_argument("--vox_file", type = str, default=None,  help="Vox file to be colorized")
parser.add_argument("--data_dir", type=str,default=None, help="Project folder")
parser.add_argument("--saturate", action="store_true", help="Boost saturation of voxel colors")
parser.add_argument("--color_mode", type=str,default="palette", choices = ["palette", "classifier"], help="Type of colorizing. Either classifier (kmeans clustering) or palette")
parser.add_argument("--n_clusters", type=int,default=8, help="Number of colors used in classifier")
parser.add_argument("--paletize", action="store_true", help="Use a palette and output colors accordingly. You should also input a palette filename")
parser.add_argument("--palette_filename", type = str, default="/workspace/data/vox_palette.png",  help="Palette png")
parser.add_argument("--debug_folder", type=str,default=None, help="debug folder for saving stuff")
args = parser.parse_args()
data_dir = args.data_dir
saturate = args.saturate
vox_file = args.vox_file
debug_folder = args.debug_folder
color_mode = args.color_mode
n_clusters = args.n_clusters
paletize = args.paletize
palette_filename = args.palette_filename

## -----

device = "cpu"

# ---- Load vox file

m = VoxParser(vox_file).parse()
voxel_data = m.to_dense()
orig_voxel_data = voxel_data
original_palette = m.palette

vox_pal = palette_from_file("/workspace/data/vox_palette.png")

grid_dim = voxel_data.shape[0]
if len(np.unique(voxel_data.shape)) > 1 :
        raise ValueError("Voxel is not cubic")

result_folder = Path(data_dir)/"result"/"voxel"
grid_data_path = Path(result_folder)/"grid_data.pkl"

with open(str(grid_data_path.resolve()), 'rb') as f:
        grid_data = pickle.load(f)

color = grid_data["color"]

if color_mode == "palette":
        color_labels, vox_pal = colorize_using_palette(voxel_data, color, palette_filename, grid_dim, add_offset = True, color_factor = None, saturate=False, saturation_factor = 1.6, device = "cuda:0")
elif color_mode == "classifier":
        color_labels, vox_pal = colorize_using_classifier(voxel_data, color, grid_dim = grid_dim, n_clusters=n_clusters, saturate = saturate, paletize=paletize, palette_filename=palette_filename, device = device)
else:
        raise ValueError("Unrecognized color mode")

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
