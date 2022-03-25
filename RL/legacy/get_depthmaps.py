import sys
from pathlib import Path
from datetime import datetime
import argparse
import json

import torch
from torchvision.utils import save_image
import torchvision

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle

import open3d as o3d


from svox2 import *
from pyvox.models import Vox, Color
from pyvox.writer import VoxWriter

from importlib import reload as reload

reload(svox2)
from svox2 import *

#TODO> modify this:
sys.path.append("/workspace/svox2/opt")
from util.dataset import datasets
from util import config_util

# Our nice tools
sys.path.append("/workspace/aseeo-research")
import RLResearch.utils.depth_utils as du

data_dir = "/workspace/datasets/dog"
exp_name = "std"
checkpoint_path = Path(data_dir)/"ckpt"/exp_name/"ckpt.npz"


# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"

# Load arguments from json
json_config_path = Path(data_dir)/"ckpt"/exp_name/"args.json"
parser = argparse.ArgumentParser()
with open(str(json_config_path.resolve()), 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

# parser = argparse.ArgumentParser()
# args = parser.parse_args([])

dataset = datasets["nsvf"](
            data_dir,
            split="test",
            device=device,
            factor=1,
            n_images=None,
            **config_util.build_data_options(args))

grid = SparseGrid.load(str(checkpoint_path.resolve()))
# grid.requires_grad_(True)
config_util.setup_render_opts(grid.opt, args)
print('Render options', grid.opt)


## Single camera position
img_id = 0
c2w = dataset.c2w[img_id].to(device = device)
print("Rendering pose:", img_id)
print(c2w)
print("ndc")
print(dataset.ndc_coeffs)

cam = svox2.Camera(c2w,
                    dataset.intrins.get('fx', img_id),
                    dataset.intrins.get('fy', img_id),
                    dataset.intrins.get('cx', img_id),
                    dataset.intrins.get('cy', img_id),
                    width=dataset.get_image_size(img_id)[1],
                    height=dataset.get_image_size(img_id)[0],
                    ndc_coeffs=dataset.ndc_coeffs)

print("Cam is cuda", cam.is_cuda)

print("Using thres", args.log_depth_map_use_thresh)

# NOTE: no_grad enables the fast image-level rendering kernel for cuvol backend only
# other backends will manually generate rays per frame (slow)
with torch.no_grad():
        depth_img = grid.volume_render_depth_image(cam,
                                    args.log_depth_map_use_thresh if
                                    args.log_depth_map_use_thresh else None
                                , batch_size=500)


## Export colored pointcloud to check in meshlab
depth_o3d = o3d.geometry.Image(depth_img.numpy())
intrinsics = o3d.camera.PinholeCameraIntrinsic(
                                        cam.width,
                                        cam.height,
                                        dataset.intrins.get('fx', img_id),
                                        dataset.intrins.get('fy', img_id),
                                        dataset.intrins.get('cx', img_id),
                                        dataset.intrins.get('cx', img_id)
                                )

pointcloud = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsics, stride = 8)

o3d.io.write_point_cloud("/workspace/data/pointcloud.ply", pointcloud)

a = 5