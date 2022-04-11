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
import RLResearch.utils.gen_utils as gu

img_id = 3

NPY_PREFIX = "depth_"

project_folder = "/workspace/datasets/cctv_2"
data_folder = project_folder + "/result/depth_npy/"
ply_folder = project_folder + "/result/debug_ply"
Path(ply_folder).mkdir(exist_ok=True, parents=True)


dataset = datasets["nsvf"](
            project_folder,
            split="test_train"
            )


def export_pointcloud(img_id, data_folder, ply_folder):
    depth_filename = data_folder + "/" + NPY_PREFIX + str(img_id) + ".npy"
    depth_img_np = np.load(depth_filename)
    pointcloud_filename =  Path(ply_folder)/Path(depth_filename).name
    pointcloud_filename = pointcloud_filename.with_suffix(".ply")


    c2w = dataset.c2w[img_id].to(device = "cpu")
    print("Rendering pose:", img_id)
    print(c2w)
    # print("ndc")
    # print(dataset.ndc_coeffs)

    # Can take from somewhere else that takes less time to load
    # width=dataset.get_image_size(0)[1]
    # height=dataset.get_image_size(0)[0]
    # fx = dataset.intrins.get('fx', 0)
    # fy = dataset.intrins.get('fy', 0)
    # cx = dataset.intrins.get('cx', 0)
    # cy = dataset.intrins.get('cy', 0)

    width = 3840
    height = 2160
    fx = 3243.357296552027
    fy = 3243.357296552027
    cx = 1920.0
    cy = 1080.0

    # depth_images = [depth_img_np]

    radial_weight = du.make_radial_weight(width, height, fx)
    depth_img_np = depth_img_np * radial_weight
    depth_img_np = depth_img_np.astype(np.float32)
    du.write_pointcloud_from_depth(
                            depth_img_np, 
                            str(pointcloud_filename.resolve()), 
                            w = width,
                            h = height,
                            fx = fx,
                            fy = fy,
                            cx = cx,
                            cy = cy,
                            stride = 8, transform = c2w) 


for i in [0, 3, 5, 8, 10, 90, 130, 180]:
    export_pointcloud(i, data_folder , ply_folder)


