# Copyright 2021 Alex Yu
# Eval

import torch
import svox2
import svox2.utils
import math
import argparse
import numpy as np
import os
from os import path
from pathlib import Path
from util.dataset import datasets
from util.util import Timing, compute_ssim, viridis_cmap
from util import config_util

import imageio
import cv2
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('ckpt', type=str)

config_util.define_common_args(parser)


# TODO: [SERGIO] Cleanup and get rid of unnecesary options
parser.add_argument('--n_eval', '-n', type=int, default=100000, help='images to evaluate (equal interval), at most evals every image')
parser.add_argument('--train', action='store_true', default=False, help='render train set')
parser.add_argument('--render_path',
                    action='store_true',
                    default=False,
                    help="Render path instead of test images (no metrics will be given)")
parser.add_argument('--timing',
                    action='store_true',
                    default=False,
                    help="Run only for timing (do not save images or use LPIPS/SSIM; "
                    "still computes PSNR to make sure images are being generated)")
parser.add_argument('--no_lpips',
                    action='store_true',
                    default=False,
                    help="Disable LPIPS (faster load)")
parser.add_argument('--no_vid',
                    action='store_true',
                    default=False,
                    help="Disable video generation")
parser.add_argument('--no_imsave',
                    action='store_true',
                    default=False,
                    help="Disable image saving (can still save video; MUCH faster)")
parser.add_argument('--fps',
                    type=int,
                    default=30,
                    help="FPS of video")

# Camera adjustment
parser.add_argument('--crop',
                    type=float,
                    default=1.0,
                    help="Crop (0, 1], 1.0 = full image")

# Foreground/background only
parser.add_argument('--nofg',
                    action='store_true',
                    default=False,
                    help="Do not render foreground (if using BG model)")
parser.add_argument('--nobg',
                    action='store_true',
                    default=False,
                    help="Do not render background (if using BG model)")

# Random debugging features
parser.add_argument('--blackbg',
                    action='store_true',
                    default=False,
                    help="Force a black BG (behind BG model) color; useful for debugging 'clouds'")
parser.add_argument('--ray_len',
                    action='store_true',
                    default=False,
                    help="Render the ray lengths")

args = parser.parse_args()
config_util.maybe_merge_config_file(args, allow_invalid=True)
device = 'cuda:0'

if args.timing:
    args.no_lpips = True
    args.no_vid = True
    args.ray_len = False

if not args.no_lpips:
    import lpips
    lpips_vgg = lpips.LPIPS(net="vgg").eval().to(device)
if not path.isfile(args.ckpt):
    args.ckpt = path.join(args.ckpt, 'ckpt.npz')

render_dir = path.join(path.dirname(args.ckpt),
            'train_renders' if args.train else 'test_renders')
want_metrics = True
if args.render_path:
    assert not args.train
    render_dir += '_path'
    want_metrics = False

# Handle various image transforms
if not args.render_path:
    # Do not crop if not render_path
    args.crop = 1.0
if args.crop != 1.0:
    render_dir += f'_crop{args.crop}'
if args.ray_len:
    render_dir += f'_raylen'
    want_metrics = False

dset = datasets[args.dataset_type](args.data_dir, split="test_train",
                                    **config_util.build_data_options(args))
grid = svox2.SparseGrid.load(args.ckpt, device=device)

# TODO: Check if these things can apply to depth images. (Perhaps we can remove background here and reduce postprocessing efforts)
if grid.use_background:
    if args.nobg:
        #  grid.background_cubemap.data = grid.background_cubemap.data.cuda()
        grid.background_data.data[..., -1] = 0.0
        render_dir += '_nobg'
    if args.nofg:
        grid.density_data.data[:] = 0.0
        #  grid.sh_data.data[..., 0] = 1.0 / svox2.utils.SH_C0
        #  grid.sh_data.data[..., 9] = 1.0 / svox2.utils.SH_C0
        #  grid.sh_data.data[..., 18] = 1.0 / svox2.utils.SH_C0
        render_dir += '_nofg'

    # DEBUG
    #  grid.links.data[grid.links.size(0)//2:] = -1
    #  render_dir += "_chopx2"

config_util.setup_render_opts(grid.opt, args)

if args.blackbg:
    print('Forcing black bg')
    render_dir += '_blackbg'
    grid.opt.background_brightness = 0.0

# TODO: Clean this up
print('Writing to', render_dir)
os.makedirs(render_dir, exist_ok=True )

# TODO: Revise this naming strategy
NPY_PREFIX = "depth_"
depth_output_dir = Path(args.data_dir)/"result/depth_npy"
depth_output_dir.mkdir(exist_ok=True,parents=True)
print("Writing depths to ", str(depth_output_dir))

if not args.no_imsave:
    print('Will write out all frames as PNG (this take most of the time)')

# NOTE: no_grad enables the fast image-level rendering kernel for cuvol backend only
# other backends will manually generate rays per frame (slow)
with torch.no_grad():
    n_images = dset.render_c2w.size(0) if args.render_path else dset.n_images
    img_eval_interval = max(n_images // args.n_eval, 1)
    avg_psnr = 0.0
    avg_ssim = 0.0
    avg_lpips = 0.0
    n_images_gen = 0
    c2ws = dset.render_c2w.to(device=device) if args.render_path else dset.c2w.to(device=device)
    # DEBUGGING
    #  rad = [1.496031746031746, 1.6613756613756614, 1.0]
    #  half_sz = [grid.links.size(0) // 2, grid.links.size(1) // 2]
    #  pad_size_x = int(half_sz[0] - half_sz[0] / 1.496031746031746)
    #  pad_size_y = int(half_sz[1] - half_sz[1] / 1.6613756613756614)
    #  print(pad_size_x, pad_size_y)
    #  grid.links[:pad_size_x] = -1
    #  grid.links[-pad_size_x:] = -1
    #  grid.links[:, :pad_size_y] = -1
    #  grid.links[:, -pad_size_y:] = -1
    #  grid.links[:, :, -8:] = -1

    #  LAYER = -16
    #  grid.links[:, :, :LAYER] = -1
    #  grid.links[:, :, LAYER+1:] = -1

    frames = []
    #  im_gt_all = dset.gt.to(device=device)

    for img_id in tqdm(range(0, n_images, img_eval_interval)):
        dset_h, dset_w = dset.get_image_size(img_id)
        im_size = dset_h * dset_w
        w = dset_w if args.crop == 1.0 else int(dset_w * args.crop)
        h = dset_h if args.crop == 1.0 else int(dset_h * args.crop)

        cam = svox2.Camera(c2ws[img_id],
                           dset.intrins.get('fx', img_id),
                           dset.intrins.get('fy', img_id),
                           dset.intrins.get('cx', img_id) + (w - dset_w) * 0.5,
                           dset.intrins.get('cy', img_id) + (h - dset_h) * 0.5,
                           w, h,
                           ndc_coeffs=dset.ndc_coeffs)
        # im = grid.volume_render_image(cam, use_kernel=True, return_raylen=args.ray_len)
        depth_img = grid.volume_render_depth_image(cam,None)
        # depth_img = viridis_cmap(depth_img.cpu())
        depth_img_np = depth_img.cpu().numpy()
        if args.nofg:
            depth_filename = args.data_dir+"/result/depth_mask/" + NPY_PREFIX + str(img_id)+".npy"
        else:
            depth_filename = args.data_dir+"/result/depth_npy/" + NPY_PREFIX + str(img_id)+".npy"
        np.save(depth_filename, depth_img_np)
        
        print("Saving depths in ", depth_filename )

        

