import cv2 as cv
import numpy as np
from PIL import Image

mask_filename = "/workspace/datasets/cctv/source/masks/img_0000.jpg"

mask = Image.open(mask_filename)
mask = np.array(mask)
dilatation_size = 20
dilate_element = cv.getStructuringElement(cv.MORPH_RECT, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))

dilated_mask = cv.dilate(mask, dilate_element)

pili = Image.fromarray(dilated_mask, mode="L")
pili.save("/workspace/garbage/mask_test.jpg")
