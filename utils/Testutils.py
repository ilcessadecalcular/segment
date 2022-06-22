import os
import pydicom
import numpy as np
from scipy import ndimage
import cv2 as cv
import nibabel as nib
import argparse
from tqdm import tqdm
import nibabel as nib


# half of the mask length



def p_resample(img, spacing, new_spacing):
    resize_factor = spacing / new_spacing
    new_real_shape = img.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / img.shape
    img = ndimage.interpolation.zoom(img.cpu(), real_resize_factor, mode='nearest')
    return img