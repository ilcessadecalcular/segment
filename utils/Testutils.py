import os
import pydicom
import numpy as np
from scipy import ndimage
import cv2 as cv
import nibabel as nib
import argparse
from tqdm import tqdm
import nibabel as nib
import matplotlib.pyplot as plt
import torchio as io
import gdcm
import pylibjpeg

# half of the mask length
MASK_LENGTH = 128


def HU_process(img, slope, intercept):
    # transfer pixels to HUs
    # and set those not 200 < HU < 500 to 0
    # return pixels then
    pic_HU = img * slope + intercept
    pic_HU = np.where(pic_HU > 500, 0, pic_HU)
    pic_HU = np.where(pic_HU < 200, 0, pic_HU)
    pic_pixel = ((pic_HU - intercept) / slope).astype(np.uint16)
    return pic_pixel


def locate(img):
    # set a center
    img = img.astype(np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_OPEN, np.ones((5, 5)))
    # do the opening morpholohy, get an img with some connected domains
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # find the contour of each connected domins
    rectangle = [0, 0, 0, 0, 0]
    for i in contours:
        # find the maximal connected domins by compare their external square size
        x, y, w, h = cv.boundingRect(i)
        if rectangle[0] < w * h:
            rectangle = [w * h, x, y, w, h]
        else:
            continue
    # x, y is the top-left point, compare the center by plus half the length
    x = int(rectangle[1] + rectangle[3] // 2)
    y = int(rectangle[2] + rectangle[4] // 2)
    return (x, y)


def read_file(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    img = np.array([slices[i].pixel_array for i in range(len(slices))])
    # in each example, the intercept, slope and spacing should be the same, get one
    intercept = int(slices[0][0x0028, 0x1052].value)
    slope = int(slices[0][0x0028, 0x1053].value)
    thick = np.array(slices[0][0x0018, 0x0050].value, dtype=np.float32)
    pixel_spacing = np.array(slices[0][0x0028, 0x0030].value, dtype=np.float32)
    spacing = np.append(thick, pixel_spacing)
    return img, intercept, slope, spacing


def resample(img, spacing, new_spacing=[1, 0.7, 0.7]):
    resize_factor = spacing / new_spacing
    new_real_shape = img.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / img.shape
    img = ndimage.interpolation.zoom(img, real_resize_factor, mode='nearest')
    return img


def process(read_data_path, read_label_path, save_data_path, save_label_path):
    for patient in tqdm(os.listdir(read_label_path)):
        # use label as a standard, For the filename has both upper and lower letters
        label = nib.load(os.path.join(read_label_path, patient))
        affine = label.affine.copy()
        hdr = label.header.copy()
        patient, _ = patient.split('.')
        # we dont want the extension, drop it and transfer the filename to all lower letters
        patient = patient.lower()
        # we process the label half and stop, for we need some parameters provided by the data
        # it's awful, but... ok
        img, intercept, slope, spacing = read_file(os.path.join(read_data_path, patient))
        img = resample(img, spacing)
        # seems resampling is necessary
        b, h, w = img.shape
        coords = np.zeros((b, 2))
        # write down each center coord, a half
        for i in range(b):
            pic_pixel = HU_process(img[i], slope, intercept)
            coords[i] = locate(pic_pixel)
        x, y = np.mean(coords[b // 2:b], axis=0)
        print(x, y)
        x = int(np.clip(x, 128, w - 128))
        y = int(np.clip(x, 128, h - 128))
        # in case of beyond the boundary
        mask = np.zeros(img.shape)
        mask[:, y - MASK_LENGTH: y + MASK_LENGTH, x - MASK_LENGTH: x + MASK_LENGTH] = 1
        # make a mask
        img_np = (img * mask)[:, y - MASK_LENGTH: y + MASK_LENGTH, x - MASK_LENGTH: x + MASK_LENGTH]
        # plt.imshow(img_np[100], cmap=plt.cm.gray)
        # plt.show()

        img_image = nib.Nifti1Image(img_np, affine, hdr)

        # try torchio method
        # img_np = np.expand_dims(img_np, axis=0)
        #
        # source_image = io.ScalarImage(tensor=img_np, affine=affine)
        # source_image.save(os.path.join(save_data_path, patient + '.nii.gz'))

        nib.save(img_image, os.path.join(save_data_path, patient) + '.nii.gz')

        # np.save(os.path.join(save_data_path, patient) + '.npy', img_np)

        label = np.array(label.dataobj).transpose(2, 1, 0)
        # nii gives the shape of (w, h, b), permute it
        label = resample(label, spacing)
        label_np = (label * mask)[:, y - MASK_LENGTH: y + MASK_LENGTH, x - MASK_LENGTH: x + MASK_LENGTH]
        label_image = nib.Nifti1Image(label_np, affine, hdr)
        nib.save(label_image, os.path.join(save_label_path, patient) + '.nii.gz')
        # np.save(os.path.join(save_label_path, patient) + '_label.npy', label_np)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='path')
    parser.add_argument('--mode', default='train', help='type train or test')
    parser.add_argument('--save_path', default='data_try2', help='path to save files')

    args = parser.parse_args()
    mode = args.mode
    assert (mode == 'train' or mode == 'test'), 'mode should be train or test'

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    save_data_path = os.path.join(save_path, mode)
    save_label_path = os.path.join(save_path, mode + '_label')
    os.makedirs(save_data_path, exist_ok=True)
    os.makedirs(save_label_path, exist_ok=True)

    if mode == 'train':
        data_path = './data/train'
        label_path = './data/label'
    else:
        data_path = './test'
        label_path = './test_label'

    process(data_path, label_path, save_data_path, save_label_path)