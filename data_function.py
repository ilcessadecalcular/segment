from glob import glob
from os.path import dirname, join, basename, isfile
import sys
sys.path.append('./')
import csv
import torch
from medpy.io import load
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
import random
import torchio as tio
from torchio import AFFINE, DATA
import torchio
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
from torchio.data import UniformSampler
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)
from pathlib import Path
import os
import SimpleITK as sitk
from hparam import hparams as hp
import pydicom
import cv2 as cv
from torch.nn import functional as F
from scipy import ndimage


class MedData_train(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir):
        patch_size = hp.patch_size

        queue_length = 5
        samples_per_volume = 5


        self.subjects = []


        images_dir = Path(images_dir)
        self.image_paths = sorted(images_dir.glob(hp.fold_arch))
        # 读取文件名
        labels_dir = Path(labels_dir)
        self.label_paths = sorted(labels_dir.glob(hp.fold_arch))

        for (image_path, label_path) in zip(self.image_paths, self.label_paths):
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path),
            )
            self.subjects.append(subject)


        self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)


        # self.queue_dataset = Queue(
        #     self.training_set,
        #     queue_length,
        #     samples_per_volume,
        #     UniformSampler(patch_size),
        # )

    def __len__(self):
        return len(self.image_paths)



    def transform(self):

        if hp.aug:
            training_transform = Compose([
             ToCanonical(),
            CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
             RandomMotion(),
            RandomBiasField(),
            ZNormalization(),
            RandomNoise(),
            RandomFlip(axes=(0,)),
            OneOf({
                RandomAffine(): 0.8,
                RandomElasticDeformation(): 0.2,
            }),
            ])
        else:
            training_transform = Compose([
            # CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
            ZNormalization(),
            ])


        return training_transform


class MedData_eval(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir):
        self.subjects = []
        images_dir = Path(images_dir)
        self.image_paths = sorted(images_dir.glob(hp.fold_arch))
        labels_dir = Path(labels_dir)
        self.label_paths = sorted(labels_dir.glob(hp.fold_arch))

        for (image_path, label_path) in zip(self.image_paths, self.label_paths):
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label=tio.LabelMap(label_path),
            )
            self.subjects.append(subject)

        self.transforms = self.transform()

        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)

    def __len__(self):
        return len(self.image_paths)

    def transform(self):
        training_transform = Compose([
        # CropOrPad((hp.crop_or_pad_size), padding_mode='reflect'),
        ZNormalization()
        ])
        return training_transform


class MedData_test(torch.utils.data.Dataset):
    def __init__(self,data_path):
        self.data_path = data_path
        self.data = os.listdir(data_path)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        patient = self.data[index]
        patient_path = os.path.join(self.data_path,patient)
        # get image detail
        img, spacing, slope, intercept, origin, direction = self.read_file(patient_path)
        resample_spacing = np.array(spacing, dtype=np.float32)
        resample_spacing = resample_spacing[::-1]
        # patient, _ = patient.split('.')
        # patient = patient.lower()
        # # resample image array
        img = self.resample(img, resample_spacing)
        _, h, w = img.shape
        # # locate max zone
        x, y = self.img_locate(img, slope, intercept)
        x = int(np.clip(x, 128, w - 128))
        y = int(np.clip(x, 128, h - 128))
        img_input = torch.tensor(self.crop(img, x, y))
        img_input = img_input.unsqueeze(0)
        mn = img_input.mean()
        sd = img_input.std()
        znorm_img_input = (img_input - mn) / sd

        return patient,znorm_img_input,spacing,origin,direction,x,y,h,w

    def read_file(self,path):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom_names)
        image2 = reader.Execute()
        # get image detail
        # image_array = sitk.GetArrayFromImage(image2)  # t, h, w
        origin = image2.GetOrigin()  # x, y, z
        spacing = image2.GetSpacing()  # x, y, z
        direction = image2.GetDirection()  # x, y, z

        # print(resample_spacing)
        slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        img = np.array([slices[i].pixel_array for i in range(len(slices))])
        # in each example, the intercept, slope and spacing should be the same, get one
        intercept = int(slices[0][0x0028, 0x1052].value)
        slope = int(slices[0][0x0028, 0x1053].value)
        # thick = np.array(slices[0][0x0018, 0x0050].value, dtype=np.float32)
        # pixel_spacing = np.array(slices[0][0x0028, 0x0030].value, dtype=np.float32)
        # spacing = np.append(thick, pixel_spacing)
        # print(n_spacing)
        # img_array = img * slope + intercept
        # print((img_array == image_array))

        # print((resample_spacing==n_spacing))
        return img, spacing, slope, intercept, origin, direction

    def HU_process(self,img, slope, intercept):
        # transfer pixels to HUs
        # and set those not 200 < HU < 500 to 0
        # return pixels then
        pic_HU = img * slope + intercept
        pic_HU = np.where(pic_HU > 500, 0, pic_HU)
        pic_HU = np.where(pic_HU < 200, 0, pic_HU)
        pic_pixel = ((pic_HU - intercept) / slope).astype(np.uint16)
        return pic_pixel

    def single_locate(self,img):
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

    def img_locate(self,img, slope, intercept):
        b, h, w = img.shape
        coords = np.zeros((b, 2))
        # write down each center coord, a half
        for i in range(b):
            pic_pixel = self.HU_process(img[i], slope, intercept)
            coords[i] = self.single_locate(pic_pixel)
            # print(coords[i])
        x, y = np.mean(coords[b // 2:b], axis=0)
        return x, y

    def resample(self,img, spacing, new_spacing=[1, 0.7, 0.7]):
        resize_factor = spacing / new_spacing
        new_real_shape = img.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / img.shape
        img = ndimage.interpolation.zoom(img, real_resize_factor, mode='nearest')
        return img

    def crop(self,img, x, y, MASK_LENGTH=128):
        _, h, w = img.shape
        # in case of beyond the boundary
        mask = np.zeros(img.shape)
        mask[:, y - MASK_LENGTH: y + MASK_LENGTH, x - MASK_LENGTH: x + MASK_LENGTH] = 1
        # make a mask
        img_np = (img * mask)[:, y - MASK_LENGTH: y + MASK_LENGTH, x - MASK_LENGTH: x + MASK_LENGTH]
        return img_np




