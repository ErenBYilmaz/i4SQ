# -*- coding: utf-8 -*-

import json
import os.path

import SimpleITK
import numpy
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt


def main():
    img_path = r'C:\Temp\DemoCase1_spine_cpr.nii.gz'
    load_and_plot(img_path)


def load_and_plot(img_path):
    png_path = img_path.replace('.nii.gz', '_overlay.png')
    if not img_path.endswith('.nii.gz'):
        raise ValueError(img_path)
    if img_path.endswith('_seg.nii.gz'):
        raise ValueError(img_path)
    if os.path.isfile(png_path):
        return
    segmentation_path = img_path.replace('.nii.gz', '_seg.nii.gz')
    json_path = img_path.replace('.nii.gz', '.json')
    # load straightened CPR (curved planar reformation) of image and segmentation
    cpr_img = sitk.ReadImage(img_path)
    cpr_segmentation = sitk.ReadImage(segmentation_path)
    # load landmarks in CPR space
    with open(json_path) as fid:
        cpr_landmarks = json.load(fid)

    # go to numpy
    cpr_img_data = SimpleITK.GetArrayFromImage(cpr_img)
    cpr_segmentation_data = SimpleITK.GetArrayFromImage(cpr_segmentation)

    plot_and_save_image_with_segmentation_overlay_and_landmarks(cpr_img_data, cpr_landmarks, cpr_segmentation_data, to_file=png_path)


def plot_and_show_image_with_segmentation_overlay_and_landmarks(cpr_img, cpr_landmarks, cpr_segmentation):
    plt.figure(figsize=(2.0, 1.0 * (cpr_img.shape[0] / (cpr_img.shape[1]))))
    plot_image_with_segmentation_overlay_and_landmarks(cpr_img, cpr_segmentation, cpr_landmarks)
    plt.show()


def plot_and_save_image_with_segmentation_overlay_and_landmarks(cpr_img, cpr_landmarks, cpr_segmentation, to_file):
    plt.figure(figsize=(2.0, 1.0 * (cpr_img.shape[0] / (cpr_img.shape[1]))))
    plot_image_with_segmentation_overlay_and_landmarks(cpr_img, cpr_segmentation, cpr_landmarks)
    plt.savefig(to_file)
    plt.close()


def plot_image_with_segmentation_overlay_and_landmarks(cpr_img_data: numpy.ndarray, cpr_segmentation_data: numpy.ndarray, cpr_landmarks):
    # set plotting limits
    vmin = 500
    vmax = 1500

    # plot overlay
    plt.axes([0, 0, 0.5, 1])
    s = cpr_img_data[::-1, :, int(cpr_img_data.shape[2] / 2)]
    plt.imshow(s, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    s = cpr_segmentation_data[::-1, :, int(cpr_img_data.shape[2] / 2)]
    plt.imshow(np.ma.masked_where((s < 0) | (s == 0), s, copy=True), cmap='jet', alpha=0.5, interpolation='nearest', aspect='auto')
    plt.axis('off')

    # plot landmarks
    plt.axes([0.5, 0, 0.5, 1])
    s = cpr_img_data[::-1, :, int(cpr_img_data.shape[2] / 2)]
    plt.imshow(s, cmap='gray', aspect='auto', vmin=vmin, vmax=vmax)
    for key in cpr_landmarks:
        plt.text(50, cpr_img_data.shape[0] - cpr_landmarks[key][0]['pos'][2], key, verticalalignment='center', color='k')
    plt.axis('off')


if __name__ == '__main__':
    main()
