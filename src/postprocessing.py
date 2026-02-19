# %% [markdown]
# PhD team 
# Presenting Code for PhD Thesis

# %%        info [markdown]
# Here, we are presenting a completed routine, a thorough connected blocks, to perform the very idea of my Ph.D. thesis, conducting a comprehensively automatic and longitudinal analyses of a given MS patient.
# All of the rights of this routine are reserved for the developer(s).
# 
# 
# Mahdi Bashiri Bawil
# Developer

# %% [markdown]
# 

# %% [markdown]
# # Attempt : Comprehensive Analyses 

# %% [markdown]
# %% [markdown]
# ## Phase 0: Dependencies & Functions

# %% [markdown]
# ### Packages

# %%        Packages
import os
import re
import sys
import cv2
import csv
import g4f
import json
import glob
import time
import ctypes
import PyPDF2
import pickle
import plotly
import shutil
import hashlib
import zipfile
import imageio
import skimage
import logging
import requests
import warnings
import tempfile
import anthropic
import threading
import webbrowser
import subprocess
import langdetect
import unicodedata
import numpy as np
import pandas as pd
import pydicom as dc
import nibabel as nib
# import tkinter as tk
from PIL import Image
from fpdf import FPDF
import tensorflow as tf
from scipy import stats
import plotly.io as pio
from pathlib import Path
from scipy import ndimage
# from tkinter import font
# import ttkbootstrap as ttk
import plotly.express as px
from datetime import datetime
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import plotly.graph_objects as pxf
from skimage.transform import rescale
from typing import Dict, Any, Optional
from numpy.f2py.auxfuncs import errmess
# from tkinter import filedialog, messagebox
from skimage.color import gray2rgb, rgb2gray
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, send_file
from scipy.ndimage import gaussian_filter, binary_dilation, label
from skimage.morphology import diamond, binary_opening, binary_closing, erosion, dilation, rectangle, disk, remove_small_objects

# Call the in-house Functions
from general_utility_functions import *
from specific_utility_functions import *
from preprocessing_functions import *
from model_inferences_and_masks_production import *
from mask_processing_and_feature_extraction import *
from feature_plotting_functions import *
from qualitative_presentation import *
from timepoint_analysis import *
from prompt_generation import *
from LLM_communication import *
from AI_report_production import *
from unet_model import build_unet_3class



# %%                        Default Configurations for Visualizations
# color_codes should be defined globally or passed as parameter
font_family = "Arial"
title_font_size = 18
axis_font_size = 18
legend_font_size = 18
annotation_font_size = 20

# Use the same color scheme and font settings as the count function
"""    
color_codes = {
    "peri": "rgb(228, 40, 31)",
    "peri_hex": "#e4281f",

    "para": "rgb(252, 125, 48)",
    "para_hex": "#fc7d30",

    "juxt": "rgb(255, 247, 59)",
    "juxt_hex": "#fff73b",

    "total": "rgb(47, 72, 88)",      # Dark blue-gray for annotations
    "total": "#2F4858",

    "jvent": "rgb(0, 201, 35)",
    "jvent_hex": "#00c923",

    "vent": "rgb(33, 98, 254)",
    "vent_hex": "#2162fe",

    "csf": "rgb(33, 190, 254)",
    "csf_hex": "#21bffe",

    "darkgray": "rgb(50, 50, 50)",
    "darkgray_hex": "#323232",

    "navyblue": "rgb(20, 30, 80)",
    "navyblue_hex": "#142050",

    "softbeige": "rgb(240, 240, 230)",
    "softbeige_hex": "#f0f0e6",

    "ivorywhite": "rgb(255, 255, 245)",
    "ivorywhite_hex": "#fffff5",

    "white": "rgb(255, 255, 255)",
    
    "tot": "rgb(33, 47, 60)",
    "tot_hex": "#212f3c",

}
"""


# %%        Main Process
def main_process(query_path, nID, temp_root_save, pix2pix_generator_4l, pix2pix_generator_gm):
    """Placeholder for the main process logic."""

    try:
        # %
        # Paths
        global temp_root
        temp_root = temp_root_save
        print(temp_root)
        os.makedirs(temp_root, exist_ok=True)

        # % [markdown]
        # ## Phase 1-2: Preparing Data

        # % [markdown]
        # ### FLAIR Processing

        # %
        # Assume we have the all needed input data as query_path, query_nID, query_pIDs
        global voxel_size

        # # Find out the valid PID(s)
        # if len(user_patient_ids) < 6:
        #     pIDs = None
        # else:
        #     pIDs = user_patient_ids.split(',')
        #     pID_list = []
        #     for p in pIDs:
        #         if int(p) < 100000:
        #             pass
        #         else:
        #             pID_list.append(str(int(p)))
        #     if len(pID_list) == 0:
        #         raise Warning('\nThe entered PID(s) are not valid, so the program will proceed with no previous PID(s).\n')

        # Proceed loading the data
        files = os.listdir(query_path)
        files_type = [file.split('.')[-1] for file in files]

        if len(files) > 1:

            # Check for DICOM data possibility

            if all(x == 'dcm' for x in files_type):
                # hence, the provided query data is a set of dicom images
                # judge for imaging weight, plane and fat sat mode:
                np_data, axial, plane, fs_mode, img_w, img_siz, voxel_size, n_slices, standard, pid, age, sex, acq_date, acq_time, pname = slices_checker(query_path, windowing=True)
                np_data = np_data[..., ::-1]

                if axial == 0:
                    # the imaging plane is not Axial and then not suitable for this workflow:
                    # Print a user-friendly error message
                    status_message, error_message = error_window("Error: The input data must be an axial scan. \nPlease provide a valid input and try again.")
                    return (status_message, error_message, None)

                # # If the condition is met, proceed
                # print("Condition satisfied. Continuing execution...")

                if img_w[:9] == 'T2W_FLAIR':
                    img_w = 'AX FLAIR'

                if img_w != 'AX FLAIR':
                    # the imaging weight is not T2-FLAIR and then not suitable for this workflow:
                    # Print a user-friendly error message
                    status_message, error_message = error_window("Error: The input data must be an Axial T2-FLAIR scan. \nPlease provide a valid input and try again.")
                    return (status_message, error_message, None)

                # # If the condition is met, proceed
                # print("Condition satisfied. Continuing execution...")

            else:
                # the provided folder contains not entirely and exclusively DICOM data or a mere NIFTI file:
                # Print a user-friendly error message
                status_message, error_message = error_window("Error: The input data must contain entirely and exclusively DICOM files. \nPlease provide a valid input and try again.")
                return (status_message, error_message, None)

        elif len(files) == 1:

            # Check for NIFTI data possibility
            # print(files_type)

            if files_type[0] == 'nii' or files_type[0] == 'gz':    
                # load data:
                data_path = os.path.join(query_path, files[0])
                nifti_data = nib.load(data_path)
                # print(nifti_data)
                voxel_size = nifti_data.header['pixdim'][1:4]      
                # print(voxel_size)      

                # get nifiti data:
                nifti_data = nifti_data.get_fdata()
                np_data = np.rot90(nifti_data)                   # to have the scans/slices oriented as HFS

                # Load brain masks
                data_path_brain = os.path.join(os.path.dirname(query_path), 'Brain_Masks', files[0])
                np_data_brain = nib.load(data_path_brain).get_fdata()
                np_data_brain = np.rot90(np.where(np_data_brain > 0, 1, 0).astype(np.uint8))
                
                # np_data = np_data[..., -40:]
                np_data_temp = np.zeros((256, 256, 60))
                np_data_temp_b = np.zeros((256, 256, 60))
                half_bottom = int((np_data_temp.shape[-1] - np_data.shape[-1]) // 2)
                np_data_temp[..., half_bottom:half_bottom+np_data.shape[-1]] = np_data
                np_data_temp_b[..., half_bottom:half_bottom+np_data_brain.shape[-1]] = np_data_brain

                np_data_ = []
                np_data_b = []
                for il in np.linspace(0, 60, 21):
                    if il == 60:
                        continue
                    # print(il)
                    np_data_.append(np_data_temp[..., int(il)])
                    np_data_b.append(np_data_temp_b[..., int(il)])
                np_data = np.stack(np_data_, axis=-1)
                np_data_brain = np.stack(np_data_b, axis=-1)

                # contants:
                fs_mode = 1
                img_w = 'AX FLAIR'
                voxel_size = np.array([1, 1, 6])
                n_slices = np_data.shape[-1]
                pid = files[0].split('.')[0]
                age = 50
                sex = 0
                acq_date = 20250101
                acq_time = 102030
                pname = 'Classified'

            else:
                # the provided folder contains not entirely and exclusively DICOM data or a mere NIFTI file:
                # Print a user-friendly error message
                print("Error: The input data must contain entirely and exclusively DICOM files, or a mere NIFTI file.")
                print("Please provide a valid input and try again.")
                
                # Exit the program with an error code
                sys.exit(1)  # Exit code 1 indicates an error


        else:
            # the provided folder contains no file(s):
            # Print a user-friendly error message
            print("Please provide a valid input and try again.")
            status_message, error_message = error_window("Error: There is no detectable file(s). The input data must contain entirely and exclusively DICOM files.")
            return (status_message, error_message, None)

        # Make unique analysis title name:

        # First, get current date and time
        current_datetime = datetime.now()

        # Format date as 'yymmdd'
        formatted_date = current_datetime.strftime("%y%m%d")

        # Format time as 'hhmmss'
        formatted_time = current_datetime.strftime("%H%M%S")

        unique_name = nID + '_' + pid[:6] + '_' + formatted_date + formatted_time

        # Display acquired information so far:
        print(f"Patinet's Name: {pname}")
        print(f"          Age: {age}")
        print(f"          Sex: {'Female' if sex == 0 else 'Male'}")
        print(f"          Test ID: {pid[:6]}")
        print(f"          National ID: {nID}")
        # print(f"          Previous Test ID(s): {[p for p in pID_list]}")

        print(f"\nImaging Protocol: {img_w}")
        print(f"        Date: {acq_date}")
        print(f"        Time: {acq_time}")
        print(f"        Slices: {n_slices}")
        print(f"        Fat Sat: {'No' if fs_mode==0 else 'Yes'}")
        print(f"        Voxel Size: {np.round(voxel_size[0], 2)} \u00D7 {np.round(voxel_size[1], 2)} \u00D7 {np.round(voxel_size[2], 2)}")

        print(f'\n Unique Name: {unique_name}')

        # all_ids = pID_list.append(pid[:6])

        # Final outputs will be saved in a unique folder:
        u_folder = os.path.join(temp_root, unique_name)
        os.makedirs(u_folder, exist_ok=True)

        np.save(os.path.join(u_folder, 'np_data.npy'), np_data)
        np.save(os.path.join(u_folder, 'voxel_size.npy'), voxel_size)
        np.save(os.path.join(u_folder, 'fs_mode.npy'), fs_mode)
        np.save(os.path.join(u_folder, 'img_w.npy'), 'T2W_F')
        # np.save(os.path.join(u_folder, 'all_ids.npy'), all_ids)
        np.save(os.path.join(u_folder, 'unique_name.npy'), unique_name)

        # %
        # display_nii(np_data)

        # % [markdown]
        # ## Phase 2-1: Inputs

        # % [markdown]
        # ### Load prepared data and information

        # %
        # %
        #
        # u_folder = find_most_recent_folder(temp_root)

        # np_data = np.load(os.path.join(u_folder, 'np_data.npy'))
        # voxel_size = np.load(os.path.join(u_folder, 'voxel_size.npy'))
        # fs_mode = np.load(os.path.join(u_folder, 'fs_mode.npy'))
        img_w = np.load(os.path.join(u_folder, 'img_w.npy'))

        # % [markdown]
        # ## Phase 2-2: Pre-processing

        # % [markdown]
        # ### Loading data and labels

        # %
        # get nifiti data:
        nifti_data = np.copy(np_data)

        # Constants:
        desired_dim = 256

        # primary noise reduction:
        nifti_data_nr = noise_red(nifti_data)

        # convert to meet 256*256 array shape:
        voxel_size[-1] = 1
        nifti_data_s = size_check(nifti_data_nr, voxel_size, dim=(desired_dim, desired_dim))

        # brain extraction:
        masks_data, brain_cnt, brain_ax = brain_mask_new(nifti_data_s)
        masks_data = np.where(masks_data < 128, 0, 1).astype(np.uint8)

        # BET Brain Extractor:
        if len(files) == 1:
            nii_save_path = os.path.join(query_path, files[0])
            masks_data_bet = np_data_brain.copy()
            # dilate the previous precise brain mask
            for ik in range(masks_data_bet.shape[-1]):
                masks_data_bet[..., ik] = (binary_dilation(masks_data_bet[..., ik] > 0, disk(3))) * 1

        else:
            nii_save_path = os.path.join(u_folder, 'data.nii.gz')
            dicom_to_nifti(query_path, nii_save_path)

            brain_image_nii_path = fsl_bet(nii_save_path, fractional_intensity=0.45, generate_mask=True)
            masks_data_bet = nib.load(brain_image_nii_path.replace('.nii.gz', '_mask.nii.gz')).get_fdata()

        masks_data = (masks_data * masks_data_bet).astype(np.uint8)

        # align images to be near:

        # image normalization:
        nifti_data_n = normalization(nifti_data_s, masks_data, img_w, fs_mode, a=0, b=1, type='uint16')
        # display_nii(np_data)

        # z-score image normalization:
        nifti_data_z = z_score_normalization((nifti_data_n).astype(np.float32))

        # image alignment:

        # generate CSF-candidate masks:
        csf_masks = generate_adaptive_dark_masks(nifti_data_n) * masks_data

        # save preprocessed data:

        np.save(os.path.join(u_folder, 'normalized_data.npy'), nifti_data_n)
        np.save(os.path.join(u_folder, 'csf_candidate_masks.npy'), csf_masks)
        np.savez(os.path.join(u_folder, 'brain_info.npz'), brain_mask=masks_data, brain_cnt=brain_cnt, brain_ax=brain_ax)


        # % [markdown]
        # ## Phase 3-1: Inputs

        # %
        # Load preprocessed data

        # u_folder = find_most_recent_folder(temp_root)

        normalized_data = np.copy(nifti_data_n)       # np.load(os.path.join(u_folder, 'normalized_data.npy'))
        # brain_info = np.load(os.path.join(u_folder, 'brain_info.npz'))

        brain_mask = masks_data     # brain_info['brain_mask']
        brain_cnts = brain_cnt      # brain_info['brain_cnt']
        brain_axes = brain_ax       # brain_info['brain_ax']

        # % [markdown]
        # ## Phase 3-2: Preparing Inputs

        # %
        # Generate the suitable images for fetching the trained models

        # Filter by brain mask
        nifti_data_n_bet = np.copy(nifti_data_n)
        nifti_data_z_bet = np.copy(nifti_data_z)

        for i_slc in range(nifti_data_z.shape[-1]):

            # Binarize (any non-zero value becomes 1)
            brain_mask_bool = brain_mask[..., i_slc] > 0

            # Brain extraction
            # print("\n\t Doing the BET")
            nifti_data_z_slc  = nifti_data_z_bet[..., i_slc]
            nifti_data_z_slc[~brain_mask_bool] = np.min(nifti_data_z_slc)
            nifti_data_z_bet[..., i_slc] = nifti_data_z_slc

            # nifti_data_n_slc  = nifti_data_n_bet[..., i_slc]
            # nifti_data_n_slc[~brain_mask_bool] = np.min(nifti_data_n_slc)
            # nifti_data_n_bet[..., i_slc] = nifti_data_n_slc

        to_infr_data = np.zeros_like(nifti_data_z_bet)
        to_infr_data_sp = np.zeros_like(nifti_data_n_bet)

        # Go in loop
        for i in range(3, nifti_data_z_bet.shape[-1]):

            if np.sum(brain_mask[..., i]) < (3.14 * 40 * 40):
                print(f'Small Seen Brain in Slice: {i+1} / {nifti_data_z_bet.shape[-1]}')
                continue                                # to avoid tiny extracted brain from the FLAIR images

            # Save the fit image
            to_infr_data[..., i] = nifti_data_z_bet[..., i]
            to_infr_data_sp[..., i] = nifti_data_n_bet[..., i]

        # np.save(os.path.join(u_folder, 'to_infr_data.npy'), to_infr_data)


        # % [markdown]
        # ## Phase 3-3: Segmentations

        # % [markdown]
        # ### Load Data

        # %
        # Load and Convert the data to suitable input
        to_infr_data_sp = (2.0 * (to_infr_data_sp / 65535.0) - 1.0).astype(np.float32)
        input_tensors_sp = load_image_stack(to_infr_data_sp)
        to_infr_data = np.rot90(to_infr_data, -1, axes=(0, 1))
        input_tensors = load_image_stack(to_infr_data)


        # % [markdown]
        # ### Predict Specilized GM Masks

        # Generate the output images
        sp_gm_masks = generate_images(pix2pix_generator_gm, input_tensors_sp, training_status=True)

        # Reshape the array to (256, 256, 20)
        sp_gm_masks = np.squeeze(sp_gm_masks, axis=-1)          # Remove the singleton dimension (axis=-1)
        sp_gm_masks = np.transpose(sp_gm_masks, (1, 2, 0))      # Transpose to (256, 256, 20)

        # Normalize back to suit the uint16 format
        sp_gm_masks = np.round(((sp_gm_masks * 0.5) + 0.5) * 65535.0).astype(np.uint16)


        # % [markdown]
        # ### Predict 4L Masks, including ventricle and two types of WMH

        # Generate the output images
        vent_wmh_masks_softmax = generate_images(pix2pix_generator_4l, input_tensors, training_status=False)

        # Reshape the array to (256, 256, 20)
        vent_wmh_masks = np.argmax(vent_wmh_masks_softmax, axis=-1)
        vent_wmh_masks = np.transpose(vent_wmh_masks, (1, 2, 0))      # Transpose to (256, 256, 20)
        vent_wmh_masks = np.rot90(vent_wmh_masks, 1, axes=(0, 1))


        # % [markdown]
        # ## Phase 3-4: Post-processing

        # % [markdown]
        # ### Sp. GM Mask Post-processing

        # %
        # Morphologically post-processing the sp. GM masks to obtain more dedicated and accurate masks.

        sp_gm = sp_gm_masks / 65535.0
        sp_gm[sp_gm<0.5] = 0
        sp_gm = sp_gm.astype(bool)

        for i in range(sp_gm.shape[-1]):
            sp_gm1 = sp_gm[..., i]
            sp_gm1 = remove_small_objects(sp_gm1, min_size=5)
            sp_gm2 = binary_closing(sp_gm1, disk(1))
            sp_gm2 = remove_small_objects(sp_gm2, min_size=20)
            sp_gm[..., i] = sp_gm1 & sp_gm2


        # %
        np.save(os.path.join(u_folder, 'sp_GM_masks.npy'), sp_gm)
        
        # Save the specific slices:
        """
        n_slc_ = 13    
        output_path_sp = os.path.join(u_folder, f'sp_gm_slice{n_slc_}.png')
        skimage.io.imsave(output_path_sp, np.uint8(sp_gm[35:-35, 35:-35, n_slc_-1] * 255))
        """

        # % [markdown]
        # ### 4L Mask Post-processing

        # %
        # Extract four distinct classes named background, ventricles, juxtaventircle WMH, and other WMH.

        discrete_labels = np.copy(vent_wmh_masks)

        rgb_label = labels_to_rgb(discrete_labels)


        # %
        # Morphologically post-processing the three masks.

        # for ventricle masks:

        vent_m = np.where(discrete_labels ==1, 1, 0)
        vent_m = vent_m.astype(bool)

        for i in range(vent_m.shape[-1]):
            vent_m1 = vent_m[..., i]
            vent_m1 = remove_small_objects(vent_m1, min_size=5)
            vent_m1 = binary_closing(vent_m1, disk(1))
            vent_m1 = binary_opening(vent_m1, disk(1))
            # vent_m = remove_small_objects(vent_m, min_size=20)
            vent_m[..., i] = vent_m1

        vent_m_int = (vent_m *255).astype(np.uint16)

        rgb_label[..., 2] = vent_m_int

        # for juxtaventricle WMH masks:

        v_wmh = np.where(discrete_labels ==2, 1, 0)
        v_wmh = v_wmh.astype(bool)

        for i in range(v_wmh.shape[-1]):
            v_wmh1 = v_wmh[..., i]

            # filtering by the approximity to the ventricle masks
            vent_m1 = vent_m[..., i]
            vent_m1 = dilation(vent_m1, disk(3))
            v_wmh1 = v_wmh1 & vent_m1

            v_wmh1 = remove_small_objects(v_wmh1, min_size=5)
            # v_wmh1 = binary_closing(v_wmh1, disk(1))
            # v_wmh1 = binary_opening(v_wmh1, disk(1))
            # v_wmh1 = remove_small_objects(v_wmh, min_size=20)
            v_wmh[..., i] = v_wmh1

        v_wmh_int = (v_wmh *255).astype(np.uint16)

        rgb_label[..., 1] = v_wmh_int

        # update the vent masks:
        vent_m = vent_m & ~v_wmh
        vent_m_int = (vent_m *255).astype(np.uint16)

        rgb_label[..., 2] = vent_m_int

        # for WMH masks:

        wmh = np.where(discrete_labels ==3, 1, 0)
        wmh = wmh.astype(bool)
        wmh_main = wmh.copy()

        for i in range(wmh.shape[-1]):
            wmh1 = wmh[..., i]

            # filtering by the approximity to the ventricle and juxtavntricle WMH masks
            vent_m1 = vent_m[..., i]
            v_wmh1 = v_wmh[..., i]
            vent_m1_ = dilation(vent_m1, disk(1))
            wmh1_main = wmh1 & ~vent_m1_
            wmh1_main = wmh1_main & ~v_wmh1
            vent_m1 = dilation(vent_m1, disk(3))
            wmh1 = wmh1 & ~vent_m1
            wmh1 = wmh1 & ~v_wmh1

            wmh1 = remove_small_objects(wmh1, min_size=5)
            wmh1 = binary_closing(wmh1, disk(1))
            # wmh1 = binary_opening(wmh1, disk(1))
            # wmh1 = remove_small_objects(wmh1, min_size=20)
            wmh[..., i] = wmh1

            wmh1_main = remove_small_objects(wmh1_main, min_size=5)
            wmh1_main = binary_closing(wmh1_main, disk(1))
            wmh_main[..., i] = wmh1_main

        wmh_int = (wmh *255).astype(np.uint16)

        rgb_label[..., 0] = wmh_int

        # Save the specific slices:
        """        
        n_slc_ = 13    
        output_path_sp = os.path.join(u_folder, f'4l_vent_slice{n_slc_}.png')
        skimage.io.imsave(output_path_sp, np.uint8(vent_m_int[35:-35, 35:-35, n_slc_-1]))

        output_path_sp = os.path.join(u_folder, f'4l_nwmh_slice{n_slc_}.png')
        skimage.io.imsave(output_path_sp, np.uint8(v_wmh_int[35:-35, 35:-35, n_slc_-1]))

        output_path_sp = os.path.join(u_folder, f'4l_awmh_slice{n_slc_}.png')
        skimage.io.imsave(output_path_sp, np.uint8(wmh_int[35:-35, 35:-35, n_slc_-1]))
        """

        # Post-process the jv_wmh for its potential of being part of the wmh masks
        
        # wmh: abnormal WMH mask
        # v_wmh: normal/juxtaventricular WMH mask  
        # ventricle_masks: ventricle mask
        # voxel_size: tuple of voxel dimensions in mm

        min_area_pixels = max(1, int(10 / (voxel_size[0] * voxel_size[1])))  # ~10 mm²

        wmh, v_wmh = process_wmh_masks_with_ventricles(
            wmh, v_wmh, vent_m, voxel_size, min_area_pixels
        )

        wmh |= wmh_main
        wmh = wmh & ~v_wmh

        # # previous method of context-wise lesion checking
        # candidate_jvwmh = []
        # for u in range(wmh.shape[-1]):
        #     candidate_jvwmh.append(adjacency_finder_v1(wmh[..., u], v_wmh[..., u], min_area=(10 / (voxel_size[0] * voxel_size[1])).round() ))       # aiming for an area of 10 mm2
        # candidate_jvwmh = np.array(candidate_jvwmh).transpose((1, 2, 0))
        # v_wmh = v_wmh & ~candidate_jvwmh
        # wmh = wmh | candidate_jvwmh


        # %
        np.save(os.path.join(u_folder, 'vent_masks.npy'), vent_m)
        np.save(os.path.join(u_folder, 'jv_wmh_masks.npy'), v_wmh)
        np.save(os.path.join(u_folder, 'wmh_masks.npy'), wmh)

        # %
        # display_nii(rgb_label)


        status_message = completion_window()
        return (status_message, None)

    except Exception as e:
        status_message = "Something Went Wrong!"
        error_message = str(e) + " Please, ensure of following instructions."
        logging.error(f"Main process error: {error_message}")
        return (status_message, error_message, [])  # Return empty list instead of undefined variable



def get_default_history_directory():
    """Get the default directory for saving history."""
    try:
        exe_dir = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    except NameError:
        exe_dir = os.getcwd()
    history_dir = os.path.join(exe_dir, "MRI_MS_Analyzer_History")
    os.makedirs(history_dir, exist_ok=True)
    return history_dir


if __name__ == "__main__":
    list_files = ['0101', '0102', '0701', '0702', '0801', '0802']
    for list_file in list_files:

        # Get data from the query parameters
        national_id = f'002016{list_file}' # request.args.get('patient_id')
        folder_name = '0007-S_F/0100-FLAIR'
        folder_name = 'Raw'
        folder_name = list_file
        directory_path = os.path.join('/home/sai/challenge/web2/UPLOAD_FOLDER', folder_name) # request.args.get('directory_path')
        output_dir = get_default_history_directory()

        logging.debug(f"Received parameters: national_id={national_id}, directory_path={directory_path}")

        # pre-loading pix2pix models to accelerate the main process
        try:
            # Load the paths
            model_GM_path = get_resource_path('models/pix2pix_generator_sp_GM')

            # Load the saved model
            pix2pix_generator_gm = load_model(model_GM_path)
            print("✅ Sp. GM Segmentation Model loaded successfully\n")

            # Load the paths
            # model_4L_path = get_resource_path('models/pix2pix_generator_4L_ex')
            model_4L_path = get_resource_path('models/best_dice_generator.h5')

            # Load the saved model
            # pix2pix_generator_4l = load_model(model_4L_path)

            # Build model architecture first
            pix2pix_generator_4l = build_unet_3class(
                input_shape=(256, 256, 1), 
                num_classes=4
            )
            
            # Load weights
            pix2pix_generator_4l.load_weights(model_4L_path)

            print("✅ 4L Segmentation Model loaded successfully\n")

        except Exception as e:
            pix2pix_generator_gm = None
            pix2pix_generator_4l = None
        
        # Trigger the main processing function
        status_msg, error_msg = main_process(
            directory_path, national_id, output_dir,
            pix2pix_generator_4l=pix2pix_generator_4l,
            pix2pix_generator_gm=pix2pix_generator_gm
        )

        # main_process()
