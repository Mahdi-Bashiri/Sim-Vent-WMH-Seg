# %
"""
[description including article name , developer (Mahdi Bashiri Bawil), date, etc.]

"""

# %
import os
import cv2
import time
import shutil
import skimage
import subprocess
import numpy as np
import nibabel as nib
from skimage.transform import rescale
from scipy.ndimage import gaussian_filter, binary_dilation, label
from skimage.morphology import diamond, binary_opening, binary_closing, erosion, dilation, rectangle, disk, remove_small_objects


# %
def noise_red(n_array_, sigma=1, alpha=1.5):
    out_array = np.zeros_like(n_array_)

    for i in range(n_array_.shape[2]):
        n_array = n_array_[..., i]
        input_img = n_array / np.max(n_array)

        # input_img = np.round((n_array / np.max(n_array)) * 255)

        # image = cv2.fastNlMeansDenoising(input_img.astype(np.uint8), h=10)

        # image = image / 255.0

        # choosing sigma based on the harshness of noisy images from [0.5, 2]
        blurred_img = gaussian_filter(input_img, sigma)

        # Step 2: Apply Unsharp Masking to enhance edges
        mask = input_img - blurred_img
        sharpened_img = input_img + alpha * mask

        # Clip the result to the valid range
        sharpened_img = np.clip(sharpened_img, np.min(input_img), np.max(input_img))

        # restore the maximum
        output_img = sharpened_img * np.max(n_array)

        out_array[..., i] = output_img

    return out_array

# %
def size_check(data_all, v_size, dim=(256, 256)):
    padded_data_all = np.zeros((dim[0], dim[1], data_all.shape[2]))

    for i in range(data_all.shape[2]):

        data_ = data_all[..., i]

        # Assuming 'data_' is the 2D image matrix
        # Here, we define scaling factors for each dimension
        data_shape = data_.shape
        # print(data_.shape)
        scaling_factors = v_size[:2]

        rescaled_data = rescale(data_, scaling_factors, anti_aliasing=True, mode='reflect')

        # The rescaled_image variable now contains the rescaled image matrix wit isometric (1,1,1) voxels
        image_shape = rescaled_data.shape
        # print(image_shape)

        # Define the desired dimensions for padding
        desired_shape = dim  # Specify the specific dimensions you want to reach
        image = np.zeros((dim))

        # Calculate the amount of padding needed for each dimension
        pad_h = desired_shape[0] - image_shape[0]
        if pad_h < 0:
            pad_height = 0
        else:
            pad_height = pad_h

        pad_w = desired_shape[1] - image_shape[1]
        if pad_w < 0:
            pad_width = 0
        else:
            pad_width = pad_w

        # Calculate the padding configuration
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Pad the image symmetrically to reach the desired dimension
        padded_data = np.pad(rescaled_data, ((pad_top, pad_bottom), (pad_left, pad_right)),
                             mode='constant', constant_values=np.min(rescaled_data))

        # Truncate the padded image to fit into desired dim size
        if pad_h < 0:
            image = padded_data[int(-pad_height / 2):desired_shape[0] + int(-pad_height / 2), :]
            padded_data = image
        if pad_w < 0:
            image = padded_data[:, int(-pad_width / 2):desired_shape[1] + int(-pad_width / 2)]
            padded_data = image

        padded_data_all[..., i] = padded_data

    return padded_data_all

# %
def brain_mask_new(data_img):

    mask_img = np.zeros((data_img.shape), dtype=np.uint8)
    pad_width = 28

    brain_centers = np.zeros((1, 2, data_img.shape[2]))
    brain_axes = np.zeros((1, 2, data_img.shape[2]))

    area_e = 0
    for h in range(data_img.shape[2]):

        # Load the grayscale MRI image
        image = 255 * (data_img[..., h] / np.max(data_img[..., h]))
        image = np.pad(image,
                       pad_width=((pad_width, pad_width), (pad_width, pad_width)),
                       mode='constant',
                       constant_values=0)


        # 1. Thresholding the Image
        threshold_value = int(255 / 10)  # min_val + (np.percentile(image, 10) - min_val)
        _, initial_mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

        # Convert mask to boolean for morphological operations
        initial_mask_bool = initial_mask.astype(bool)

        # 2. Morphological Operations (Open/Close) using a diamond-shaped structuring element

        struct_elem = diamond(1)  # Create a diamond-shaped structuring element

        # Apply opening and closing to fill the mask
        opened_mask = binary_opening(initial_mask_bool, struct_elem)
        closed_mask = binary_closing(opened_mask, struct_elem)

        dilated_mask = dilation(closed_mask, diamond(1))

        struct_elem = diamond(4)  # Create a diamond-shaped structuring element

        # Apply opening and closing to fill the mask
        # opened_mask = binary_opening(initial_mask_bool, struct_elem)
        closed_mask = binary_closing(dilated_mask, struct_elem)

        # Convert the processed mask back to uint8
        filled_mask = (closed_mask * 255).astype(np.uint8)

        # # 3. Apply the Eroded Mask to the Original Image to Extract the Skull
        # skull_image = cv2.bitwise_and(image, image, mask=eroded_mask_uint8)

        # 4. Find contours in the mask
        contours, _ = cv2.findContours(filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # print(f'\t\t Contours: {len(contours)}')

            # Find the largest contour (assuming the skull is the largest object in the mask)
            largest_contour = max(contours, key=cv2.contourArea)

            # Fit an ellipse to the largest contour
            if len(largest_contour) >= 5:  # At least 5 points are needed to fit an ellipse
                ellipse = cv2.fitEllipse(largest_contour)

                # Calculate the area of the ellipse
                axes = ellipse[1]
                brain_axes[..., h] = axes
                ellipse_area = np.pi * (axes[0] / 2) * (axes[1] / 2)

                # Calculate the center coordinates
                if ellipse_area > area_e:
                    # update area_e:
                    # area_e = ellipse_area

                    # save the cneters:
                    center_x, center_y = map(int, ellipse[0])
                    brain_centers[0, 0, h] = center_x  - pad_width
                    brain_centers[0, 1, h] = center_y  - pad_width

                # Create a blank image to draw the ellipse
                ellipse_image = np.zeros_like(filled_mask)

                # Draw the ellipse on the blank image
                cv2.ellipse(ellipse_image, ellipse, 255, thickness=-1)  # Filled ellipse with white color

                # 5. Erosion to Shrink the Mask by 10 Pixels
                eroded_mask = erosion(ellipse_image, diamond(10))
                eroded_mask_uint8 = (eroded_mask * 1).astype(np.uint8)

                # 6. Fit again an ellipse to the final mask:
                contours, _ = cv2.findContours(eroded_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                largest_contour = max(contours, key=cv2.contourArea)   
                ellipse = cv2.fitEllipse(largest_contour)
                axes = ellipse[1]
                brain_axes[..., h] = axes
                center_x, center_y = map(int, ellipse[0])
                brain_centers[0, 0, h] = center_x  - pad_width
                brain_centers[0, 1, h] = center_y  - pad_width
                
                # 7. Unpad the obtained eroded mask
                eroded_mask_uint8_unpad = eroded_mask_uint8[pad_width:-pad_width, pad_width:-pad_width]
                mask_img[..., h] = eroded_mask_uint8_unpad

            else:
                print("\t\tNot enough points to fit an ellipse.")
        else:
            print("\t\tNo Contours to fit an ellipse")

    return mask_img, brain_centers, brain_axes

# %
def normalize(data, data_m, a=-1, b=1, fs_m=0, w_='T2W_F', eps=2e-5):
    data[np.isnan(data)] = np.min(data)
    # print(f"min: {np.min(data)},  max: {np.max(data)}")

    # brain mask analyzing:
    for i in range(data.shape[2]):
        if np.sum(data_m[
                      ..., i]) > 2800:  # smallest size of brain tissue interested for us or resonable for assuming that slice as a brain slice.
            data_ = data[..., i]

            """# double checking for tissue existence
            input_bw = np.where(data_ > 0.1*np.min(data_), 1, 0)
            if np.sum(input_bw) < 0.5*np.sum(data_m[..., i]):
                # print('black        ', subject, '    ', v_slc + slc)
                data[..., i] = a
                continue"""

            if fs_m == 0 and w_ != 'T2W_T':
                data_res = data[..., i] * np.where(data_m[..., i] > 0, 0, 1)
            else:
                data_res = data[..., i]

            # for defining the max and min of data in each slice:
            hist, bin_edges = np.histogram(data_res, bins=10, range=(np.min(data_res), np.max(data_res)))
            max_s = np.average(data_res[np.where(data_res > bin_edges[-2])])
            min_s = np.average(data_res[np.where(data_res < bin_edges[1])])
            # final normalization:
            data_[np.where(data_ > max_s)] = max_s
            data_[np.where(data_ < min_s)] = min_s

            # if FS: multiply in a factor: 0.55
            if fs_m == 1:
                data_ = (b - a) * 0.55 * ((data_ - min_s) / (max_s - min_s)) + a

            else:
                data_ = (b - a) * ((data_ - min_s) / (max_s - min_s)) + a

            data[..., i] = data_

        else:
            data[..., i] = a

    # print(f"min: {np.min(data)},  max: {np.max(data)}")
    return data

# %
def normalization(data, data_m, w='T2W_F', fs_mod=0, a=-1, b=1, type='float32'):
    data_n = normalize(data, data_m, a, b, fs_mod, w)

    if type == 'uint8':
        data_n = (data_n * 255).astype(np.uint8)
    elif type == 'uint16':
        data_n = (data_n * 65535).astype(np.uint16)
    elif type == 'float' or type == 'float32':
        data_n = (data_n * 1).astype(np.float32)

    return data_n

# %
def load_nifti(file_path):
    """Load a NIfTI file and return the image data and the nibabel object."""
    img = nib.load(file_path)
    data = img.get_fdata()
    return data, img

# %
def save_nifti(data, ref_img, out_path):
    """Save data as a NIfTI file using a reference image for header/affine."""
    new_img = nib.Nifti1Image(data, affine=ref_img.affine, header=ref_img.header)
    nib.save(new_img, out_path)
    print(f"Saved pre-processed data to {out_path}")

# %
def z_score_normalization(data_array):
    z_data_array = np.zeros_like(data_array)
    for k in range(data_array.shape[-1]):
        img_data = data_array[..., k]
        z_img_data = (img_data - np.mean(img_data)) / (np.std(img_data) + 1e-7)
        z_data_array[..., k] = z_img_data
    return z_data_array

# %
def fsl_bet(input_path, fractional_intensity=0.5, generate_mask=False):
    """
    Perform brain extraction using FSL's BET tool.
    
    Parameters:
    -----------
    input_path : str
        Path to input brain image (.nii.gz)
    fractional_intensity : float, default=0.5
        Fractional intensity threshold (0->1); smaller values give larger brain outline
    generate_mask : bool, default=False
        Generate binary brain mask in addition to brain-extracted image
    
    Returns:
    --------
    str : Path to the brain-extracted output image
    """
        
    # Output path for the brain-extracted image
    output_image = input_path.replace('.nii.gz', '_brain.nii.gz')
    
    # Base BET command
    bet_command = [
        '/home/sai/fsl/bin/bet',
        input_path,
        output_image.replace('.nii.gz', ''),  # BET adds .nii.gz automatically
        '-f', str(fractional_intensity)
    ]
    
    # Add mask generation if requested
    if generate_mask:
        bet_command.append('-m')
    
    try:
        subprocess.run(bet_command, check=True)
        print(f"\nBET brain extraction completed successfully.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during BET brain extraction: {e}")
    except FileNotFoundError:
        print("BET command not found. Make sure FSL is installed and in your PATH.")
    
    return output_image

# %
def extract_and_save_images(query_save_dir,
                            query_gt_dir,
                            q_file
                            ):
    query_gt_file_path = os.path.join(query_gt_dir, q_file)
    query_data, query_obj = load_nifti(query_gt_file_path)
    for slc in range(query_data.shape[-1]):
        query_data[..., slc] = np.where(query_data[..., slc] > (0.5*np.max(query_data[..., slc])), 1, 0)
        img_save_path = os.path.join(query_save_dir, f"{q_file[:6]}_{slc+1}.png")
        skimage.io.imsave(img_save_path, np.uint8(255 * query_data[..., slc]))

    save_nifti(query_data, query_obj, query_gt_file_path)

    return query_data.astype(np.uint8), query_obj

# %
def fitter(main_image, secondary_images, brain_mask, brain_center, brain_axes, output_shape=(256, 256)):

    new_width = np.max(brain_axes)
    new_height = np.max(brain_axes)

    half_width = np.round(new_width // 2).astype(np.uint16)
    half_height = np.round(new_height // 2).astype(np.uint16)

    # Calculate cropping box coordinates
    top = int(max(brain_center[1] - half_height, 0))
    bottom = int(min(brain_center[1] + half_height, main_image.shape[0]))
    left = int(max(brain_center[0] - half_width, 0))
    right = int(min(brain_center[0] + half_width, main_image.shape[1]))

    # Crop the image using slicing
    cropped_image = main_image[top:bottom, left:right]
    
    # Calculate zoom factor
    zoom_factor_y = output_shape[0] / cropped_image.shape[0]
    zoom_factor_x = output_shape[1] / cropped_image.shape[1]
    zoom_factor = (zoom_factor_y, zoom_factor_x)  # (y, x) format

    # Resize the cropped image to the desired output shape (256x256)
    resized_main_image = np.float64(
        # np.round(skimage.transform.resize(cropped_image / 1.0, output_shape, anti_aliasing=True) * 1.0))
        skimage.transform.resize(cropped_image / 1.0, output_shape, anti_aliasing=True) * 1.0)

    # Crop the secondary images using slicing
    fit_secondary_images = []
    for second_image in secondary_images:
        cropped_image2 = second_image[top:bottom, left:right]
        # Resize the cropped image2 to the desired output shape (256x256)
        resized_image2 = np.float64(
            # np.round(skimage.transform.resize(cropped_image2 / 1.0, output_shape, anti_aliasing=True) * 1.0))
            skimage.transform.resize(cropped_image2 / 1.0, output_shape, anti_aliasing=True) * 1.0)
        # Since the secondary files are binary masks:
        bw_resized_image2 = np.where(resized_image2 > 0.5, 1, 0).astype(np.uint8)
        fit_secondary_images.append(bw_resized_image2)

    return resized_main_image, fit_secondary_images, zoom_factor

# %
def preprocess(
        COHORT_DIR,
        DATASET_NAME,
        imaging_o='not_FS', 
        imaging_w='T2W_F', 
        desired_dim=256
        ):

    # Preparing sub-directories:
    flair_dir = os.path.join(COHORT_DIR, DATASET_NAME, 'FLAIR', 'Raw - main')
    gt_dir = os.path.join(COHORT_DIR, DATASET_NAME, 'GroundTruth')
    vent_gt_dir = os.path.join(gt_dir, 'files', 'Vent_Masks')
    abwmh_gt_dir = os.path.join(gt_dir, 'files', 'abWMH_Masks')
    nwmh_gt_dir = os.path.join(gt_dir, 'files', 'nWMH_Masks')
    brain_gt_dir = os.path.join(gt_dir, 'files', 'Brain_Masks')

    SAVE_DIR = {
    'FLAIR File Save': os.path.join(COHORT_DIR, DATASET_NAME, 'FLAIR', 'Preprocessed', 'files'),
    'FLAIR Image Save': os.path.join(COHORT_DIR, DATASET_NAME, 'FLAIR', 'Preprocessed', 'images'),
    'GT VENT Save': os.path.join(gt_dir, 'images', 'Vent_Masks'),
    'GT abWMH Save': os.path.join(gt_dir, 'images', 'abWMH_Masks'),
    'GT nWMH Save': os.path.join(gt_dir, 'images', 'nWMH_Masks'),
    'Brain Mask Save': os.path.join(gt_dir, 'images', 'Brain_Masks')
    }

    for name, save_directory in SAVE_DIR.items():
        os.makedirs(save_directory, exist_ok=True)

    ZOOMED_SAVE_DIR = {
    'Zoomed FLAIR File Save': os.path.join(COHORT_DIR, DATASET_NAME, 'FLAIR', 'Preprocessed', 'zoomed', 'files'),
    'Zoomed FLAIR Image Save': os.path.join(COHORT_DIR, DATASET_NAME, 'FLAIR', 'Preprocessed', 'zoomed', 'images'),
    'Zoomed GT VENT Save': os.path.join(gt_dir, 'zoomed', 'images', 'Vent_Masks'),
    'Zoomed GT abWMH Save': os.path.join(gt_dir, 'zoomed', 'images', 'abWMH_Masks'),
    'Zoomed GT nWMH Save': os.path.join(gt_dir, 'zoomed', 'images', 'nWMH_Masks'),
    'Zoomed Brain Mask Save': os.path.join(gt_dir, 'zoomed', 'images', 'Brain_Masks')
    }

    for name, save_directory in ZOOMED_SAVE_DIR.items():
        os.makedirs(save_directory, exist_ok=True)

    # Constants
    # find out Fat Sat mode:
    if imaging_o[:2] == 'FS':
        fs_mode = 1
    else:
        fs_mode = 0
        
    files = [f for f in os.listdir(flair_dir) if f.endswith('.nii.gz')]

    block = False
    for file in files:

        if block and (file[:6] != '114585' and file[:6] != '101228' and file[:6] != '101627' and file[:6] != '115788'):
            continue
        else:
            block = False

        # ##  Pre-processing

        # % STEP 0
        # Loading FLAIR data
        flair_file_path = os.path.join(flair_dir, file)
        flair_data, flair_obj = load_nifti(flair_file_path)
        if DATASET_NAME=='Public_MSSEG':
            for cc in range(flair_data.shape[-1]):
                flair_data[..., cc] = (flair_data[..., cc] - np.min(flair_data[..., cc])) / (np.max(flair_data[..., cc]) - np.min(flair_data[..., cc]))

        voxel_size = np.array(flair_obj.header['pixdim'][1:4])
        file = file.replace('_FLAIR.nii.gz', '.nii.gz')

        # % STEP 1
        # primary noise reduction:
        nifti_data_nr = noise_red(flair_data)

        # % STEP 2
        # convert to meet 256*256 array shape:
        voxel_size[-1] = 1
        # nifti_data_s = size_check(nifti_data_nr, voxel_size, dim=(desired_dim, desired_dim))
        nifti_data_s = np.copy(nifti_data_nr) # to be removed 

        # % STEP 3
        # Simple Elliptical Brain Extraction:
        masks_data, brain_cnt, brain_ax = brain_mask_new(nifti_data_s)
        masks_data = np.where(masks_data < 128, 0, 1).astype(np.uint8)

        # BET Brain Extractor:
        nii_tempo_save_path = os.path.join(SAVE_DIR['Brain Mask Save'], file)
        save_nifti(flair_data, flair_obj, nii_tempo_save_path)

        brain_image_nii_path = fsl_bet(nii_tempo_save_path, fractional_intensity=0.45, generate_mask=True)
        bet_brain_mask_path = brain_image_nii_path.replace('.nii.gz', '_mask.nii.gz')
        masks_data_bet = nib.load(bet_brain_mask_path).get_fdata()
        os.remove(nii_tempo_save_path)
        # bet_brain_mask_path = os.path.join(brain_gt_dir, file)
        masks_data_bet = nib.load(bet_brain_mask_path).get_fdata()

        masks_data_2 = (masks_data * masks_data_bet).astype(np.uint8)

        # % STEP 4
        # image normalization:
        nifti_data_n = normalization(nifti_data_s, masks_data, imaging_w, fs_mode, a=0, b=1, type='float')
        # nifti_data_n = np.copy(flair_data) # to be removed 

        # % STEP 5
        # z-score image normalization:
        nifti_data_z = z_score_normalization(nifti_data_n)

        # Save the preprocessed data. You might want to copy the header/affine from the original data.
        falir_out_path = os.path.join(SAVE_DIR['FLAIR File Save'], file)
        save_nifti(nifti_data_z, flair_obj, falir_out_path)

        for slc in range(nifti_data_z.shape[-1]):
            img_save_path = os.path.join(SAVE_DIR['FLAIR Image Save'], f"{file[:6]}_{slc+1}.npy")
            np.save(img_save_path, nifti_data_z[..., slc].astype(np.float32))

        for slc in range(nifti_data_n.shape[-1]):
            img_save_path = os.path.join(SAVE_DIR['FLAIR Image Save'], f"{file[:6]}_{slc+1}.png")
            skimage.io.imsave(img_save_path, np.uint16(65535.0 * nifti_data_n[..., slc]))

        # Save brain info
        np.savez(falir_out_path.replace('.nii.gz', '_binfo.npz'), brain_cnt=brain_cnt, brain_ax=brain_ax, brain_mask=masks_data)

        # % STEP 6
        # Ground Truth Mask Process     
        # for 3 manually segmented masks:
        vent_data, vent_obj = extract_and_save_images(SAVE_DIR['GT VENT Save'], vent_gt_dir, file)
        abwmh_data, abwmh_obj = extract_and_save_images(SAVE_DIR['GT abWMH Save'], abwmh_gt_dir, file)
        nwmh_data, nwmh_obj = extract_and_save_images(SAVE_DIR['GT nWMH Save'], nwmh_gt_dir, file)

        for slc in range(masks_data_2.shape[-1]):
            img_save_path = os.path.join(SAVE_DIR['Brain Mask Save'], f"{file[:6]}_{slc+1}.png")
            skimage.io.imsave(img_save_path, 255 * masks_data_2[..., slc])

        # % STEP 7
        # Brain Zooming, Our Innovative Preprocessing Step
        fit_flair_array = np.zeros_like(nifti_data_n)
        fit_vent_array = np.zeros_like(vent_data)
        fit_abwmh_array = np.zeros_like(abwmh_data)
        fit_nwmh_array = np.zeros_like(nwmh_data)
        fit_brain_array = np.zeros_like(masks_data)

        zooming_factors = []

        for i in range(nifti_data_n.shape[-1]):

            flair_slice = np.nan_to_num(nifti_data_n[..., i])

            vent_slice = vent_data[..., i]
            abwmh_slice = abwmh_data[..., i]
            nwmh_slice = nwmh_data[..., i]

            brain_slice = masks_data_2[..., i]
            brain_cnt_slice = brain_cnt[..., i][0]
            brain_ax_slice = brain_ax[..., i][0]

            if np.sum(brain_slice) < (3.14 * 30 * 30):
                print(file, '\tlow seen brain in FLAIR slice')
                # store the results
                fit_flair_array[..., i] = 0
                fit_vent_array[..., i] = 0
                fit_abwmh_array[..., i] = 0
                fit_nwmh_array[..., i] = 0
                fit_brain_array[..., i] = 0
                zooming_factors.append((1, 1))
                continue  # to avoid almost empty FLAIR slices

            # Perform the zooming function
            secondary_slice_images = [vent_slice, abwmh_slice, nwmh_slice, brain_slice]
            fit_flair, fit_secondary_images, zooming_factor = fitter(flair_slice, 
                                                     secondary_slice_images, 
                                                     brain_slice, brain_cnt_slice, brain_ax_slice)
            
            # store the results
            fit_flair_array[..., i] = fit_flair
            fit_vent_array[..., i] = fit_secondary_images[0]
            fit_abwmh_array[..., i] = fit_secondary_images[1]
            fit_nwmh_array[..., i] = fit_secondary_images[2]
            fit_brain_array[..., i] = fit_secondary_images[3]

            zooming_factors.append(zooming_factor)
            
        fit_flair_array_z = z_score_normalization(fit_flair_array)

        # save the zoomed files:
        zoomed_falir_out_path = os.path.join(ZOOMED_SAVE_DIR['Zoomed FLAIR File Save'], file)
        save_nifti(fit_flair_array_z, flair_obj, zoomed_falir_out_path)
        
        for slc in range(fit_flair_array_z.shape[-1]):
            img_save_path = os.path.join(ZOOMED_SAVE_DIR['Zoomed FLAIR Image Save'], f"{file[:6]}_{slc+1}.npy")
            np.save(img_save_path, fit_flair_array_z[..., slc].astype(np.float32))
        
        for slc in range(fit_flair_array.shape[-1]):
            img_save_path = os.path.join(ZOOMED_SAVE_DIR['Zoomed FLAIR Image Save'], f"{file[:6]}_{slc+1}.png")
            skimage.io.imsave(img_save_path, np.uint16(65535 * fit_flair_array[..., slc]))

        for slc in range(fit_vent_array.shape[-1]):
            img_save_path = os.path.join(ZOOMED_SAVE_DIR['Zoomed GT VENT Save'], f"{file[:6]}_{slc+1}.png")
            skimage.io.imsave(img_save_path, 255 * fit_vent_array[..., slc])
        
        for slc in range(fit_abwmh_array.shape[-1]):
            img_save_path = os.path.join(ZOOMED_SAVE_DIR['Zoomed GT abWMH Save'], f"{file[:6]}_{slc+1}.png")
            skimage.io.imsave(img_save_path, 255 * fit_abwmh_array[..., slc])
        
        for slc in range(fit_nwmh_array.shape[-1]):
            img_save_path = os.path.join(ZOOMED_SAVE_DIR['Zoomed GT nWMH Save'], f"{file[:6]}_{slc+1}.png")
            skimage.io.imsave(img_save_path, 255 * fit_nwmh_array[..., slc])
        
        for slc in range(fit_brain_array.shape[-1]):
            img_save_path = os.path.join(ZOOMED_SAVE_DIR['Zoomed Brain Mask Save'], f"{file[:6]}_{slc+1}.png")
            skimage.io.imsave(img_save_path, 255 * fit_brain_array[..., slc])

        zooming_factors_file_path = os.path.join(ZOOMED_SAVE_DIR['Zoomed FLAIR Image Save'], f"{file[:6]}_zooming_factors.npy")
        np.save(zooming_factors_file_path, zooming_factors)


if __name__ == '__main__':

    COHORT_DIR = '/mnt/e/MBashiri/ours_articles/Paper#2/Data/Cohort'
    DATASET_NAME = 'Local_SAI' # choose from 'Local_SAI' or 'Public_MSSEG'

    imaging_o = 'not_FS' if DATASET_NAME=='Local_SAI' else 'FS'
    
    preprocess(
        COHORT_DIR,
        DATASET_NAME,
        imaging_o=imaging_o, 
        imaging_w='T2W_F', 
        desired_dim=256
        )
