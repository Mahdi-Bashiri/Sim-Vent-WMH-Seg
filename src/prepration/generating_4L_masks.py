

# %% Packages
import os
import cv2
import shutil
import random
import skimage
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.morphology import diamond, binary_opening, binary_closing, erosion, dilation, rectangle


# %% Functions
def load_nifti(file_path):
    """Load a NIfTI file and return the image data and the nibabel object."""
    img = nib.load(file_path)
    data = img.get_fdata()
    return data, img


def save_nifti(data, ref_img, out_path):
    """Save data as a NIfTI file using a reference image for header/affine."""
    new_img = nib.Nifti1Image(data, affine=ref_img.affine, header=ref_img.header)
    nib.save(new_img, out_path)
    print(f"Saved pre-processed data to {out_path}")


def mapping_4l(min, max):

    """
    (0.0*(max-min) + min):background ,
    (0.25*(max-min) + min):vent ,
    (0.75*(max-min) + min):n_wmh ,
    (1.0*(max-min) + min):ab_wmh
    """
    return [(0.0*(max-min) + min), (0.25*(max-min) + min), (0.75*(max-min) + min), (1.0*(max-min) + min)]


def nii_display(data1, data2, data3, data4,
                data1_title='Ventricles', data2_title='Normal WMH',
                data3_title='Abnormal WMH', data4_title='4-level Mask'):
    for i in range(10, 18):  # num_slices):
        im1 = data1[:, :, i]
        im2 = data2[:, :, i]
        im3 = data3[:, :, i]
        im4 = data4[:, :, i]

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(im1, cmap='gray')
        axes[0].set_title(data1_title)
        axes[1].imshow(im2, cmap='gray')
        axes[1].set_title(data2_title)
        axes[2].imshow(im3, cmap='gray')
        axes[2].set_title(data3_title)
        axes[3].imshow(im4, cmap='gray')
        axes[3].set_title(data4_title)

        for ax in axes:
            ax.axis('off')

        plt.suptitle(f'Slice {i + 1}')
        plt.pause(1)  # Pause to observe each slice
        plt.show(block=False)
        # plt.close()


def nii_display_1(data1, data2, data3, data4,
                data1_title='Ventricles', data2_title='Normal WMH',
                data3_title='Abnormal WMH', data4_title='4-level Mask'):
    im1 = data1
    im2 = data2
    im3 = data3
    im4 = data4

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(im1, cmap='gray')
    axes[0].set_title(data1_title)
    axes[1].imshow(im2, cmap='gray')
    axes[1].set_title(data2_title)
    axes[2].imshow(im3, cmap='gray')
    axes[2].set_title(data3_title)
    axes[3].imshow(im4, cmap='gray')
    axes[3].set_title(data4_title)

    for ax in axes:
        ax.axis('off')

    plt.suptitle(f'Slice {i + 1}')
    plt.pause(1)  # Pause to observe each slice
    plt.show(block=False)
    # plt.close()


def brain_mask_2(data_img):
    # data_img = np.mean(data_img, axis=-1)

    mask_img = np.zeros((data_img.shape), dtype=np.uint8)
    pad_width = 28

    brain_centers = np.zeros((2))
    brain_axes = np.zeros((2))

    area_e = 0

    # Load the grayscale MRI image
    image = 255 * (data_img / np.max(data_img))
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
            brain_axes = axes
            ellipse_area = np.pi * (axes[0] / 2) * (axes[1] / 2)

            # Calculate the center coordinates
            if ellipse_area > area_e:
                # update area_e:
                area_e = ellipse_area

                # save the cneters:
                center_x, center_y = map(int, ellipse[0])
                brain_centers[0] = center_x - pad_width
                brain_centers[1] = center_y - pad_width

            # Create a blank image to draw the ellipse
            ellipse_image = np.zeros_like(filled_mask)

            # Draw the ellipse on the blank image
            cv2.ellipse(ellipse_image, ellipse, 255, thickness=-1)  # Filled ellipse with white color

            # 5. Erosion to Shrink the Mask by 10 Pixels
            eroded_mask = erosion(ellipse_image, diamond(20))
            eroded_mask_uint8 = (eroded_mask * 1).astype(np.uint8)

            # 7. Unpad the obtained eroded mask
            eroded_mask_uint8_unpad = eroded_mask_uint8[pad_width:-pad_width, pad_width:-pad_width]
            mask_img[...] = eroded_mask_uint8_unpad

        else:
            print("\t\tNot enough points to fit an ellipse.")
    else:
        print("\t\tNo Contours to fit an ellipse")

    return mask_img, brain_centers, brain_axes


def fitter(image, second_image, output_shape=(256, 256)):
    im_mask, center, axs = brain_mask_2(image)

    new_width = np.max(axs)
    new_height = np.max(axs)

    half_width = np.round(new_width // 2).astype(np.uint16)
    half_height = np.round(new_height // 2).astype(np.uint16)

    # Calculate cropping box coordinates
    top = int(max(center[1] - half_height, 0))
    bottom = int(min(center[1] + half_height, image.shape[0]))
    left = int(max(center[0] - half_width, 0))
    right = int(min(center[0] + half_width, image.shape[1]))

    # Crop the image using slicing
    cropped_image = image[top:bottom, left:right]
    # Resize the cropped image to the desired output shape (256x256)
    resized_image = np.uint16(
        np.round(skimage.transform.resize(cropped_image / 65535.0, output_shape, anti_aliasing=True) * 65535.0))

    # Crop the second image using slicing
    cropped_image2 = second_image[top:bottom, left:right]
    # Resize the cropped image2 to the desired output shape (256x256)
    resized_image2 = np.uint16(
        np.round(skimage.transform.resize(cropped_image2 / 65535.0, output_shape, anti_aliasing=True) * 65535.0))

    return resized_image, resized_image2, im_mask


# %% Define Directories

# Directory of Pre-processed FLAIR Data
flair_dir = r'C:/Users/SAI/Desktop/Desktop/Original_FLAIRs_prep'
# Directory of Manually Segmented Abnormal WMH Data
man_seg_wmh_dir = r'C:/Users/SAI/Desktop/Desktop/abWMH_manual_segmentations'
# Directory of Automatically/Manually Segmented Normal WMH Data
man_seg_normal_wmh_dir = r'C:/Users/SAI/Desktop/Desktop/nWMH_manual_segmentations'
# Directory of Manually Segmented Ventricles Data
man_seg_vent_dir = r'C:/Users/SAI/Desktop/Desktop/vent_manual_segmentations'
# Directory of Generated 4-level Masks Data
save_path = r'C:/Users/SAI/Desktop/Desktop/manual_4L_masks_april'
os.makedirs(save_path, exist_ok=True)
png_save_path_1 = r'C:/Users/SAI/Desktop/Desktop/manual_4L_masks_new_png_nonZero'
png_save_path_2 = r'C:/Users/SAI/Desktop/Desktop/manual_4L_masks_new_png_Zero'
os.makedirs(png_save_path_1, exist_ok=True)
os.makedirs(png_save_path_2, exist_ok=True)


# %% Make a 4L Mask and Save it

files = os.listdir(man_seg_wmh_dir)

for file in files:

    #
    # ## Loading data and masks

    flair_file_path = os.path.join(flair_dir, file)
    flair_data, flair_obj = load_nifti(flair_file_path)

    ab_wmh_file_path = os.path.join(man_seg_wmh_dir, file)
    ab_wmh_data, ab_wmh_obj = load_nifti(ab_wmh_file_path)
    ab_wmh_data = np.where(ab_wmh_data > 0, 1, 0).astype(bool)

    n_wmh_file_path = os.path.join(man_seg_normal_wmh_dir, file)
    n_wmh_data, n_wmh_obj = load_nifti(n_wmh_file_path)
    n_wmh_data = np.where(n_wmh_data > 0, 1, 0).astype(bool)


    vent_file_path = os.path.join(man_seg_vent_dir, file)
    vent_data, vent_obj = load_nifti(vent_file_path)
    vent_data = np.where(vent_data > 0, 1, 0).astype(bool)

    # ## Excluding WMH Masks and Ventricles Mask from nWMH Masks & WMH from Ventricles

    n_wmh_data = (n_wmh_data & ~ab_wmh_data) & ~vent_data
    vent_data = vent_data & ~ab_wmh_data

    # ## Creating a 4-level Mask

    # mask_points = np.round(mapping_4l(min=-1, max=1), 1)
    mask_points = np.uint16(mapping_4l(min=0, max=65535))

    mask_4l = vent_data * mask_points[1] + n_wmh_data * mask_points[2] + ab_wmh_data * mask_points[3]

    # ## Show and Save Results
    # nii_display(vent_data*65535, n_wmh_data*65535, ab_wmh_data*65535, mask_4l)

    # Save the created 4L mask. You might want to copy the header/affine from the original data.
    save_nifti(mask_4l, vent_obj, os.path.join(save_path, file))
    # # Save the updated masks.
    # save_nifti(vent_data, vent_obj, os.path.join(man_seg_vent_dir, file))
    # save_nifti(n_wmh_data, n_wmh_obj, os.path.join(man_seg_normal_wmh_dir, file))


# %% Save PNG Images from 4L Masks

files = os.listdir(save_path)

for file in files:

    # Generate the paired images

    data_flair, obj_flair = load_nifti(os.path.join(flair_dir, file))
    data_4L, obj_4L = load_nifti(os.path.join(save_path, file))

    # nii_display(data_flair, data_4L, data_flair, data_4L)
    # Go in loop
    for i in range(data_flair.shape[-1]):

        im_fl = np.nan_to_num(data_flair[..., i]) * 65535.0            # for float data range between 0 and 1
        im_4l = data_4L[..., i]

        if (np.sum(im_fl) / 65535.0) < (3.14 * 20 * 20):
            print(file, 'low flair')
            continue  # to avoid almost empty FLAIR slices

        # Perform the fit function
        fit_flair, fit_4l, b_mask = fitter(im_fl, im_4l)

        # nii_display_1(im_fl, im_4l, fit_flair, fit_4l)

        if np.sum(b_mask) < (3.14 * 100 * 100):
            print(file, 'low masks')
            continue  # to avoid tiny extracted brain from the FLAIR images

        # Save the paired images
        merged_image = np.concatenate((np.rot90(fit_flair, 1), np.rot90(fit_4l, 1)), axis=1).astype(np.uint16)

        if np.sum(fit_4l) == 0:  # to separate the empty 4L masks from the rest
            dst_path = os.path.join(png_save_path_2, file.replace('.nii.gz', f'_{i + 1}.png'))
        else:
            dst_path = os.path.join(png_save_path_1, file.replace('.nii.gz', f'_{i + 1}.png'))

        skimage.io.imsave(dst_path, merged_image)


# %% Separate the paired images to test and train sets based on 70% & 30% random selection.

# Define paths
folder_name = 'dataset_4l_man_april'
train_folder = os.path.join(os.path.dirname(save_path) , folder_name, 'train')
test_folder = os.path.join(os.path.dirname(save_path) , folder_name, 'test')

# Create train and test folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Get list of unique image IDs (assuming first 6 characters represent an ID)
ims = os.listdir(png_save_path_1)
ids = list(set(file[:6] for file in ims))

# Shuffle and split 70% train, 30% test
random.shuffle(ids)
split_idx = int(len(ids) * 0.7)
train_ids, test_ids = ids[:split_idx], ids[split_idx:]

# you may disable the following line while using whole dataset, not the provided sample
test_ids = ['104252', '105074', '105465', '105755', '105911'
            ]

# Move images to respective folders
for file in ims:
    img_id = file[:6]
    src_path = os.path.join(png_save_path_1, file)

    if img_id not in test_ids:
        dest_path = os.path.join(train_folder, file)
    else:
        dest_path = os.path.join(test_folder, file)

    shutil.copyfile(src_path, dest_path)
print("Images successfully separated into train and test sets.")
