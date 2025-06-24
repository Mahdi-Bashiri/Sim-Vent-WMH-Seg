

# %%
import os
import cv2
import shutil
import skimage
import numpy as np
import nibabel as nib
from skimage.transform import rescale
from scipy.ndimage import gaussian_filter, binary_dilation, label
from skimage.morphology import diamond, binary_opening, binary_closing, erosion, dilation, rectangle, disk, remove_small_objects


# %%
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


# %%
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


# %%
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
                eroded_mask = erosion(ellipse_image, diamond(20))
                eroded_mask_uint8 = (eroded_mask * 1).astype(np.uint8)

                # 6.

                # 7. Unpad the obtained eroded mask
                eroded_mask_uint8_unpad = eroded_mask_uint8[pad_width:-pad_width, pad_width:-pad_width]
                mask_img[..., h] = eroded_mask_uint8_unpad

            else:
                print("\t\tNot enough points to fit an ellipse.")
        else:
            print("\t\tNo Contours to fit an ellipse")

    return mask_img, brain_centers, brain_axes


# %%
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


# %%
def normalization(data, data_m, w='T2W_F', fs_mod=0, a=-1, b=1, type='float32'):
    data_n = normalize(data, data_m, a, b, fs_mod, w)

    if type == 'uint8':
        data_n = (data_n * 255).astype(np.uint8)
    elif type == 'uint16':
        data_n = (data_n * 65535).astype(np.uint16)
    elif type == 'float' or type == 'float32':
        data_n = (data_n * 1).astype(np.float32)

    return data_n


# %%
def load_nifti(file_path):
    """Load a NIfTI file and return the image data and the nibabel object."""
    img = nib.load(file_path)
    data = img.get_fdata()
    return data, img


# %%
def save_nifti(data, ref_img, out_path):
    """Save data as a NIfTI file using a reference image for header/affine."""
    new_img = nib.Nifti1Image(data, affine=ref_img.affine, header=ref_img.header)
    nib.save(new_img, out_path)
    print(f"Saved pre-processed data to {out_path}")


# %%  Direcorties
Database_dir = r"E:\MBashiri\Thesis\local_dataset\SAI\Demyelinated"     # main database directory of the raw acquired MRI data

save_path = r'C:/Users/SAI/Desktop/Desktop/Original_FLAIRs_prep'
os.makedirs(save_path, exist_ok=True)

# parsing onto the local database and
# listing out the flair files:
flair_file_paths_list = []          # the total path list of raw FLAIR data found from the main database directory
case_folders = os.listdir(Database_dir)
for case_folder in case_folders:
    raw_dir = os.path.join(Database_dir, case_folder, 'Raw')
    files = os.listdir(raw_dir)
    se_number_holder = 100000
    for file in files:
        if 'AXFLAIR' in file:
            se_number = int(file.split('.')[0].split('_')[-1][2:])
            # Choose only the first SE numbers if there are more FLAIR data acquired for a patient
            if se_number < se_number_holder:
                flair_file_paths_list.append(os.path.join(raw_dir, file))
                se_number_holder = se_number

# %%  Constants


# %%  performing the preprocessing 
for file in flair_file_paths_list:

    #
    # ## Loading data

    flair_file_path = file
    pid = os.path.basename(file).split('_')[0]
    if int(pid) < 141302:
        continue
    out_path = os.path.join(save_path, f"{pid}.nii.gz")

    flair_data, flair_obj = load_nifti(flair_file_path)

    # get more slice information:
    imaging_o = 'not_FS' # flair_obj.ScanOptions            # Since, we are sure about the non-Fat-Sat natue of our acquired data.
    imaging_w = 'T2W_F' # flair_obj.ProtocolName            # Since, we are sure about the T2-FLAIR protocol of our acquired data.
    voxel_size = np.array(flair_obj.header['pixdim'][1:4])

    # find out Fat Sat mode:
    if imaging_o[:2] == 'FS':
        fs_mode = 1
    else:
        fs_mode = 0

    #
    # ##  Pre-processing

    # %
    # get nifti data:
    nifti_data = flair_data

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

    # align images to be near:

    # image normalization:
    nifti_data_n = normalization(nifti_data_s, masks_data, imaging_w, fs_mode, a=0, b=1, type='uint16')

    # Save the preprocessed data. You might want to copy the header/affine from the original data.
    save_nifti(nifti_data_n, flair_obj, out_path)

    # Save brain info
    np.savez(out_path.replace('.nii.gz', '_binfo.npz'), brain_cnt=brain_cnt, brain_ax=brain_ax, brain_mask=masks_data)

    # Save slice samples
    slc_nums = [4, 9]
    for slc_num in slc_nums:
        img = nifti_data_n[..., slc_num]
        img = np.uint8(255.0 * img)
        skimage.io.imsave(out_path.replace('.nii.gz', f'_{slc_num+1}.png'), img)
