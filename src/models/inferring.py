# %% 
"""
Here, we aim for inferencing on the test or training dataset based on the trained model.
"""

# %%
import os
import cv2
import sys
import time
import shutil
import skimage
import numpy as np
import nibabel as nib
import tensorflow as tf
from time import monotonic
import matplotlib.pyplot as plt
from skimage import io, color, img_as_ubyte
from tensorflow.keras.models import load_model
from scipy.ndimage import gaussian_filter, binary_dilation, label
from skimage.morphology import diamond, binary_opening, binary_closing, erosion, dilation, rectangle, disk, remove_small_objects


# % [markdown]
# ## Phase 0: Functions

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
    # print(f"Saved Predicted data to {out_path}")

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

# %% Function to get the correct path for bundled resources
def get_resource_path(relative_path):

    if getattr(sys, 'frozen', False):  # Running as a bundled executable
        base_path = sys._MEIPASS

    else:  # Running as a script

        try:
            # Try to use __file__ for script-based execution
            base_path = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # Fallback for environments where __file__ is not available
            base_path = os.getcwd()

    return os.path.join(base_path, relative_path)

# %%
def mapping_4l(min, max):

    """
    (0.0*(max-min) + min):background ,
    (0.25*(max-min) + min):vent ,
    (0.75*(max-min) + min):n_wmh ,
    (1.0*(max-min) + min):ab_wmh
    """
    return [(0.0*(max-min) + min), (0.25*(max-min) + min), (0.75*(max-min) + min), (1.0*(max-min) + min)]

# %%
def labels_to_rgb(labels):
    """
    Convert label array to an RGB image.
    :param labels: Discrete label array with values 0 (background), 1 (ventricles), 
                   2 (juxtaventricular WMH), 3 (other WMH).
    :return: RGB image with colors assigned.
    """
    # Create an empty RGB image
    rgb_image = np.zeros((*labels.shape, 3), dtype=np.uint8)

    # Assign colors
    rgb_image[labels == 1] = [0, 0, 255]  # Blue for ventricles
    rgb_image[labels == 2] = [0, 255, 0]  # Green for juxtaventricular WMH
    rgb_image[labels == 3] = [255, 0, 0]  # Red for other WMH
    # Background (labels == 0) remains black: [0, 0, 0]

    return rgb_image

# %%
def midpoint_clustering(predictions, midpoints):
    """
    Apply midpoint-based thresholding to map probabilities to discrete classes.
    :param predictions: Normalized prediction array (0-65535).
    :return: Discrete label array.
    """
    # Create an empty array for discrete labels
    labels = np.zeros_like(predictions, dtype=np.uint16)

    # Apply thresholds based on midpoints
    labels[predictions <= midpoints[0]] = 0  # Background
    labels[(predictions > midpoints[0]) & (predictions <= midpoints[1])] = 1  # Ventricles
    labels[(predictions > midpoints[1]) & (predictions <= midpoints[2])] = 2  # Juxtaventricular WMH
    labels[predictions > midpoints[2]] = 3  # Other WMH

    return labels

# %%
def de_fitter(image, brn_cnt, brn_axs):
    
    new_height = min(256, np.max(brn_axs))
    new_width = min(256, np.max(brn_axs))
    if new_height <= 0 or new_width <= 0:
        raise ValueError(f"Invalid dimensions: new_height={new_height}, new_width={new_width}")

    # Resize the image to the previous bounding box

    # resized_image = np.uint16(np.round(skimage.transform.resize((image + 0.0000001) / 65535.0, (new_height, new_width), anti_aliasing=True) * 65535))   
    resized_image = skimage.transform.resize(image, (new_height, new_width), anti_aliasing=True)

    # Translate the resized image to meet the centers of main brain image

    translated_image = np.zeros_like(image)

    main_height, main_width = translated_image.shape
    small_height, small_width = resized_image.shape

    # Calculate top-left corner for placing the small image
    top_left_x = int(brn_cnt[0] - small_width / 2)
    top_left_y = int(brn_cnt[1] - small_height / 2)

    # Ensure the coordinates don't exceed the boundaries of the main image
    top_left_x = max(0, min(top_left_x, main_width - small_width))
    top_left_y = max(0, min(top_left_y, main_height - small_height))

    # Place the small image on the main image
    translated_image[top_left_y:top_left_y + small_height, top_left_x:top_left_x + small_width] = resized_image
    
    return translated_image

# %%
def transform_back(pred_masks, b_cnts, b_axes):

    de_fit_data = np.zeros_like(pred_masks)

    # Go in loop
    for i in range(pred_masks.shape[-1]):

        im_mask = pred_masks[..., i]

        # Perform the fit function

        if b_axes[..., i][0].any() == 0:
            continue

        de_fit_mask = de_fitter(im_mask, b_cnts[..., i][0], b_axes[..., i][0])

        # Save the fit image
        de_fit_data[..., i] = de_fit_mask

    return de_fit_data

# %%
def generate_images(generator, input_image):
    """Generate an image using the trained Pix2Pix generator."""

    prediction = generator(input_image, training=True)
    
    return prediction.numpy()

# %%
def load_image_stack(image_stack):
    
    """
    Convert a 3D NumPy array of shape (256, 256, 20) to a 4D array 
    suitable for TensorFlow Pix2Pix model input.
    """
    # Ensure the input is in the expected format
    if image_stack.ndim != 3:
        raise ValueError("Input must be a 3D NumPy array of shape (height, width, num_slices).")

    # Normalizing the input to [-1, 1] 
    image_stack = (image_stack / (65535.0 / 2)) - 1

    # Rearrange dimensions: (256, 256, 20) -> (20, 256, 256, 1)
    # Add a channel dimension for grayscale images
    image_stack_4d = np.expand_dims(np.transpose(image_stack, (2, 0, 1)), axis=-1)
    
    # Convert to TensorFlow tensor
    image_tensor = tf.convert_to_tensor(image_stack_4d, dtype=tf.float32)

    return image_tensor

# %%
def get_class_probabilities(model_output):
    """
    Convert single-channel pix2pix output to four separate probability masks
    
    Args:
        model_output: Tensor of shape [batch_size, height, width, 1] with values in range [-1, 1]
    
    Returns:
        List of four tensors, each representing probability of one class
    """
    # Normalize from [-1, 1] to [0, 1] if needed
    if model_output.min() < 0:
        normalized_output = (model_output + 1) / 2
    else:
        normalized_output = model_output
    
    # Define the target values for each class
    class_values = tf.constant([0.0, 0.25, 0.75, 1.0], dtype=tf.float32)
    
    # Expand dimensions for broadcasting
    normalized_output = tf.expand_dims(normalized_output, axis=-1)  # [batch, h, w, 1, 1]
    class_values = tf.reshape(class_values, [1, 1, 1, 1, 4])  # [1, 1, 1, 1, 4]
    
    # Compute distance to each class value
    distances = tf.abs(normalized_output - class_values)  # [batch, h, w, 1, 4]
    distances = tf.squeeze(distances, axis=3)  # [batch, h, w, 4]
    
    # Convert distances to probabilities (closer = higher probability)
    # Using softmax with negative distances (so smaller distance = larger probability)
    # Add temperature parameter to make softmax more decisive
    temperature = 10.0  # Increase this value for sharper probabilities
    probabilities = tf.nn.softmax(-distances * temperature, axis=-1)
    # # Should be very close to 1 for every pixel
    # prob_sum = tf.reduce_sum(probabilities, axis=-1)  # [batch, h, w]
    # max_probs = tf.reduce_max(probabilities, axis=-1)  # [batch, h, w]
    # print(prob_sum, max_probs)
    
    # Split into separate masks
    background_prob = np.transpose(probabilities[..., 0].numpy(), (1, 2, 0))
    ventricles_prob = np.transpose(probabilities[..., 1].numpy(), (1, 2, 0))
    normal_wmh_prob = np.transpose(probabilities[..., 2].numpy(), (1, 2, 0))
    wmh_prob = np.transpose(probabilities[..., 3].numpy(), (1, 2, 0))
    
    return [background_prob, ventricles_prob, normal_wmh_prob, wmh_prob]

# %%
def fitter(image, brn_cnt, brn_axs, output_shape=(256, 256)):

    new_width = np.max(brn_axs)
    new_height = np.max(brn_axs)

    half_width = np.round(new_width // 2).astype(np.uint16)
    half_height = np.round(new_height // 2).astype(np.uint16)
    
    # Calculate cropping box coordinates
    top = int(max(brn_cnt[1] - half_height, 0))
    bottom = int(min(brn_cnt[1] + half_height, image.shape[0]))
    left = int(max(brn_cnt[0] - half_width, 0))
    right = int(min(brn_cnt[0] + half_width, image.shape[1]))
    
    # Crop the image using slicing
    cropped_image = image[top:bottom, left:right]
    # Resize the cropped image to the desired output shape (256x256)
    resized_image = np.uint16(np.round(skimage.transform.resize(cropped_image / np.max(image), output_shape, anti_aliasing=True) * 65535))   
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    return resized_image

# %% Plot Histogram
def plot_log_hist(array, array_name):

    # Flatten the array to include all pixel values
    flat_arr = array.flatten()

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(flat_arr, bins=100, color='blue', alpha=0.7)  # Adjust bins as needed
    plt.yscale('log')  # Set logarithmic scale for the y-axis
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Log-Frequency")
    plt.title("Histogram of Pixel Intensities (Log Scale)")
    plt.grid(True)

    # Save the figure instead of displaying it
    plt.savefig(f"log_histogram_{array_name}.png", dpi=300, bbox_inches='tight')

    # Close the figure to free memory
    plt.close()

# %%
def normalize_array(arr, ctrl):
    if ctrl:
        """Normalize an array to the range [0, 1] using min-max scaling."""
        min_val = np.min(arr)
        max_val = np.max(arr)
        return (arr - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(arr)
    else:
        return arr

# %%
def illustrate_prediction(input_data, pred_data, save_dir):

    save_dir2 = os.path.join(save_dir, 'illustration')
    # if os.path.isdir(save_dir2):
    #     shutil.rmtree(save_dir2)
    os.makedirs(save_dir2, exist_ok=True)

    base_name = os.path.basename(save_dir)

    # RGB colors for masks: red, green, blue
    mask_colors = [
        np.array([1, 0, 0]),  # Red
        np.array([0, 1, 0]),  # Green
        np.array([0, 0, 1])   # Blue
    ]

    for i in range(input_data.shape[2]):  # Iterate over 20 slices
        flair_slice = input_data[:, :, i]
        rgb_image = np.stack([flair_slice]*3, axis=-1)  # Convert grayscale to RGB

        if len(pred_data) > 0:
            for j, mask in enumerate(pred_data):

                # print(f"Mask {j} shape: {mask.shape}, unique vals: {np.unique(mask[:, :, i])}")

                if j >= 3:
                    break
                binary_mask = mask[:, :, i] > 0.5
                color_mask = mask_colors[j]
                # Blend color on masked areas
                rgb_image[binary_mask] = (
                    0.5 * rgb_image[binary_mask] + 0.5 * color_mask
                )
                
                # # Save as PNG using skimage
                # image_path = os.path.join(save_dir2, f"{base_name}_mask_{j}_{i}.png")
                # io.imsave(image_path, np.uint8((mask[:, :, i].clip(0, 1))*255))

        # Save as PNG using skimage
        image_path = os.path.join(save_dir2, f"{base_name}_{i}.png")
        io.imsave(image_path, np.rot90(np.uint8((rgb_image.clip(0, 1))*255), 1))

# %%
def main(prep_flair_path, brain_info_path, model, save_dir):

    # % [markdown]
    # ## Phase 1: Inputs

    # %
    # Load preprocessed data
    os.makedirs(save_dir, exist_ok=True)

    normalized_data, normalized_img = load_nifti(prep_flair_path)   

    # brain extraction:
    brain_info = np.load(brain_info_path)
    # masks_data, brain_cnt, brain_ax = brain_mask_new(normalized_data)
    # masks_data = np.where(masks_data < 128, 0, 1).astype(np.uint8)

    brain_mask = brain_info["brain_mask"]   # masks_data     
    brain_cnts = brain_info["brain_cnt"]    # brain_cnt      
    brain_axes = brain_info["brain_ax"]     # brain_ax      

    # % [markdown]
    # ## Phase 2: Preparing Inputs

    # %
    # Generate the suitable images for feeding the trained models

    fit_data = np.zeros_like(normalized_data)

    # Go in loop
    for i in range(3, normalized_data.shape[-1]):

        im_fl = normalized_data[..., i]

        if np.sum(brain_mask[..., i]) < (3.14 * 40 * 40):
            # print(f'Small Seen Brain in Slice: {i+1} / {normalized_data.shape[-1]}')
            continue                                # to avoid tiny extracted brain from the FLAIR images

        # Perform the fit function
        fit_flair = fitter(im_fl, brain_cnts[..., i][0], brain_axes[..., i][0])

        # Save the rotated fit image
        fit_data[..., i] = np.rot90(fit_flair, 1)

    # np.save(os.path.join(u_folder, 'fit_data.npy'), fit_data)


    # % [markdown]
    # ## Phase 3: Segmentations

    # % [markdown]
    # ### Load Data

    # %
    # Load and Convert the data to suitable input
    input_tensors = load_image_stack(fit_data)


    # % [markdown]
    # ### Predict 4L Masks, including ventricle and two types of WMH

    # %
    # Generate the output images
    model_output = generate_images(model, input_tensors)
    class_probabilities = get_class_probabilities(model_output)

    pred_back = class_probabilities[0]
    pred_vent = class_probabilities[1]
    pred_nwmh = class_probabilities[2]
    pred_awmh = class_probabilities[3]

    # #
    # print(f"\n Pred Back: {np.max(pred_back)}, {np.min(pred_back)}")
    # print(f"\n Pred Vent: {np.max(pred_vent)}, {np.min(pred_vent)}")
    # print(f"\n Pred nWMH: {np.max(pred_nwmh)}, {np.min(pred_nwmh)}")
    # print(f"\n Pred aWMH: {np.max(pred_awmh)}, {np.min(pred_awmh)}")
    # plot_log_hist(pred_back, 'pred_back')

    # Reshape the array to (256, 256, 20)
    vent_wmh_masks = np.squeeze(model_output, axis=-1)          # Remove the singleton dimension (axis=-1)
    vent_wmh_masks = np.transpose(vent_wmh_masks, (1, 2, 0))      # Transpose to (256, 256, 20)

    # Normalize back to suit the uint16 format
    vent_wmh_masks = np.round(((vent_wmh_masks * 0.5) + 0.5) * 255.0).astype(np.uint8)

    # show the predictions
    for i in range(vent_wmh_masks.shape[-1]):
        os.makedirs(os.path.join(save_dir, 'illustration'), exist_ok=True)
        skimage.io.imsave(os.path.join(save_dir, 'illustration', f'v_w_mask_{i+1}.png'), vent_wmh_masks[..., i])

    # Rotate back the resulted mask
    for slc_ in range(3, vent_wmh_masks.shape[-1]):
        # Rotate back the arrays
        vent_wmh_masks[..., slc_] = np.rot90(vent_wmh_masks[..., slc_], 1)
        pred_back[..., slc_] = np.rot90(pred_back[..., slc_], -1)
        pred_vent[..., slc_] = np.rot90(pred_vent[..., slc_], -1)
        pred_nwmh[..., slc_] = np.rot90(pred_nwmh[..., slc_], -1)
        pred_awmh[..., slc_] = np.rot90(pred_awmh[..., slc_], -1)

    # % [markdown]
    # ## Phase 4: Post-processing

    # % [markdown]
    # ### Convert back the Predicted Masks

    # %
    # Transform back the predicted masks to main FLAIR images space
    pred_main = transform_back(vent_wmh_masks, brain_cnts, brain_axes)
    pred_back = transform_back(pred_back, brain_cnts, brain_axes)
    pred_vent = transform_back(pred_vent, brain_cnts, brain_axes)
    pred_nwmh = transform_back(pred_nwmh, brain_cnts, brain_axes)
    pred_awmh = transform_back(pred_awmh, brain_cnts, brain_axes)


    # % [markdown]
    # ### 4L Mask Post-processing

    # Make a wholesome WMH mask containing both normal and abnormal predictions
    pred_wwmh = pred_awmh + pred_nwmh    # Since there is almost no overlap between two arrays

    # %
    # Morphologically post-process the three masks.
    thr = 0.35
    mask_back = np.where(pred_back > thr, 1, 0).astype(np.uint8)
    mask_vent = np.where(pred_vent > thr, 1, 0).astype(np.uint8)
    mask_nwmh = np.where(pred_nwmh > thr, 1, 0).astype(np.uint8)
    mask_awmh = np.where(pred_awmh > thr, 1, 0).astype(np.uint8)
    mask_wwmh = np.where(pred_wwmh > thr, 1, 0).astype(np.uint8)

    # for ventricle masks:

    vent_m = mask_vent.astype(bool)

    for i in range(vent_m.shape[-1]):
        vent_m1 = vent_m[..., i]
        vent_m1 = remove_small_objects(vent_m1, min_size=5)
        vent_m1 = binary_closing(vent_m1, disk(1))
        vent_m1 = binary_opening(vent_m1, disk(1))
        # vent_m = remove_small_objects(vent_m, min_size=20)
        vent_m[..., i] = vent_m1

    mask_vent_p = (vent_m *1).astype(np.uint8)

    # for juxtaventricle WMH masks:

    v_wmh = mask_nwmh.astype(bool)

    for i in range(v_wmh.shape[-1]):
        v_wmh1 = v_wmh[..., i]

        # filtering by the approximity to the ventricle masks
        vent_m1 = vent_m[..., i]
        vent_m1 = dilation(vent_m1, disk(3))
        v_wmh1 = v_wmh1 & vent_m1

        v_wmh1 = remove_small_objects(v_wmh1, min_size=5)
        v_wmh1 = binary_closing(v_wmh1, disk(1))
        # v_wmh1 = binary_opening(v_wmh1, disk(1))
        # v_wmh1 = remove_small_objects(v_wmh, min_size=20)
        v_wmh[..., i] = v_wmh1

    mask_nwmh_p = (v_wmh *1).astype(np.uint8)

    # update the vent masks:
    vent_m = vent_m & ~v_wmh
    mask_vent_p = (vent_m *1).astype(np.uint8)

    # for WMH masks:

    wmh = mask_awmh.astype(bool)

    for i in range(wmh.shape[-1]):
        wmh1 = wmh[..., i]

        # filtering by the approximity to the ventricle and juxtavntricle WMH masks
        vent_m1 = vent_m[..., i]
        v_wmh1 = v_wmh[..., i]
        vent_m1 = dilation(vent_m1, disk(3))
        wmh1 = wmh1 & ~vent_m1
        wmh1 = wmh1 & ~v_wmh1

        wmh1 = remove_small_objects(wmh1, min_size=5)
        wmh1 = binary_closing(wmh1, disk(1))
        # wmh1 = binary_opening(wmh1, disk(1))
        # wmh1 = remove_small_objects(wmh1, min_size=20)
        wmh[..., i] = wmh1

    mask_awmh_p = (wmh *1).astype(np.uint8)

    #
    mask_vent_p_bool = mask_vent_p.astype(bool)
    pred_vent_rf = np.zeros_like(pred_vent)
    pred_vent_rf[mask_vent_p_bool] = pred_vent[mask_vent_p_bool]

    mask_nwmh_p_bool = mask_nwmh_p.astype(bool)
    pred_nwmh_rf = np.zeros_like(pred_nwmh)
    pred_nwmh_rf[mask_nwmh_p_bool] = pred_nwmh[mask_nwmh_p_bool]

    mask_awmh_p_bool = mask_awmh_p.astype(bool)
    pred_awmh_rf = np.zeros_like(pred_awmh)
    pred_awmh_rf[mask_awmh_p_bool] = pred_awmh[mask_awmh_p_bool]

    mask_wwmh_p_bool = mask_awmh_p.astype(bool) | mask_nwmh_p.astype(bool)
    pred_wwmh_rf = np.zeros_like(pred_wwmh)
    pred_wwmh_rf[mask_wwmh_p_bool] = pred_wwmh[mask_wwmh_p_bool]

    pred_main_rf = pred_back + pred_vent_rf + pred_nwmh_rf + pred_awmh_rf

    # % 
    # Save the predicted masks into nifti files

    save_path = os.path.join(save_dir, f"{os.path.basename(prep_flair_path).split('.')[0]}_our_main.nii.gz")
    save_nifti(pred_main, normalized_img, save_path)
    save_path = os.path.join(save_dir, f"{os.path.basename(prep_flair_path).split('.')[0]}_our_main_rf.nii.gz")
    save_nifti(pred_main_rf, normalized_img, save_path)

    save_path = os.path.join(save_dir, f"{os.path.basename(prep_flair_path).split('.')[0]}_our_back.nii.gz")
    save_nifti(pred_back, normalized_img, save_path)

    save_path = os.path.join(save_dir, f"{os.path.basename(prep_flair_path).split('.')[0]}_our_vent.nii.gz")
    save_nifti(pred_vent, normalized_img, save_path)
    save_path = os.path.join(save_dir, f"{os.path.basename(prep_flair_path).split('.')[0]}_our_vent_rf.nii.gz")
    save_nifti(pred_vent_rf, normalized_img, save_path)
    save_path = os.path.join(save_dir, f"{os.path.basename(prep_flair_path).split('.')[0]}_our_vent_mask.nii.gz")
    save_nifti(mask_vent_p, normalized_img, save_path)

    save_path = os.path.join(save_dir, f"{os.path.basename(prep_flair_path).split('.')[0]}_our_nwmh.nii.gz")
    save_nifti(pred_nwmh, normalized_img, save_path)
    save_path = os.path.join(save_dir, f"{os.path.basename(prep_flair_path).split('.')[0]}_our_nwmh_rf.nii.gz")
    save_nifti(pred_nwmh_rf, normalized_img, save_path)
    save_path = os.path.join(save_dir, f"{os.path.basename(prep_flair_path).split('.')[0]}_our_nwmh_mask.nii.gz")
    save_nifti(mask_nwmh_p, normalized_img, save_path)

    save_path = os.path.join(save_dir, f"{os.path.basename(prep_flair_path).split('.')[0]}_our_awmh.nii.gz")
    save_nifti(pred_awmh, normalized_img, save_path)
    save_path = os.path.join(save_dir, f"{os.path.basename(prep_flair_path).split('.')[0]}_our_awmh_rf.nii.gz")
    save_nifti(pred_awmh_rf, normalized_img, save_path)
    save_path = os.path.join(save_dir, f"{os.path.basename(prep_flair_path).split('.')[0]}_our_awmh_mask.nii.gz")
    save_nifti(mask_awmh_p, normalized_img, save_path)

    save_path = os.path.join(save_dir, f"{os.path.basename(prep_flair_path).split('.')[0]}_our_wwmh.nii.gz")
    save_nifti(pred_wwmh, normalized_img, save_path)
    save_path = os.path.join(save_dir, f"{os.path.basename(prep_flair_path).split('.')[0]}_our_wwmh_rf.nii.gz")
    save_nifti(pred_wwmh_rf, normalized_img, save_path)
    save_path = os.path.join(save_dir, f"{os.path.basename(prep_flair_path).split('.')[0]}_our_wwmh_mask.nii.gz")
    save_nifti(np.uint8(mask_wwmh_p_bool *1), normalized_img, save_path)

    # %
    # Illustrate the predictions on the main input data and save the images
    # illustrate_prediction(normalized_data, [mask_awmh_p, mask_nwmh_p, mask_vent_p], save_dir=save_dir)

