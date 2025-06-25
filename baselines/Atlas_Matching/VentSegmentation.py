# Here, we are going to segment ventricles from gien MRI images.

import os
import cv2
import skimage
import subprocess
import numpy as np
import numpy as npp
import nibabel as nib
from time import monotonic
import matplotlib.pyplot as plt

# from display_nii import display_nii_1 as n_show
# from display_nii import display_nii_mu as mu_show


def halfing(p):
    if p % 2 == 0:
        k = np.array([int(p/2), int(p/2)])
    else:
        k = np.array([int((p-1)/2), int((p+1)/2)])
    return k


def mni2flair(flair_path,
              vent_mni_path='/home/sai/fsl/data/standard/MNI152_T1_2mm.nii.gz',
              mni_path='/home/sai/fsl/data/standard/MNI152_T1_2mm.nii.gz'
              ):
    start_time = monotonic()

    # t1_mni_path = '/home/shamsi/fsl/data/standard/MNI152_T1_2mm.nii.gz'
    # vent_mni_path = '/home/shamsi/fsl/data/standard/MNI152_T1_2mm_VentricleMask.nii.gz'

    input_image = mni_path
    secondary_image = vent_mni_path
    reference_image = flair_path

    # Output path for the registered image
    output_image = secondary_image[:-7] + '_on_FLAIR.nii.gz'  # Replace with the desired output path

    # FLIRT command
    flirt_command = [
        '/home/sai/fsl/bin/flirt',
        '-in', input_image,
        '-ref', reference_image,
        '-out', output_image,
        '-omat', output_image[:-7] + '.mat',  # Optional: Save the transformation matrix
        '-searchrx', '-180', '180',
        '-searchry', '-180', '180'
    ]

    # Apply the transformation matrix to the secondary input
    flirt_apply_cmd = [
        "flirt",
        "-in", secondary_image,
        "-ref", reference_image,
        "-out", output_image,
        "-init", output_image[:-7] + '.mat',  # Use the matrix from the main input registration
        "-applyxfm"
    ]

    try:
        subprocess.run(flirt_command, check=True)
        print("\nFLIRT registration completed successfully.")
        subprocess.run(flirt_apply_cmd, check=True)
        print("Second FLIRT registration completed successfully.")
        # Run Time:
        print(f"\nRun Time:  {np.round((monotonic() - start_time), 1)}  seconds\n")

    except subprocess.CalledProcessError as e:
        print(f"Error during FLIRT registration: {e}")
    except FileNotFoundError:
        print("FLIRT command not found. Make sure FSL is installed and in your PATH.")

    return output_image


def vent_seg_1(csf_path, v_path='/home/sai/fsl/data/standard/MNI152_T1_2mm_VentricleMask.nii.gz'):
    # here, we use the csf map of the normalized flair data on MNI152
    # to have it filtered by a given ventricle mask of MNI152
    # So, we need SPM:Normalize for flair ==> flair_on_MNI, then, SPM:Segment for flair_on_MNI ==> csf_flair_on_MNI
    # and finally, we have to use FSL:flirt for vent_flair_on_MNI ==> vent_flair

    # v_path = head_path + v_path

    start_time = monotonic()

    vent_g = nib.load(v_path)
    csf = nib.load(csf_path)

    vent_g_data = vent_g.get_fdata()
    vent_g_data = np.where(vent_g_data > 0, 1, 0)

    csf_data = csf.get_fdata()
    csf_data = np.where(csf_data > 0.5, csf_data, 0)
    csf_data = (csf_data / np.max(csf_data))

    # n_show(csf_data)
    # print(np.shape(vent_g_data))
    # print(np.shape(csf_data))

    # mu_show(s_n_data, mni_data)

    # resizing the normalized images to be fitted on the mni sizes
    m, n, k = np.array(np.shape(vent_g_data)) - np.array(np.shape(csf_data))
    m = halfing(m)
    n = halfing(n)
    k = halfing(k)
    blank = np.zeros(np.shape(vent_g_data))
    blank[m[0]:-m[1], n[0]:-n[1], 0:-(k[0] + k[1])] = csf_data

    # print(np.shape(vent_g_data))
    # print(np.shape(blank))
    # print(np.max(vent_g_data))
    # print(np.max(blank))

    # mu_show(blank, mni_data)

    ## masking
    # removing extra ventricle masks of very superior laterals
    vent_g_data[..., 53:] = 0
    # removing midline:
    vent_g_data[44: 48, :, 44:] = 0

    # n_show(vent_g_data[..., 43:], 1, 10)
    mask = blank * vent_g_data

    # post-processing:
    # print(np.max(mask))
    morph = np.copy(mask)
    morph = np.where(morph > 0, 1, 0).astype(np.uint8)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel1)
    # morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel2)

    mask_m = morph
    # n_show(mask_morph, 1, 100)

    # further morphing:
    mask_morph = np.copy(mask_m)
    for i in range(0, np.shape(mask_morph)[2]):
        # print(f"                                SLICE : {i}")

        contours, _ = cv2.findContours(mask_morph[..., i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            excluding_mask = np.zeros(np.shape(mask_morph[..., i]))

            # Find the contour with the largest area
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)

            if largest_area <= 1:
                # print("     SMALL", i)
                mask_morph[..., i] = 0
                continue

            for contour in contours:
                # print(f"\n contour:  {contour}")
                area = cv2.contourArea(contour)
                # print(f"largest Area: {largest_area},       area :  {area}")
                if area < 5 * (largest_area / 100):
                    # print(i)
                    if area == 0:
                        # print('contour : ', len(contour))
                        # print((largest_area * 4 / np.sqrt(largest_area)))
                        if len(contour) < 0.2 * (largest_area * 4 / np.sqrt(
                                largest_area)):
                                # inside of paranthesis refers to the probable surounding of an object with given area

                            for j in contour:
                                # print(j)
                                excluding_mask[j[0][1] - 1:j[0][1] + 2, j[0][0] - 1:j[0][0] + 2] = 1
                    else:
                        cv2.drawContours(excluding_mask, [contour], -1, 1, -1)

            excluding_mask = np.where(excluding_mask > 0, 0, 1).astype(np.uint8)

            """       test = skimage.transform.rotate(excluding_mask, 90)
            plt.figure(i + 1)
            skimage.io.imshow(test)
            plt.pause(200)
            plt.show(block=False)
            plt.close()
            print('in for', test.dtype)
            print(np.max(test), np.min(test))
            print(test.shape)"""

            mask_morph[..., i] = mask_morph[..., i] * excluding_mask

    # post-processing:
    # print(np.max(mask_morph))
    morph = np.copy(mask_morph)
    morph = np.where(morph > 0, 1, 0).astype(np.uint8)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel1)
    # morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel2)

    mask_morph_p = morph

    ## resizing the processed images back to the normalized image size

    final_image = mask_morph_p[m[0]:-m[1], n[0]:-n[1], 0:-(k[0] + k[1])]

    """# saving
    for i in range(0, np.shape(final_image)[2]):
        image = csf_data[:, :, i]
        image = skimage.transform.rotate(image, 90)
        skimage.io.imsave('/home/shamsi/test_vent/im_' + str(i+1) + '.png', np.uint8(image*255))

        image = final_image[:, :, i]
        image = skimage.transform.rotate(image.astype(np.float64), 90)
        skimage.io.imsave('/home/shamsi/test_vent/im_' + str(i+1) + '_vents.png', np.uint8(image*255))
    """
    """   # saving
    for i in range(0, np.shape(blank)[2]):

        image = mask[:, :, i]
        image = skimage.transform.rotate(image, 90)
        skimage.io.imsave('/home/shamsi/test_vent/im_' + str(i+1) + '_m.png', np.uint8(image*255))

        image = mask_m[:, :, i]
        image = skimage.transform.rotate(image.astype(np.float64), 90)
        skimage.io.imsave('/home/shamsi/test_vent/im_' + str(i+1) + '_p.png', np.uint8(image*255))

        image = mask_morph[:, :, i]
        image = skimage.transform.rotate(image.astype(np.float64), 90)
        skimage.io.imsave('/home/shamsi/test_vent/im_' + str(i+1) + '_r.png', np.uint8(image*255))

        image = mask_morph_p[:, :, i]
        image = skimage.transform.rotate(image.astype(np.float64), 90)
        skimage.io.imsave('/home/shamsi/test_vent/im_' + str(i+1) + '_s.png', np.uint8(image*255))

        image = vent_g_data[:, :, i]
        image = skimage.transform.rotate(image.astype(np.float64), 90)
        skimage.io.imsave('/home/shamsi/test_vent/im_' + str(i+1) + '_v.png', np.uint8(image*255))

    mu_show(mask, vent_g_data)"""

    # Save the vents only images:
    nifti_img = nib.Nifti1Image(final_image, affine=csf.affine)
    if csf_path[-1] == 'i':
        final_image_path = csf_path[:-4] + '_vents.nii.gz'
    elif csf_path[-1] == 'z':
        final_image_path = csf_path[:-7] + '_vents.nii.gz'

    nib.save(nifti_img, final_image_path)

    print(f"\n                                 Run Time:  {np.round((monotonic() - start_time), 2)}  seconds\n")

    return final_image, final_image_path


def vent_seg_2(csf_dir, flair_dir, head_path, vent_mni_dir='/home/sai/fsl/data/standard/MNI152_T1_2mm_VentricleMask.nii.gz'):
    # here, we use registered vent_MNI on the given FLAIR, and previously produced csf map of the given FLAIR.
    # So, we just need using FSL:Flirt and then, simply filtering the given csf map.

    vent_mni_dir = os.path.join(head_path, vent_mni_dir)

    start_time = monotonic()

    vent_g = nib.load(mni2flair(flair_dir, vent_mni_dir))
    csf = nib.load(csf_dir)

    vent_g_data = vent_g.get_fdata()
    vent_g_data = np.where(vent_g_data > 0, 1, 0)

    csf_data = csf.get_fdata()
    csf_data = np.where(csf_data > 0.9, csf_data, 0)
    csf_data = (csf_data / np.max(csf_data))

    # n_show(csf_data)
    # print(np.shape(vent_g_data))
    # print(np.shape(csf_data)

    ## masking

    mask = csf_data * vent_g_data

    # post-processing:
    # print(np.max(mask))
    morph = np.copy(mask)
    morph = np.where(morph > 0, 1, 0).astype(np.uint8)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel1)
    # morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel2)

    mask_m = morph
    # n_show(mask_morph, 1, 100)

    # further morphing:
    mask_morph = np.copy(mask_m)
    for i in range(0, np.shape(mask_morph)[2]):
        # print(f"                                SLICE : {i}")

        contours, _ = cv2.findContours(mask_morph[..., i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            excluding_mask = np.zeros(np.shape(mask_morph[..., i]))

            # Find the contour with the largest area
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)

            if largest_area <= 1:
                # print("     SMALL", i)
                mask_morph[..., i] = 0
                continue

            for contour in contours:
                # print(f"\n contour:  {contour}")
                area = cv2.contourArea(contour)
                # print(f"largest Area: {largest_area},       area :  {area}")
                if area < 5 * (largest_area / 100):
                    # print(i)
                    if area == 0:
                        # print('contour : ', len(contour))
                        # print((largest_area * 4 / np.sqrt(largest_area)))
                        if len(contour) < 0.2 * (largest_area * 4 / np.sqrt(
                                largest_area)):
                                # inside of paranthesis refers to the probable surounding of an object with given area

                            for j in contour:
                                # print(j)
                                excluding_mask[j[0][1] - 1:j[0][1] + 2, j[0][0] - 1:j[0][0] + 2] = 1
                    else:
                        cv2.drawContours(excluding_mask, [contour], -1, 1, -1)

            excluding_mask = np.where(excluding_mask > 0, 0, 1).astype(np.uint8)

            """       test = skimage.transform.rotate(excluding_mask, 90)
            plt.figure(i + 1)
            skimage.io.imshow(test)
            plt.pause(200)
            plt.show(block=False)
            plt.close()
            print('in for', test.dtype)
            print(np.max(test), np.min(test))
            print(test.shape)"""

            mask_morph[..., i] = mask_morph[..., i] * excluding_mask

    # post-processing:
    # print(np.max(mask_morph))
    morph = np.copy(mask_morph)
    morph = np.where(morph > 0, 1, 0).astype(np.uint8)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel1)
    # morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel2)

    final_image = morph

    """    # saving
    for i in range(0, np.shape(final_image)[2]):
        image = csf_data[:, :, i]
        image = skimage.transform.rotate(image, 90)
        skimage.io.imsave('/home/shamsi/test_vent/im_' + str(i+1) + '.png', np.uint8(image*255))

        image = final_image[:, :, i]
        image = skimage.transform.rotate(image.astype(np.float64), 90)
        skimage.io.imsave('/home/shamsi/test_vent/im_' + str(i+1) + '_vents.png', np.uint8(image*255))
    """
    """    # saving
    for i in range(0, np.shape(final_image)[2]):

        image = mask[:, :, i]
        image = skimage.transform.rotate(image, 90)
        skimage.io.imsave('/home/shamsi/test_vent/im_' + str(i+1) + '_m.png', np.uint8(image*255))

        image = mask_m[:, :, i]
        image = skimage.transform.rotate(image.astype(np.float64), 90)
        skimage.io.imsave('/home/shamsi/test_vent/im_' + str(i+1) + '_p.png', np.uint8(image*255))

        image = mask_morph[:, :, i]
        image = skimage.transform.rotate(image.astype(np.float64), 90)
        skimage.io.imsave('/home/shamsi/test_vent/im_' + str(i+1) + '_r.png', np.uint8(image*255))

        image = vent_g_data[:, :, i]
        image = skimage.transform.rotate(image.astype(np.float64), 90)
        skimage.io.imsave('/home/shamsi/test_vent/im_' + str(i+1) + '_v.png', np.uint8(image*255))
        """
    # Save the vents only images:
    nifti_img = nib.Nifti1Image(final_image, affine=csf.affine)
    if csf_data[-1] == 'i':
        final_image_path = csf_dir[:-4] + '_vents.nii.gz'
    elif csf_data[-1] == 'z':
        final_image_path = csf_dir[:-7] + '_vents.nii.gz'

    nib.save(nifti_img, final_image_path)

    print(f"\n                                 Run Time:  {np.round((monotonic() - start_time), 2)}  seconds\n")

    # mu_show(final_image, vent_g_data)

    return final_image, final_image_path


def vent_seg_mass(main_path, head_dir):
    main_path = os.path.join(head_dir, main_path)
    subjects = os.listdir(main_path)
    vent_flair_paths = []
    for subject in subjects:
        print(subject)

        if subject[0] != 's':  # to avoid attending not interested folders
            continue

        # if subject[-6:] != '102035':
        #     continue

        start_time = monotonic()

        files = os.listdir(os.path.join(main_path, subject))
        for file in files:
            if not file.endswith('.nii') and not file.endswith('.gz'):
                continue
            elif file[0] == '1':            
                flair_file = file
                # print(flair_file)
            elif file[0:2] == 'c3' and file[-8] != 's' and file[-8] != 'R':
                csf_file = file
                # print(csf_file)
            elif file[0:3] == 'mni':
                mni_file = file
                # print(flair_file)

        flair_path = os.path.join(main_path, subject, flair_file)
        csf_path = os.path.join(main_path, subject, csf_file)
        mni_path = os.path.join(main_path, subject, mni_file)

        _, vent_mni_path = vent_seg_1(csf_path)             # this will produce vent_flair_on_MNI images/masks
        vent_flair_path = mni2flair(flair_path, vent_mni_path, mni_path)
        vent_flair_paths.append(vent_flair_path)

    return vent_flair_paths


if __name__ == '__main__':

    head_path = r'C:/Users/DR-Shamsi-LAB/Desktop/'
    head_path = r'/mnt/c/Users/DR-Shamsi-LAB/Desktop/'

    v_dir = '/home/sai/fsl/data/standard/MNI152_T1_2mm_VentricleMask.nii.gz'

    csf_dire = 'tri_SEG/subj_101228/c3101228_AXFLAIR_SE6001.nii.gz'

    flair_dire = 'tri_SEG/subj_101228/101228_AXFLAIR_SE6001.nii.gz'

    main_dir = 'flair_vent'

    # vents = vent_seg_2(csf_dire, flair_dire, v_dir)
    # vents = vent_seg_2(csf_dire, v_dir)
    vent_seg_mass(main_dir, head_path)
