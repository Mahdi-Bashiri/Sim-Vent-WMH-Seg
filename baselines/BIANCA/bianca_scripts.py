import os
import time
import shutil
import subprocess
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.pyplot import pause


def load_nifti(file_path):
    """Load a NIfTI file and return the image data and the nibabel object."""
    img = nib.load(file_path)
    data = img.get_fdata()
    return data, img


def save_nifti(data, ref_img, out_path):
    """Save data as a NIfTI file using a reference image for header/affine."""
    new_img = nib.Nifti1Image(data, affine=ref_img.affine, header=ref_img.header)
    nib.save(new_img, out_path)
    print(f"Saved refined mask to {out_path}")


def nii_display(data1, data2, data3,
                data1_title='Ref. Manual Segmentation', data2_title='Auto Segmentation', data3_title='Residual Segmentation'):
    for i in range(10, 18):  # num_slices):
        im1 = data1[:, :, i]
        im2 = data2[:, :, i]
        im3 = data3[:, :, i]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(im1, cmap='gray')
        axes[0].set_title(data1_title)
        axes[1].imshow(im2, cmap='gray')
        axes[1].set_title(data2_title)
        axes[2].imshow(im3, cmap='gray')
        axes[2].set_title(data3_title)

        for ax in axes:
            ax.axis('off')

        plt.suptitle(f'Slice {i + 1}')
        plt.pause(1)  # Pause to observe each slice
        plt.show(block=False)
        # plt.close()


def bianca_model_training(master_file, output_dir):
    """
    Train the BIANCA model using the provided master file.

    Args:
        master_file (str): Path to the master file (e.g., 'master_file_training.txt').
        output_dir (str): Directory where the trained model will be saved.

    Returns:
        str: Path to the trained model.
    """
    
    start_time = time.time()  # Start timing the whole process

    output_model = os.path.join(output_dir, "bianca_model_09")
    
    bianca_command = [
        "bianca",
        "--singlefile", master_file,
        "--labelfeaturenum", "4",
        "--brainmaskfeaturenum", "1",
        "--trainingnums", "all",
        "--featuresubset", "1,2",
        "--matfeaturenum", "3",
        "--trainingpts", "2000",
        "--nonlespts", "10000",
        "--selectpts", "noborder",
        "--saveclassifierdata", output_model,
        "--querysubjectnum", "1",
        "-v"
    ]

    try:
        subprocess.run(bianca_command, check=True)
        print(f"BIANCA training completed. Model saved as {output_model}")
    
        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")

        return output_model

    except subprocess.CalledProcessError as e:
        print(f"Error during BIANCA training: {e}")
        return None
    except FileNotFoundError:
        print("BIANCA command not found. Make sure FSL is installed and in your PATH.")
        return None


def threshold_bianca_output(input_file, output_dir, threshold=0.9):
    """
    Apply thresholding to BIANCA output.

    Args:
        input_file (str): Path to the BIANCA probability map.
        output_dir (str): Directory to save the binary mask.
        threshold (float): Probability threshold for lesion classification.

    Returns:
        str: Path to the binary mask.
    """
    output_bin = os.path.join(output_dir, os.path.basename(input_file).replace(".nii.gz", "_thr09bin.nii.gz"))

    threshold_command = [
        "fslmaths", input_file,
        "-thr", str(threshold),
        "-bin", output_bin
    ]

    try:
        subprocess.run(threshold_command, check=True)
        # print(f"Thresholding completed. Binary mask saved as {output_bin}")
        return output_bin

    except subprocess.CalledProcessError as e:
        print(f"Error during thresholding: {e}")
        return None
    except FileNotFoundError:
        print("fslmaths command not found. Make sure FSL is installed and in your PATH.")
        return None


def run_bianca(master_file, output_dir, threshold=0.9): #, bianca_options):
    """
    
    """
    start_time = time.time()  # Start timing the whole process

    # Read the master file
    with open(master_file, 'r') as f:
        lines = f.readlines()
    
    test_subjects = lines[:90]  # First 90 subjects are test subjects
    train_subjects = lines[90:]  # Remaining subjects are training subjects

    for i, test_subject in enumerate(test_subjects):

        loop_start = time.time()  # Start timing each iteration

        test_subject = test_subject.strip()  # Remove any newline characters
        
        # Construct training numbers (excluding first 90 subjects)
        training_nums = ",".join(map(str, range(91, len(lines) + 1)))  # From subject 91 onward

        bianca_output_file = os.path.join(output_dir, f"bianca_output_test{i+1}.nii.gz")

        # Construct BIANCA command
        bianca_command = [
            "bianca",
            "--singlefile", master_file,
            "--labelfeaturenum", "4",
            "--brainmaskfeaturenum", "1",
            "--featuresubset", "1,2",
            "--matfeaturenum", "3",
            "--trainingpts", "2000",
            "--nonlespts", "10000",
            "--selectpts", "noborder",
            "--trainingnums", training_nums,  # Dynamically excluding the first 10 subjects
            "--querysubjectnum", str(i + 1),  # Query subject = test subject
            "-o", bianca_output_file,
            "-v",

        ] 
        # + bianca_options  # Add additional options if needed

        print(f"Running BIANCA for test subject {i+1}...")

        try:
            subprocess.run(bianca_command, check=True)
            print(f"BIANCA training completed. BIANCA output saved: {bianca_output_file}")

            # Apply thresholding
            thresholded_output = threshold_bianca_output(bianca_output_file, output_dir, threshold)
            if thresholded_output:
                print(f"Thresholding completed. Binary mask saved: {thresholded_output}")

            

        except subprocess.CalledProcessError as e:
            print(f"Error during BIANCA training and running BIANCA for test subject {i+1}: {e}")
            
        except FileNotFoundError:
            print("BIANCA command not found. Make sure FSL is installed and in your PATH.")
            
        

        loop_end = time.time()
        print(f"Time for test subject {i+1}: {loop_end - loop_start:.2f} seconds")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")       
    

def run_pretrained_bianca(master_file, output_dir, pretrained_model, threshold=0.9):
    """
    Runs BIANCA using a pretrained model for segmentation on test subjects.

    Args:
        master_file (str): Path to the master file containing subject information.
        output_dir (str): Directory to save output segmentations.
        pretrained_model (str): Path to the pretrained BIANCA model.
        threshold (float): Threshold value for post-processing the output.

    Returns:
        None
    """

    start_time = time.time()  # Start timing the whole process

    # Read the master file
    with open(master_file, 'r') as f:
        lines = f.readlines()
    
    test_subjects = lines[:90]  # First 90 subjects are test subjects

    for i, test_subject in enumerate(test_subjects):
        loop_start = time.time()  # Start timing each iteration

        test_subject = test_subject.strip()  # Remove any newline characters
        
        # Extract the subject name (based on the first path part, using basename and split)
        subject_name = os.path.basename(os.path.basename(test_subject).split(' ')[0]).split('_')[0]
        
        # Create output file name based on subject name
        bianca_output_file = os.path.join(output_dir, f"{subject_name}_bianca_output.nii.gz")

        # bianca_output_file = os.path.join(output_dir, f"bianca_output_test{i+1}.nii.gz")

        # Construct BIANCA command using pretrained model
        bianca_command = [
            "bianca",
            "--singlefile", master_file,
            "--brainmaskfeaturenum", "1",
            "--featuresubset", "1,2",
            "--matfeaturenum", "3",
            "--querysubjectnum", str(i + 1),  # Query subject = test subject
            "--loadclassifierdata", pretrained_model,  # Load pretrained BIANCA model
            "-o", bianca_output_file,
            "-v"
        ]

        print(f"Running BIANCA for test subject {i+1} using pretrained model...")

        try:
            subprocess.run(bianca_command, check=True)
            print(f"BIANCA segmentation completed. Output saved: {bianca_output_file}")

            # Apply thresholding
            thresholded_output = threshold_bianca_output(bianca_output_file, output_dir, threshold)
            if thresholded_output:
                print(f"Thresholding completed. Binary mask saved: {thresholded_output}")

        except subprocess.CalledProcessError as e:
            print(f"Error running BIANCA for test subject {i+1}: {e}")
        except FileNotFoundError:
            print("BIANCA command not found. Make sure FSL is installed and in your PATH.")

        loop_end = time.time()
        print(f"Time for test subject {i+1}: {loop_end - loop_start:.2f} seconds")

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


# %% 

ROOT_DIR = '/mnt/c/Users/SAI/Desktop/Desktop/paper2_codes'
MODEL_DATA_DIR = os.path.join(ROOT_DIR, 'final_data_for_models')
TRAIN_DATA_DIR = os.path.join(MODEL_DATA_DIR, 'train')
TEST_DATA_DIR = os.path.join(MODEL_DATA_DIR, 'test')

###
test_folders = os.path.join(TEST_DATA_DIR, 'prep_flair_all_forWMH_validation')
test_subjects = [f.split("_")[0] for f in os.listdir(test_folders)]

flair_dir = os.path.join(TRAIN_DATA_DIR, 'prep_flair_all_forWMH')
wmh_dir = os.path.join(TRAIN_DATA_DIR, 'WMH_man_seg_refined')
n_wmh_dir = os.path.join(TRAIN_DATA_DIR, 'all_n_wmh')
vent_dir = os.path.join(TRAIN_DATA_DIR, 'vent_refines2_new2')

t1_repo_dir = '/mnt/e/MBashiri/Thesis/local_dataset/SAI/Demyelinated'

###
wmh_files = os.listdir(wmh_dir)

master_file = os.path.join(TRAIN_DATA_DIR, "master_file_training_ab.txt")
output_dir = os.path.join(TRAIN_DATA_DIR, "bianca_results_ab")
os.makedirs(output_dir, exist_ok=True)


# %%

# # Train and save BIANCA Model
bianca_model_path = bianca_model_training(master_file, TRAIN_DATA_DIR)
# bianca_model_path = '/mnt/c/Users/SAI/Desktop/Desktop/paper2_codes/final_data_for_models/train/bianca_model_09_ab'

master_file_path = "/mnt/c/Users/SAI/Desktop/Desktop/paper2_codes/final_data_for_models/test/master_file_test_ab.txt"
output_directory = "/mnt/c/Users/SAI/Desktop/Desktop/paper2_codes/final_data_for_models/test/WMH/bianca_results_ab"
os.makedirs(output_directory, exist_ok=True)

if bianca_model_path:
    run_pretrained_bianca(master_file_path, output_directory, bianca_model_path, threshold=0.9)


# %%

master_file = os.path.join(TRAIN_DATA_DIR, "master_file_training.txt")
output_dir = os.path.join(TRAIN_DATA_DIR, "bianca_results")
os.makedirs(output_dir, exist_ok=True)


# # Train and save BIANCA Model
bianca_model_path = bianca_model_training(master_file, TRAIN_DATA_DIR)
# bianca_model_path = '/mnt/c/Users/SAI/Desktop/Desktop/paper2_codes/final_data_for_models/train/bianca_model_09'

master_file_path = "/mnt/c/Users/SAI/Desktop/Desktop/paper2_codes/final_data_for_models/test/master_file_test.txt"
output_directory = "/mnt/c/Users/SAI/Desktop/Desktop/paper2_codes/final_data_for_models/test/WMH/bianca_results"
os.makedirs(output_directory, exist_ok=True)

if bianca_model_path:
    run_pretrained_bianca(master_file_path, output_directory, bianca_model_path, threshold=0.9)

