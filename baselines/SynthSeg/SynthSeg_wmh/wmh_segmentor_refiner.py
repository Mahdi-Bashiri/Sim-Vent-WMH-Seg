import os
import gc
import sys
import time
import psutil
# import torch
import subprocess
import numpy as np
import nibabel as nib
from pathlib import Path
from time import monotonic


def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB


def process_single_file(input_file, output_file):
    print(f"\nStarting processing of: {input_file}")
    print(f"Current memory usage: {get_memory_usage():.2f} MB")

    cmd = f"python inference.py --i {input_file} --o {output_file} --device cuda"
    
    try:
        start_time = time.time()
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Success! Time taken: {time.time() - start_time:.2f} seconds")
            return True
        else:
            print("Error occurred:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return False
    finally:
        # Force garbage collection
        gc.collect()


def desired_tissue_grouper(input_dir):
    wmh_labels = [77]

    segmented_data_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.startswith('wmh_')]
    for data_path in segmented_data_paths:
        seg_data = nib.load(data_path)
        seg_data_array = seg_data.get_fdata()

        # Create a binary mask: 1 for values in wmh_labels, 0 for others
        binary_mask_wmh = np.isin(seg_data_array, wmh_labels).astype(np.uint8)

        # Create a new NIfTI image with the modified data
        modified_data = nib.Nifti1Image(binary_mask_wmh, seg_data.affine, seg_data.header)

        # Save the modified NIfTI image
        output_path = data_path.replace('.nii.gz', '_wmh.nii.gz')
        nib.save(modified_data, output_path)   


def fsl_register_mat(ref_path, mat_path, secondary_path):

    # print(f"ref_path    {ref_path}")
    # print(f"mat_path    {mat_path}")
    # print(f"sec_path    {secondary_path}")

    start_time = monotonic()

    # Apply the transformation matrix to the secondary input

    output_image_sec = secondary_path.replace('.nii.gz', '_FLAIR.nii.gz')  # Replace with the desired output path

    flirt_apply_cmd = [
        "/home/sai/fsl/bin/flirt",
        "-in", secondary_path,
        "-ref", ref_path,
        "-out", output_image_sec,
        "-init", mat_path,        
        "-applyxfm"
    ]

    try:
        subprocess.run(flirt_apply_cmd, check=True)
        print(f"\n\t\tSecondary FLIRT registration completed successfully for {flirt_apply_cmd[2]}.")
        # Run Time:
        print(f"\nRun Time:  {np.round((monotonic() - start_time), 1)}  seconds\n")

    except subprocess.CalledProcessError as e:
        print(f"Error during FLIRT registration: {e}")
    except FileNotFoundError:
        print("FLIRT command not found. Make sure FSL is installed and in your PATH.")

    return output_image_sec


def register_back_mat(input_dir, mat_dir, output_dir):
    
    transformation_mats = [os.path.join(mat_dir, f) for f in os.listdir(mat_dir) if f.endswith('.mat')]
    all_candids = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.nii.gz')]
    input_subjects = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.nii.gz')]
    
    for candid in all_candids:
        
        mat_file = [f for f in transformation_mats if os.path.basename(f).startswith(os.path.basename(candid).split("_")[0])][0]
        input_file = [f for f in input_subjects if os.path.basename(f).startswith(os.path.basename(candid).split("_")[0])][0]

        fsl_register_mat(ref_path=input_file, mat_path=mat_file, secondary_path=candid)


def one_masker(input_dir, output_dir):
    
    subjects = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.nii.gz')]
    all_candids_wmh = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('_wmh.nii.gz')]

    for subject in subjects:
        
        related_wmh = [f for f in all_candids_wmh if os.path.basename(f).endswith(os.path.basename(subject).replace('.nii.gz', '_wmh.nii.gz'))][0]

        wmh_data = nib.load(related_wmh)
        wmh_img = wmh_data.get_fdata()

        all_in_one = np.where(wmh_img > 0.5, 1, 0).astype(np.uint8)   # 0.5 is the well-known threshold for prabability masks 

        nifti_img = nib.Nifti1Image(all_in_one, affine=wmh_data.affine, header=wmh_data.header)
        save_path = os.path.join(os.path.dirname(output_dir), 'final')
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, os.path.join(os.path.basename(subject).replace('.nii.gz', '_wmh_deepmask.nii.gz')))
        nib.save(nifti_img, save_path)
    
    return os.path.dirname(save_path)


def main():

    start_time_main = time.time()  # Start timing the whole process

    if len(sys.argv) != 3:
        print("Usage: python script.py input_dir output_dir")
        return
    
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of FLAIR images
    input_files = list(input_dir.glob('*FLAIR*.nii.gz'))
    print(f"Found {len(input_files)} files to process")
    
    # Process files one by one
    for i, input_file in enumerate(input_files, 1):
        print(f"\nProcessing file {i}/{len(input_files)}")
        output_file = output_dir / f"wmh_synthseg_{input_file.name}"
        
        # Skip if output already exists
        if output_file.exists():
            print(f"Skipping {input_file.name} - output already exists")
            continue
        
        # Process file
        success = process_single_file(input_file, output_file)
        
        # Force cleanup
        gc.collect()
        
        # Optional: Add a small delay between processing to allow system to stabilize
        time.sleep(2)
                
        print(f"Memory usage after processing: {get_memory_usage():.2f} MB")
        print("-" * 50)
    
    # Group the related labels for making the desired wmh mask; label 77 defined by the authors
    desired_tissue_grouper(input_dir=output_dir)
    # for one mask of three brain tissue
    final_output_dir = one_masker(input_dir=input_dir, output_dir=output_dir)

    # Register back the mask to the input space; by the transformation matrix resulted from synthseg vent analysis
    register_back_mat(input_dir=input_dir, mat_dir=f'/home/sai/challenge/dr_fahmi_dl/data_test_p2/output', output_dir=final_output_dir)


    end_time_main = time.time()
    print(f"Total execution time: {end_time_main - start_time_main:.2f} seconds")


if __name__ == "__main__":
    main()
