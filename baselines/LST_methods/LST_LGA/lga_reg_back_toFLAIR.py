import os
import time
import numpy as np
# import torch
import nibabel as nib


def fsl_register(ref_path, primary_path, secondary_path, output_dir):

    from time import monotonic
    import subprocess

    start_time = monotonic()

    # Output path for the registered image
    output_image = os.path.join(output_dir, os.path.basename(primary_path).replace('.nii.gz', '_F.nii.gz'))  # Replace with the desired output path

    # FLIRT command
    flirt_command = [
        '/home/sai/fsl/bin/flirt',
        '-in', primary_path,
        '-ref', ref_path,
        '-out', output_image,
        '-omat', output_image.replace('.nii.gz', '.mat'),  # Optional: Save the transformation matrix
        '-searchrx', '-180', '180',
        '-searchry', '-180', '180'
    ]

    # Apply the transformation matrix to the secondary input
    output_image_sec = os.path.join(output_dir, os.path.basename(secondary_path).replace('.nii.gz', '_F.nii.gz'))  # Replace with the desired output path

    flirt_apply_cmd = [
        "/home/sai/fsl/bin/flirt",
        "-in", secondary_path,
        "-ref", ref_path,
        "-out", output_image_sec,
        "-init", output_image.replace('.nii.gz', '.mat'),  # Use the matrix from the main input registration
        "-applyxfm"
    ]

    try:
        subprocess.run(flirt_command, check=True)
        print("\nFLIRT registration completed successfully.")

        subprocess.run(flirt_apply_cmd, check=True)
        print(f"\n\t\tSecondary FLIRT registration completed successfully for {flirt_apply_cmd[2]}.")
        # Run Time:
        print(f"\nRun Time:  {np.round((monotonic() - start_time), 1)}  seconds\n")

    except subprocess.CalledProcessError as e:
        print(f"Error during FLIRT registration: {e}")
    except FileNotFoundError:
        print("FLIRT command not found. Make sure FSL is installed and in your PATH.")

    return output_image


def register_back(input_dir, ref_dir, sec_dir, output_dir):
    
    subjects = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.nii.gz')]
    subjects_ref = [os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith('.nii.gz')]
    secondary = [os.path.join(sec_dir, f) for f in os.listdir(sec_dir) if f.endswith('.nii.gz')]
    
    for subject in subjects:
        
        ref_file = [f for f in subjects_ref if os.path.basename(f).startswith(os.path.basename(subject).split('_')[0])][0]
        sec_file = [f for f in secondary if os.path.basename(f).split('_')[-1][2:2+6] == os.path.basename(subject).split('_')[0]][0]

        fsl_register(ref_path=ref_file, primary_path=subject, secondary_path=sec_file, output_dir=output_dir)


def main():

    start_time = time.time()  # Start timing the whole process

    input_path = "/mnt/c/Users/SAI/Desktop/Desktop/raw_t1"
    ref_path = "/mnt/c/Users/SAI/Desktop/Desktop/raw_flair"
    sec_path = "/mnt/c/Users/SAI/Desktop/Desktop/paper2_codes/final_data_for_models/test/WMH/LGA/prop"
    
    out_path = os.path.join(os.path.dirname(sec_path), 'on_flair')
    os.makedirs(out_path, exist_ok=True)

    register_back(input_dir=input_path, ref_dir=ref_path, sec_dir=sec_path, output_dir=out_path)

    end_time = time.time()
    print(f"Total execution time until the registration back: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

