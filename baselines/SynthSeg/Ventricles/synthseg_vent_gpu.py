import os
import time
import numpy as np
import torch
import nibabel as nib
from SynthSeg.predict import predict
# from SynthSeg.brain_generator import BrainGenerator
# from SynthSeg.segmentation import predict_syntax_segmentation


def setup_gpu():
    """Configure GPU if available"""
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        # Set default tensor type to cuda
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        return True
    else:
        print("No GPU available, using CPU")
        return False


def prepare_data_directory(src_dir):
    """
    Create necessary directories for SynthSeg processing
    """
    os.makedirs(f'{src_dir}/input', exist_ok=True)
    os.makedirs(f'{src_dir}/output', exist_ok=True)
    os.makedirs(f'{src_dir}/models', exist_ok=True)


def download_pretrained_models():
    """
    Download pretrained SynthSeg models
    Note: Replace with actual download method if needed
    """
    # Placeholder for model download logic
    print("Download pretrained models manually from SynthSeg GitHub repository")


def generate_synthetic_brain_images(num_images=10):
    """
    Generate synthetic brain images using SynthSeg
    """
    brain_generator = Brain_Generator()
    
    for i in range(num_images):
        synthetic_image = brain_generator.generate_brain()
        output_path = f'data/input/synthetic_brain_{i}.nii.gz'
        nib.save(nib.Nifti1Image(synthetic_image, np.eye(4)), output_path)
    
    print(f"Generated {num_images} synthetic brain images")


def perform_brain_segmentation(input_dir='data/input', output_dir='data/output', model_dir='data/models', label_dir='SynthSeg-master/data/labels_classes_priors'):
    """
    Perform brain segmentation on input images
    """
    # List all input images
    input_images = [
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.endswith(('.nii', '.nii.gz'))
    ]
    print(input_images)
    
    for image_path in input_images:
        # Perform segmentation
        # paths to input/output files
        # Here we assume the availability of an image that we wish to segment with a model we have just trained.
        # We emphasise that we do not provide such an image (this is just an example after all :))
        # Input images must have a .nii, .nii.gz, or .mgz extension.
        # Note that path_images can also be the path to an entire folder, in which case all the images within this folder will
        # be segmented. In this case, please provide path_segm (and possibly path_posteriors, and path_resampled) as folder.
        path_images = image_path
        # path to the output segmentation
        path_segm = os.path.join(output_dir, os.path.basename(image_path).replace('.nii.gz', '_seg.nii.gz'))
        # we can also provide paths for optional files containing the probability map for all predicted labels
        path_posteriors = os.path.join(output_dir, os.path.basename(image_path).replace('.nii.gz', '_post.nii.gz'))
        # and for a csv file that will contain the volumes of each segmented structure
        path_vol = os.path.join(output_dir, os.path.basename(image_path).replace('.nii.gz', '_volumes.csv'))

        # of course we need to provide the path to the trained model (here we use the main synthseg model).
        path_model = os.path.join(model_dir, 'synthseg_2.0.h5')
        # but we also need to provide the path to the segmentation labels used during training
        path_segmentation_labels = os.path.join(label_dir, 'synthseg_segmentation_labels_2.0.npy')
        print(np.unique(np.load(path_segmentation_labels)))
        # optionally we can give a numpy array with the names corresponding to the structures in path_segmentation_labels
        path_segmentation_names = os.path.join(label_dir, 'synthseg_segmentation_names_2.0.npy')
        print(np.load(path_segmentation_names))
        # We can now provide various parameters to control the preprocessing of the input.
        # First we can play with the size of the input. Remember that the size of input must be divisible by 2**n_levels, so the
        # input image will be automatically padded to the nearest shape divisible by 2**n_levels (this is just for processing,
        # the output will then be cropped to the original image size).
        # Alternatively, you can crop the input to a smaller shape for faster processing, or to make it fit on your GPU.
        # cropping = 192
        # Finally, we finish preprocessing the input by resampling it to the resolution at which the network has been trained to
        # produce predictions. If the input image has a resolution outside the range [target_res-0.05, target_res+0.05], it will
        # automatically be resampled to target_res.
        # target_res = 1.
        # Note that if the image is indeed resampled, you have the option to save the resampled image.
        path_resampled = os.path.join(output_dir, os.path.basename(image_path).replace('.nii.gz', '_res.nii.gz'))

        # After the image has been processed by the network, there are again various options to postprocess it.
        # First, we can apply some test-time augmentation by flipping the input along the right-left axis and segmenting
        # the resulting image. In this case, and if the network has right/left specific labels, it is also very important to
        # provide the number of neutral labels. This must be the exact same as the one used during training.
        # flip = True
        # n_neutral_labels = 18
        # Second, we can smooth the probability maps produced by the network. This doesn't change much the results, but helps to
        # reduce high frequency noise in the obtained segmentations.
        sigma_smoothing = 0.5
        # Then we can operate some fancier version of biggest connected component, by regrouping structures within so-called
        # "topological classes". For each class we successively: 1) sum all the posteriors corresponding to the labels of this
        # class, 2) obtain a mask for this class by thresholding the summed posteriors by a low value (arbitrarily set to 0.1),
        # 3) keep the biggest connected component, and 4) individually apply the obtained mask to the posteriors of all the
        # labels for this class.
        # Example: (continuing the previous one)  generation_labels = [0, 24, 507, 2, 3, 4, 17, 25, 41, 42, 43, 53, 57]
        #                                             output_labels = [0,  0,  0,  2, 3, 4, 17,  2, 41, 42, 43, 53, 41]
        #                                       topological_classes = [0,  0,  0,  1, 1, 2,  3,  1,  4,  4,  5,  6,  7]
        # Here we regroup labels 2 and 3 in the same topological class, same for labels 41 and 42. The topological class of
        # unsegmented structures must be set to 0 (like for 24 and 507).
        # topology_classes = '../../data/labels_classes_priors/synthseg_topological_classes.npy'
        # Finally, we can also operate a strict version of biggest connected component, to get rid of unwanted noisy label
        # patch that can sometimes occur in the background. If so, we do recommend to use the smoothing option described above.
        # keep_biggest_component = True

        # Regarding the architecture of the network, we must provide the predict function with the same parameters as during
        # training.
        # n_levels = 5
        # nb_conv_per_level = 2
        # conv_size = 3
        # unet_feat_count = 24
        # activation = 'elu'
        # feat_multiplier = 2

        # Finally, we can set up an evaluation step after all images have been segmented.
        # In this purpose, we need to provide the path to the ground truth corresponding to the input image(s).
        # This is done by using the "gt_folder" parameter, which must have the same type as path_images (i.e., the path to a
        # single image or to a folder). If provided as a folder, ground truths must be sorted in the same order as images in
        # path_images.
        # Just set this to None if you do not want to run evaluation.
        # gt_folder = '/the/path/to/the/ground_truth/gt.nii.gz'
        # Dice scores will be computed and saved as a numpy array in the folder containing the segmentation(s).
        # This numpy array will be organised as follows: rows correspond to structures, and columns to subjects. Importantly,
        # rows are given in a sorted order.
        # Example: we segment 2 subjects, where output_labels = [0,  0,  0,  2, 3, 4, 17,  2, 41, 42, 43, 53, 41]
        #                             so sorted output_labels = [0, 2, 3, 4, 17, 41, 42, 43, 53]
        # dice = [[xxx, xxx],  # scores for label 0
        #         [xxx, xxx],  # scores for label 2
        #         [xxx, xxx],  # scores for label 3
        #         [xxx, xxx],  # scores for label 4
        #         [xxx, xxx],  # scores for label 17
        #         [xxx, xxx],  # scores for label 41
        #         [xxx, xxx],  # scores for label 42
        #         [xxx, xxx],  # scores for label 43
        #         [xxx, xxx]]  # scores for label 53
        #         /       \
        #   subject 1    subject 2
        #
        # Also we can compute different surface distances (Hausdorff, Hausdorff99, Hausdorff95 and mean surface distance). The
        # results will be saved in arrays similar to the Dice scores.
        # compute_distances = True

        # All right, we're ready to make predictions !!
        seg_path = predict(path_images,
                path_segm,
                path_model,
                path_segmentation_labels,
                # n_neutral_labels=n_neutral_labels,
                path_posteriors=path_posteriors,
                path_resampled=path_resampled,
                path_volumes=path_vol,
                names_segmentation=path_segmentation_names,
                # cropping=cropping,
                # target_res=target_res,
                # flip=flip,
                # topology_classes=topology_classes,
                sigma_smoothing=sigma_smoothing,
                # keep_biggest_component=keep_biggest_component,
                # n_levels=n_levels,
                # nb_conv_per_level=nb_conv_per_level,
                # conv_size=conv_size,
                # unet_feat_count=unet_feat_count,
                # feat_multiplier=feat_multiplier,
                # activation=activation,
                # gt_folder=gt_folder,
                # compute_distances=compute_distances
                )
        print(seg_path)


def main_tissue_grouper(input_dir='data/output'):
    vent_labels = [14, 15, 5, 44, 4, 43]

    segmented_data_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('_seg.nii.gz')]
    for data_path in segmented_data_paths:
        seg_data = nib.load(data_path)
        seg_data_array = seg_data.get_fdata()

        # Create a binary mask: 1 for values in vent_labels, 0 for others
        binary_mask_vent = np.isin(seg_data_array, vent_labels).astype(np.uint8)

        # Create a new NIfTI image with the modified data
        modified_data = nib.Nifti1Image(binary_mask_vent, seg_data.affine, seg_data.header)

        # Save the modified NIfTI image
        output_path = data_path.replace('.nii.gz', '_vent.nii.gz')
        nib.save(modified_data, output_path)   


def advanced_segmentation_analysis(input_dir='data/output'):
    """
    Advanced segmentation analysis and visualization
    """
    import matplotlib.pyplot as plt
    
    segmentation_files = [
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if f.endswith('_seg.nii.gz')
    ]
    
    for seg_file in segmentation_files:
        # Load segmentation
        seg_img = nib.load(seg_file)
        seg_data = seg_img.get_fdata()
        
        # Plot segmentation
        plt.figure(figsize=(10, 5))
        plt.imshow(seg_data[:, :, seg_data.shape[2]//2], cmap='viridis')
        plt.title(f'Segmentation: {os.path.basename(seg_file)}')
        plt.colorbar()
        plt.savefig(seg_file.replace('.nii.gz', '_viz.png'))
        plt.close()


def fsl_register(ref_path, primary_path, secondary_paths):

    from time import monotonic
    import subprocess

    start_time = monotonic()

    # Output path for the registered image
    output_image = primary_path.replace('.nii.gz', '_T1.nii.gz')  # Replace with the desired output path

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
    flirt_apply_cmd_list = []
    for secondary in secondary_paths:

        output_image_sec = secondary.replace('.nii.gz', '_T1.nii.gz')  # Replace with the desired output path

        flirt_apply_cmd = [
            "flirt",
            "-in", secondary,
            "-ref", ref_path,
            "-out", output_image_sec,
            "-init", output_image.replace('.nii.gz', '.mat'),  # Use the matrix from the main input registration
            "-applyxfm"
        ]

        flirt_apply_cmd_list.append(flirt_apply_cmd)

    try:
        subprocess.run(flirt_command, check=True)
        print("\nFLIRT registration completed successfully.")

        for flirt_cmd in flirt_apply_cmd_list:
            subprocess.run(flirt_cmd, check=True)
            print(f"\n\t\tSecondary FLIRT registration completed successfully for {flirt_cmd[2]}.")
        # Run Time:
        print(f"\nRun Time:  {np.round((monotonic() - start_time), 1)}  seconds\n")

    except subprocess.CalledProcessError as e:
        print(f"Error during FLIRT registration: {e}")
    except FileNotFoundError:
        print("FLIRT command not found. Make sure FSL is installed and in your PATH.")

    return output_image


def register_back(input_dir='/home/sai/challenge/dr_fahmi_dl/data/input', output_dir='/home/sai/challenge/dr_fahmi_dl/data/output'):
    
    subjects = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.nii.gz')]
    all_candids = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.nii.gz')]
    
    for subject in subjects:
        
        related_files = [f for f in all_candids if os.path.basename(f).startswith(os.path.basename(subject).split('_')[0])]
        primary_files = [f for f in related_files if os.path.basename(f).endswith('_res.nii.gz')]

        secomdary_files = [f for f in related_files if f not in primary_files]
        primary_file = primary_files[0]

        fsl_register(ref_path=subject, primary_path=primary_file, secondary_paths=secomdary_files)


def one_masker(input_dir='/home/sai/challenge/dr_fahmi_dl/data/input', output_dir='/home/sai/challenge/dr_fahmi_dl/data/output'):
    
    subjects = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.nii.gz')]
    all_candids_vent = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('_vent.nii.gz')]

    for subject in subjects:
        
        related_vent = [f for f in all_candids_vent if os.path.basename(f).startswith(os.path.basename(subject).split('.')[0])][0]

        vent_data = nib.load(related_vent)
        vent_img = vent_data.get_fdata()

        all_in_one = np.where(vent_img > 0.5, 1, 0).astype(np.uint8)   # 0.5 is the well-known threshold for prabability masks 

        nifti_img = nib.Nifti1Image(all_in_one, affine=vent_data.affine, header=vent_data.header)
        save_path = os.path.join(os.path.dirname(output_dir), 'final')
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, os.path.join(os.path.basename(subject).replace('.nii.gz', '_deepmask.nii.gz')))
        nib.save(nifti_img, save_path)


# If SynthSeg uses TensorFlow under the hood (which is likely), you'll also need to add:
def setup_tensorflow_gpu():
    """Configure TensorFlow to use GPU"""
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"TensorFlow GPUs available: {len(physical_devices)}")
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("Memory growth enabled for TensorFlow GPUs")
        return True
    else:
        print("No TensorFlow GPU available")
        return False


def main():
    # Setup GPU if available
    gpu_available = setup_gpu()
    setup_tensorflow_gpu()
    
    start_time = time.time()  # Start timing the whole process

    # Setup environment
    src_dir = 'data_test_p2'
    prepare_data_directory(src_dir)
    
    # Set environment variable for SynthSeg to use GPU
    if gpu_available:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU if no GPU
    
    # Perform segmentation
    perform_brain_segmentation(input_dir=f'{src_dir}/input', output_dir=f'{src_dir}/output', model_dir=f'{src_dir}/models')

    # Rest of your code remains the same
    end_time = time.time()
    print(f"\nTotal execution time until segmentation: {end_time - start_time:.2f} seconds")

    # Group the three main brain tissue separately
    main_tissue_grouper(input_dir=f'{src_dir}/output')
    
    # for one mask of three brain tissue
    one_masker(input_dir=f'{src_dir}/input', output_dir=f'{src_dir}/output')
    
    end_time = time.time()
    print(f"Total execution time until mask making: {end_time - start_time:.2f} seconds")

    # Visualize results
    advanced_segmentation_analysis(input_dir=f'{src_dir}/output')

    register_back(input_dir=f'/home/sai/challenge/dr_fahmi_dl/{src_dir}/input', output_dir=f'/home/sai/challenge/dr_fahmi_dl/{src_dir}/output')

    end_time = time.time()
    print(f"Total execution time until the registration back: {end_time - start_time:.2f} seconds")


# Then modify your main function to also call:
# setup_tensorflow_gpu()

if __name__ == "__main__":
    main()

