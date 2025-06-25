# Repository Structure Documentation

## Overview

This document describes the implementation details and organizational structure of our second research article, which focuses on the **simultaneous segmentation of brain ventricles and normal/abnormal brain white matter hyperintensities in FLAIR MRI images**.

Detailed methodology and results are provided in the accompanying files and can be referenced in our published article. This document specifically outlines the implementation structure and organization of the Python-based codebase for developing and maintaining this GitHub repository.

## Repository Structure

The main directory contains **6 folders** and **2 files** organized as follows:

```
├── Article_Figures/
├── Auxillary_compared_methods_details/
├── Phase1_data_preprocessing/
├── Phase2_data_preparation_for_model_training/
├── Phase3_model_training_and_inferencing_and_evaluation/
├── Phase4_comparison_analysis/
├── our_article_DOI.md
└── repo_explanation.docx
```

## Detailed Directory Structure

### Article_Figures/
Contains all figures used in the research article: **11 main figures** and **2 supplementary figures**.

```
├── Figure_1.png
├── Figure_2.pdf
├── Figure_2.png
├── Figure_3.pdf
├── Figure_4.png
├── Figure_5.png
├── Figure_6.png
├── Figure_7.png
├── Figure_8.png
├── Figure_9.png
├── Figure_10.png
├── Figure_11.png
├── Figure_S1_wmh_metrics_vs_threshold.png
└── Figure_S2_wmh_default_vs_optimal_comparison.png
```

### Auxillary_compared_methods_details/
Contains **7 subdirectories**: 6 directories for code and execution instructions for each literature method compared with our approach, plus one directory containing raw sample data.

```
├── Atlas_Matching/
├── Bianca/
├── LST_LGA/
├── LST_LPA/
├── raw_data/
│   ├── subjects_flair/
│   └── subjects_t1/
├── SynthSeg/
└── SynthSeg_wmh/
```

The `raw_data` directory contains sample data from **5 patients** with both FLAIR and T1 sequences. Each method directory includes necessary documentation and explanatory files as needed.

### Phase1_data_preprocessing/
Contains preprocessing scripts for raw input files.

```
├── raw_data/
└── pre_processing_flair.py
```

The `raw_data` directory contains the same sample data as mentioned above. The Python script handles preprocessing of raw input files.

### Phase2_data_preparation_for_model_training/
Contains **5 data directories** and **1 Python script** for generating input images for the pix2pix (cGAN) model.

```
├── Original_FLAIRs_prep/
├── abWMH_manual_segmentations/
├── nWMH_manual_segmentations/
├── vent_manual_segmentations/
├── manual_4l_masks_april/
└── generating_4L_masks.py
```

**Directory descriptions:**
- **Original_FLAIRs_prep/**: Contains preprocessed patient data files. Each patient has one NIfTI file and one NPZ file containing the FLAIR image, brain mask, and mask metadata.
- **abWMH_manual_segmentations/**: Contains manual segmentation masks for abnormal lesions.
- **nWMH_manual_segmentations/**: Contains manual segmentation masks for normal lesions.
- **vent_manual_segmentations/**: Contains manual segmentation masks for brain ventricles.
- **manual_4l_masks_april/**: Contains generated 4-level masks from the above segmentations.

The Python script generates 4-level masks and creates paired images for pix2pix model training.

### Phase3_model_training_and_inferencing_and_evaluation/
Contains **3 directories** and **3 Python scripts**.

```
├── dataset_4l_man_april/
│   ├── model_perf/
│   ├── test/
│   └── train/
├── model_performance/
├── pix2pix_generator_4L/
├── evaluating_parallellized.py
├── inferring.py
└── training_&_inferencing_pix2pix_4l.ipynb
```

**Directory descriptions:**
- **model_perf/**: Contains sample model training performance across different epochs on one image, showing 50 images across various epochs as examples.
- **test/** and **train/**: Contains training data for the model, with paired images in PNG format.
- **model_performance/**: Contains performance metrics and evaluation results.
- **pix2pix_generator_4L/**: Contains the trained and saved model using epoch 19.

The Python scripts handle model training, inference, and evaluation processes.

### Phase4_comparison_analysis/
Contains **2 directories** and **3 Python scripts**.

```
├── analysis_results/
│   ├── Evaluation_Plots/
│   ├── Threshold_Optimization/
│   └── Visualizations/
├── final_data_for_models/
│   ├── test/
│   │   ├── subjects/
│   │   ├── VENT/
│   │   ├── WMH/
│   │   └── analysis_results_abWMH_vent_035_ep19/
│   └── train/
├── analysis_pipeline.py
├── segmentation_metrics.py
└── segmentation_visualization.py
```

**File descriptions:**
- **analysis_pipeline.py**: Performs analysis and comparison of our model's performance (from selected epoch 19) against other methods mentioned in the article for both ventricle segmentation and lesion segmentation scenarios.
- The other two Python files are utilized by the main analysis pipeline.

**Directory descriptions:**
- **analysis_results/**: Contains 3 subdirectories with visual results of these analyses, presenting results discussed in various sections of the article.
- **final_data_for_models/**: Contains input data for the analysis Python files, including raw and preprocessed data, segmentation results from each literature method for both scenarios, and a directory with analysis and performance comparison results.

## Root Files

### our_article_DOI.md
Contains the BibTeX citation format for referencing our article and this repository.

### repo_explanation.docx
This explanatory document that describes the repository structure and organization.

## Implementation Framework

The entire implementation is developed in **Python** using the following key technologies:
- **Deep Learning Framework**: TensorFlow/Keras for pix2pix implementation
- **Image Processing**: OpenCV, scikit-image
- **Data Handling**: NumPy, NIfTI processing libraries
- **Visualization**: Matplotlib, seaborn

## Usage Instructions

1. **Data Preprocessing**: Start with Phase1 to preprocess raw FLAIR images
2. **Data Preparation**: Use Phase2 to generate training data for the pix2pix model
3. **Model Training**: Execute Phase3 scripts to train and evaluate the model
4. **Comparative Analysis**: Run Phase4 scripts to compare with baseline methods

## Citation

Please refer to `our_article_DOI.md` for proper citation format when using this repository or referencing our work.