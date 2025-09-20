[article_repo_structure_v2.md](https://github.com/user-attachments/files/22440261/article_repo_structure_v2.md)
# Repository Structure Documentation

## Overview

This document describes the implementation details and organizational structure of our research article, which focuses on the **simultaneous segmentation of brain ventricles and normal/abnormal brain white matter hyperintensities in FLAIR MRI images** with **cross-dataset validation**.

Detailed methodology and results are provided in the accompanying files and can be referenced in our published article. This document specifically outlines the implementation structure and organization of the Python-based codebase for developing and maintaining this GitHub repository.

**Version 2.0 Update**: This version incorporates cross-dataset validation on the MSSEG2016 public dataset, dual-model approach (local + fine-tuned), and comprehensive generalizability analysis.

## Repository Structure

The main directory contains **7 folders** and **2 files** organized as follows:

```
├── Article_Figures/
├── Auxiliary_compared_methods_details/
├── Phase1_data_preprocessing/
├── Phase2_data_preparation_for_model_training/
├── Phase3_model_training_and_inferencing_and_evaluation/
├── Phase4_comparison_analysis/
├── Cross_Dataset_Validation/                    # NEW
├── our_article_DOI.md
└── repo_explanation.docx
```

## Detailed Directory Structure

### Article_Figures/
Contains all figures used in the research article: **11 main figures**, **2 supplementary figures**, and **updated cross-dataset figures**.

```
├── Figure_1.png
├── Figure_2.pdf
├── Figure_2.png
├── Figure_3.pdf
├── Figure_4.png
├── Figure_5.png                                 # Fine-tuning performance (NEW)
├── Figure_6.png
├── Figure_7.png                                 # Updated confusion matrices
├── Figure_8.png                                 # Updated precision-recall curves
├── Figure_9.png                                 # Updated confusion matrices (cross-dataset)
├── Figure_10.png                                # Updated ROC curves
├── Figure_11.png                                # Updated confusion matrices
├── Figure_S1_wmh_metrics_vs_threshold.png
├── Figure_S2_wmh_default_vs_optimal_comparison.png
├── Table_1_ventricle_performance.png            # Updated performance tables (NEW)
├── Table_2_wmh_performance.png                  # Updated performance tables (NEW)
└── Table_3_computational_performance.png        # Computational comparison (NEW)
```

### Auxiliary_compared_methods_details/
Contains **7 subdirectories**: 6 directories for code and execution instructions for each literature method compared with our approach, plus directories containing raw sample data from **both datasets**.

```
├── Atlas_Matching/
├── Bianca/
├── LST_LGA/
├── LST_LPA/
├── raw_data/
│   ├── local_dataset/                          # Local clinical data samples
│   │   ├── subjects_flair/
│   │   └── subjects_t1/
│   └── msseg2016_dataset/                      # Public dataset samples (NEW)
│       ├── subjects_flair/
│       └── subjects_t1/
├── SynthSeg/
└── SynthSeg_wmh/
```

The `raw_data` directory now contains sample data from **5 patients from local dataset** and **3 patients from MSSEG2016** with both FLAIR and T1 sequences. Each method directory includes execution instructions for both datasets.

### Phase1_data_preprocessing/
Contains preprocessing scripts for raw input files from **both datasets**.

```
├── raw_data/
│   ├── local_dataset/                          # Local clinical samples
│   └── msseg2016_dataset/                      # Public dataset samples (NEW)
├── pre_processing_flair.py
└── cross_dataset_preprocessing.py               # Cross-dataset preprocessing (NEW)
```

The preprocessing scripts handle both local clinical data and MSSEG2016 data with appropriate normalization strategies for cross-dataset compatibility.

### Phase2_data_preparation_for_model_training/
Contains **6 data directories** and **2 Python scripts** for generating input images for the pix2pix (cGAN) model across datasets.

```
├── Original_FLAIRs_prep/
│   ├── local_dataset/                          # Local preprocessed data
│   └── msseg2016_dataset/                      # MSSEG2016 preprocessed data (NEW)
├── abWMH_manual_segmentations/
│   ├── local_dataset/                          # Local abnormal WMH masks
│   └── msseg2016_dataset/                      # MSSEG2016 abnormal WMH masks (NEW)
├── nWMH_manual_segmentations/
│   ├── local_dataset/                          # Local normal WMH masks
│   └── msseg2016_dataset/                      # MSSEG2016 normal WMH masks (NEW)
├── vent_manual_segmentations/
│   ├── local_dataset/                          # Local ventricle masks
│   └── msseg2016_dataset/                      # MSSEG2016 ventricle masks (NEW)
├── manual_4l_masks_april/
│   ├── local_dataset/                          # Local 4-level masks
│   └── msseg2016_dataset/                      # MSSEG2016 4-level masks (NEW)
├── fine_tuning_data/                           # Fine-tuning dataset (NEW)
├── generating_4L_masks.py
└── prepare_cross_dataset_training.py           # Cross-dataset preparation (NEW)
```

**Directory descriptions:**
- **Original_FLAIRs_prep/**: Contains preprocessed patient data files from both datasets. Each patient has one NIfTI file and one NPZ file containing the FLAIR image, brain mask, and mask metadata.
- **abWMH_manual_segmentations/**: Contains manual segmentation masks for abnormal lesions from both datasets.
- **nWMH_manual_segmentations/**: Contains manual segmentation masks for normal lesions from both datasets.
- **vent_manual_segmentations/**: Contains manual segmentation masks for brain ventricles from both datasets.
- **manual_4l_masks_april/**: Contains generated 4-level masks from the above segmentations for both datasets.
- **fine_tuning_data/**: Contains the 3 MSSEG2016 patients used for fine-tuning (NEW).

The Python scripts generate 4-level masks and create paired images for pix2pix model training across both datasets.

### Phase3_model_training_and_inferencing_and_evaluation/
Contains **5 directories** and **4 Python scripts** for training, fine-tuning, and evaluation.

```
├── dataset_4l_man_april/
│   ├── local_dataset/                          # Local training data
│   │   ├── model_perf/
│   │   ├── test/
│   │   └── train/
│   └── msseg2016_dataset/                      # MSSEG2016 data (NEW)
│       ├── fine_tuning/                        # 3 patients for fine-tuning
│       └── test/                               # 12 patients for testing
├── model_performance/
│   ├── local_training/                         # Local model performance
│   ├── fine_tuning/                            # Fine-tuning performance (NEW)
│   └── cross_dataset_evaluation/               # Cross-dataset metrics (NEW)
├── pix2pix_generator_4L_epoch19/               # Local optimal model (epoch 19)
├── pix2pix_generator_4L_epoch28_finetuned/     # Fine-tuned model (epoch 28) (NEW)
├── cross_dataset_models/                       # Model comparison storage (NEW)
├── evaluating_parallellized.py
├── inferring.py
├── fine_tuning_pipeline.py                     # Fine-tuning implementation (NEW)
└── training_&_inferencing_pix2pix_4l.ipynb
```

**Directory descriptions:**
- **model_perf/**: Contains sample model training performance across different epochs, now including fine-tuning progression.
- **test/** and **train/**: Contains training/testing data for both local and MSSEG2016 datasets.
- **model_performance/**: Contains performance metrics and evaluation results for both datasets.
- **pix2pix_generator_4L_epoch19/**: Contains the locally-trained optimal model.
- **pix2pix_generator_4L_epoch28_finetuned/**: Contains the fine-tuned model optimized for cross-dataset performance (NEW).
- **cross_dataset_models/**: Storage for model comparison and validation results (NEW).

The Python scripts handle model training, fine-tuning, inference, and evaluation processes across both datasets.

### Phase4_comparison_analysis/
Contains **3 directories** and **5 Python scripts** for comprehensive cross-dataset analysis.

```
├── analysis_results/
│   ├── local_dataset/                          # Local dataset analysis
│   │   ├── Evaluation_Plots/
│   │   ├── Threshold_Optimization/
│   │   └── Visualizations/
│   ├── msseg2016_dataset/                      # MSSEG2016 analysis (NEW)
│   │   ├── Evaluation_Plots/
│   │   ├── Threshold_Optimization/
│   │   └── Visualizations/
│   └── cross_dataset_comparison/               # Cross-dataset analysis (NEW)
│       ├── Performance_Comparison/
│       ├── Generalizability_Analysis/
│       └── Method_Consistency_Evaluation/
├── final_data_for_models/
│   ├── local_dataset/                          # Local test data
│   │   ├── test/
│   │   │   ├── subjects/
│   │   │   ├── VENT/
│   │   │   ├── WMH/
│   │   │   └── analysis_results_abWMH_vent_035_ep19/
│   │   └── train/
│   └── msseg2016_dataset/                      # MSSEG2016 test data (NEW)
│       └── test/
│           ├── subjects/
│           ├── VENT/
│           ├── WMH/
│           └── analysis_results_abWMH_vent_035_ep28/
├── baseline_results/                           # Baseline method outputs (NEW)
│   ├── local_dataset/
│   └── msseg2016_dataset/
├── analysis_pipeline.py
├── segmentation_metrics.py
├── segmentation_visualization.py
├── cross_dataset_analysis.py                   # Cross-dataset comparison (NEW)
└── baseline_comparison_pipeline.py             # Enhanced baseline comparison (NEW)
```

**File descriptions:**
- **analysis_pipeline.py**: Performs analysis and comparison of our model's performance against other methods for both datasets.
- **cross_dataset_analysis.py**: Comprehensive cross-dataset performance analysis and generalizability assessment (NEW).
- **baseline_comparison_pipeline.py**: Enhanced baseline comparison across both datasets (NEW).
- The other Python files provide utilities for the main analysis pipelines.

**Directory descriptions:**
- **analysis_results/**: Contains analysis results stratified by dataset type, with cross-dataset comparison results.
- **final_data_for_models/**: Contains input data for analysis, now organized by dataset source.
- **baseline_results/**: Contains outputs from all baseline methods on both datasets for fair comparison (NEW).

### Cross_Dataset_Validation/ (NEW)
Contains dedicated cross-dataset validation materials and analysis.

```
├── validation_protocols/
│   ├── local_to_msseg_validation.py
│   ├── msseg_to_local_validation.py
│   └── cross_dataset_metrics.py
├── fine_tuning_analysis/
│   ├── adaptation_curves/                      # Fine-tuning progression plots
│   ├── performance_evolution/                  # Epoch-by-epoch performance
│   └── optimal_epoch_selection/               # Epoch 28 selection rationale
├── generalizability_studies/
│   ├── scanner_variability_analysis/
│   ├── protocol_impact_assessment/
│   └── population_differences_study/
├── deployment_guidelines/
│   ├── site_qualification_protocol.md
│   ├── model_selection_flowchart.py
│   └── performance_monitoring_tools/
└── validation_results/
    ├── cross_dataset_performance_summary.json
    ├── model_comparison_report.pdf
    └── clinical_validation_outcomes.csv
```

**Purpose**: This directory contains all materials related to cross-dataset validation, including protocols, analysis tools, and results that demonstrate the generalizability of our approach.

## Root Files

### our_article_DOI.md
Contains the BibTeX citation format for referencing our article and this repository.

### repo_explanation.docx
This explanatory document that describes the repository structure and organization, updated for version 2.0.

## Implementation Framework

The entire implementation is developed in **Python** using the following key technologies:
- **Deep Learning Framework**: TensorFlow/Keras for pix2pix implementation
- **Image Processing**: OpenCV, scikit-image
- **Data Handling**: NumPy, NIfTI processing libraries (nibabel)
- **Visualization**: Matplotlib, seaborn
- **Statistical Analysis**: SciPy, scikit-learn
- **Cross-dataset Tools**: Custom validation and comparison utilities

## Usage Instructions

### Standard Workflow
1. **Data Preprocessing**: Start with Phase1 to preprocess raw FLAIR images from both datasets
2. **Data Preparation**: Use Phase2 to generate training data for the pix2pix model
3. **Model Training/Fine-tuning**: Execute Phase3 scripts to train and fine-tune models
4. **Comparative Analysis**: Run Phase4 scripts to compare with baseline methods across datasets

### Cross-Dataset Validation Workflow (NEW)
1. **Initial Validation**: Use Cross_Dataset_Validation/ protocols to assess generalizability
2. **Model Selection**: Choose appropriate model (epoch 19 vs. 28) based on data characteristics  
3. **Site-Specific Validation**: Follow deployment guidelines for new sites
4. **Performance Monitoring**: Use monitoring tools for ongoing quality assurance

## Key Performance Metrics

### Local Dataset (Primary Validation)
- **Ventricle Segmentation**: Dice 0.801 ± 0.025, HD95 18.46 ± 7.1mm
- **WMH Segmentation**: Dice 0.624 ± 0.061, Precision 0.755 ± 0.161
- **Processing Time**: ~4 seconds per case
- **Clinical Accuracy**: 92% (ventricles), 81% valuable (WMH classification)

### MSSEG2016 Dataset (Cross-Dataset Validation)
- **Ventricle Segmentation**: Dice 0.798 ± 0.101, HD95 24.39 ± 20.03mm
- **WMH Segmentation**: Dice 0.484 ± 0.153, Precision 0.602 ± 0.249
- **Processing Time**: ~4 seconds per case (maintained efficiency)
- **Model**: Fine-tuned epoch 28 for optimal cross-dataset performance

### Computational Advantage
- **Speed Improvement**: 18-36x faster than baseline methods
- **Resource Requirements**: 15% CPU, ~1GB RAM, 80% GPU utilization
- **Hardware Accessibility**: Compatible with standard clinical workstations

## Model Variants

### Local Model (Epoch 19)
- **Training**: 70% local dataset (210 patients)
- **Optimization**: Optimal for single-site clinical deployment
- **Best For**: Clinical data similar to training set (1.5T TOSHIBA Vantage)
- **Performance**: Highest accuracy on local dataset

### Fine-tuned Model (Epoch 28)
- **Base**: Epoch 19 model fine-tuned on MSSEG2016
- **Fine-tuning Data**: 3 MSSEG2016 patients
- **Optimization**: Cross-dataset generalization
- **Best For**: Multi-site applications, research datasets, protocol variations
- **Performance**: Balanced accuracy across diverse datasets

## Baseline Comparison Results

| Method | Ventricle Dice | WMH Dice | Processing Time | Cross-Dataset Stability |
|--------|----------------|----------|-----------------|------------------------|
| **Our Method** | **0.801±0.025** | **0.624±0.061** | **4 sec** | **Excellent** |
| SynthSeg | 0.751±0.103 | - | 124 sec | Moderate |
| Atlas Matching | 0.742±0.065 | - | 115 sec | Good |
| BIANCA | - | 0.268±0.095 | 11 sec | Poor |
| LST-LPA | - | 0.509±0.098 | 72 sec | Moderate |
| LST-LGA | - | 0.156±0.082 | 147 sec | Poor |
| WMH-SynthSeg | - | 0.376±0.120 | 78 sec | Moderate |

## Documentation Structure

The repository includes comprehensive documentation:
- **README.md**: Overview and quick start guide
- **INSTALLATION.md**: Detailed setup instructions
- **USAGE.md**: Step-by-step usage tutorial with cross-dataset guidance
- **CLINICAL_GUIDE.md**: Clinical interpretation and validation guidelines
- **CROSS_DATASET.md**: Cross-dataset validation methodology and results (NEW)
- **BASELINES.md**: Detailed baseline methods implementation guide (NEW)
- **CONTRIBUTING.md**: Development and contribution guidelines
- **TROUBLESHOOTING.md**: Common issues and solutions

## Version History

### Version 2.0 (Current)
- ✅ Cross-dataset validation on MSSEG2016
- ✅ Fine-tuning methodology and results
- ✅ Dual-model approach (local + fine-tuned)
- ✅ Enhanced baseline comparisons across datasets
- ✅ Comprehensive generalizability analysis
- ✅ Clinical deployment guidelines for multi-site applications

### Version 1.0
- Initial implementation with local dataset validation
- Core pix2pix architecture and training pipeline
- Baseline method comparisons on single dataset
- Clinical validation and expert assessment

## Citation

Please refer to `our_article_DOI.md` for proper citation format when using this repository or referencing our work. The cross-dataset validation results provide strong evidence for the clinical utility and generalizability of our approach across diverse healthcare environments.
