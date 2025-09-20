[Uploading USAGE_v2.md…]()
# Usage Tutorial

## Overview

This tutorial provides step-by-step instructions for using the simultaneous brain ventricle and white matter hyperintensity segmentation framework. The pipeline consists of four main phases:

1. **Phase 1**: Data Preprocessing
2. **Phase 2**: Data Preparation for Model Training
3. **Phase 3**: Model Training/Inference/Evaluation
4. **Phase 4**: Comparison Analysis

## Quick Start (Inference Only)

### Prerequisites
- FLAIR MRI images in NIfTI format (.nii or .nii.gz)
- Completed installation (see Installation Guide)

### Basic Usage

```python
import sys
sys.path.append('src')

from preprocessing.pre_processing_flair import main as preprocess_flair
from models.inferring import main as run_inference

# Step 1: Preprocess FLAIR image
input_path = "path/to/your/flair.nii.gz"
output_dir = "path/to/preprocessed/output"

preprocessed_image, brain_info = preprocess_flair(input_path, output_dir)

# Step 2: Run inference with appropriate model
# For data similar to local clinical dataset
model_path = "src/models/pix2pix_generator_4L_epoch19"

# For cross-dataset applications or MSSEG-type data
# model_path = "src/models/pix2pix_generator_4L_epoch28_finetuned"

results_dir = "path/to/results"

segmentation_results = run_inference(
    preprocessed_image,
    brain_info, 
    model_path, 
    results_dir
)
```

## Model Selection Guide

### Available Pre-trained Models

1. **Local Model (Epoch 19)**: `pix2pix_generator_4L_epoch19/`
   - Trained on 300 MS patients from clinical dataset
   - Best for: Clinical FLAIR images from 1.5T scanners
   - Performance: Dice 0.801 (ventricles), 0.624 (WMH)

2. **Fine-tuned Model (Epoch 28)**: `pix2pix_generator_4L_epoch28_finetuned/`
   - Fine-tuned on MSSEG2016 public dataset
   - Best for: Cross-dataset applications, research datasets
   - Performance: Dice 0.798 (ventricles), 0.484 (WMH) on MSSEG2016

### Model Selection Criteria

```python
def select_optimal_model(scanner_type, acquisition_protocol, dataset_origin):
    """
    Select the optimal model based on data characteristics
    
    Args:
        scanner_type: '1.5T', '3T', or 'unknown'
        acquisition_protocol: 'clinical', 'research', 'MSSEG-like'
        dataset_origin: 'single-site', 'multi-site', 'public'
    
    Returns:
        Recommended model path
    """
    
    if dataset_origin == 'single-site' and scanner_type == '1.5T':
        return "pix2pix_generator_4L_epoch19"
    elif dataset_origin in ['multi-site', 'public'] or acquisition_protocol == 'research':
        return "pix2pix_generator_4L_epoch28_finetuned"
    else:
        # Default to local model, can try both if uncertain
        return "pix2pix_generator_4L_epoch19"
```

## Detailed Phase-by-Phase Tutorial

### Phase 1: Data Preprocessing

**Purpose**: Clean and normalize raw FLAIR MRI images for processing.

**Location**: `Phase1_data_preprocessing/`

#### Step 1.1: Prepare Raw Data

```bash
# Place your FLAIR images in the raw_data directory
mkdir -p Phase1_data_preprocessing/raw_data/subjects_flair
cp your_flair_images/* Phase1_data_preprocessing/raw_data/subjects_flair/
```

#### Step 1.2: Run Preprocessing

```python
# Navigate to Phase1 directory
cd Phase1_data_preprocessing

# Run preprocessing script
python pre_processing_flair.py
```

**What this does**:
- Noise reduction using median and Gaussian filters
- Brain extraction with morphology-based approach
- Intensity normalization (slice-based adaptive)
- Generates .npz files with brain masks and metadata

**Expected Output**:
```
preprocessed_output/
├── patient_001.nii.gz     # Preprocessed FLAIR
├── patient_001.npz        # Brain mask + metadata
├── patient_002.nii.gz
├── patient_002.npz
└── ...
```

### Phase 2: Data Preparation for Model Training

**Purpose**: Generate paired images for pix2pix model training (only needed for training new models).

**Location**: `Phase2_data_preparation_for_model_training/`

#### Step 2.1: Organize Manual Segmentations (Training Only)

If you have ground truth segmentations:

```bash
# Copy segmentations to appropriate directories
cp ventricle_masks/* vent_manual_segmentations/
cp normal_wmh_masks/* nWMH_manual_segmentations/
cp abnormal_wmh_masks/* abWMH_manual_segmentations/
cp preprocessed_flairs/* Original_FLAIRs_prep/
```

#### Step 2.2: Generate Training Data

```python
# Generate 4-level masks and paired images
python generating_4L_masks.py
```

**Output**: Creates paired 256×512 composite images for pix2pix training.

### Phase 3: Model Training, Inference, and Evaluation

**Purpose**: Train models, run inference, and evaluate performance.

**Location**: `Phase3_model_training_and_inferencing_and_evaluation/`

#### Step 3.1: Inference with Pre-trained Models

```python
# Load the Jupyter notebook for interactive use
jupyter notebook training_&_inferencing_pix2pix_4l.ipynb
```

Or use the Python script:

```python
# Run inference with local model
python inferring.py --input_dir "path/to/preprocessed/data" \
                   --model_path "pix2pix_generator_4L_epoch19" \
                   --output_dir "path/to/results"

# Run inference with fine-tuned model for cross-dataset
python inferring.py --input_dir "path/to/preprocessed/data" \
                   --model_path "pix2pix_generator_4L_epoch28_finetuned" \
                   --output_dir "path/to/results_finetuned"
```

#### Step 3.2: Model Training and Fine-tuning

**Initial Training** (50 epochs):
```python
# If training a new model from scratch
python training_&_inferencing_pix2pix_4l.ipynb
```

**Fine-tuning Protocol** (for cross-dataset adaptation):
```python
# Fine-tune pre-trained model on new dataset
# Load epoch 19 model and continue training to epoch 29
fine_tune_model(
    base_model_path="pix2pix_generator_4L_epoch19",
    new_dataset_path="path/to/new/training/data",
    start_epoch=19,
    end_epoch=29,
    fine_tuning_patients=3  # Small subset for adaptation
)
```

**Training Parameters**:
- **Initial Training**: Batch size: 1, Epochs: 50, Optimizer: Adam (lr=0.0002)
- **Fine-tuning**: Batch size: 1, Epochs: 19-29, Same optimizer
- **Loss**: Combined adversarial + L1 loss (λ_adv=1, λ_L1=100)

#### Step 3.3: Cross-Dataset Evaluation

```python
# Evaluate on local dataset
python evaluating_parallellized.py --predictions_dir "path/to/local/predictions" \
                                  --ground_truth_dir "path/to/local/ground_truth" \
                                  --output_dir "local_evaluation_results"

# Evaluate on public dataset
python evaluating_parallellized.py --predictions_dir "path/to/msseg/predictions" \
                                  --ground_truth_dir "path/to/msseg/ground_truth" \
                                  --output_dir "msseg_evaluation_results"
```

**Evaluation Metrics**:
- Dice Coefficient
- Hausdorff Distance (HD95)
- Precision/Recall
- AUC-PR
- Confusion Matrix Analysis

### Phase 4: Comparison Analysis

**Purpose**: Compare results against baseline methods across datasets.

**Location**: `Phase4_comparison_analysis/`

#### Step 4.1: Prepare Comparison Data

```bash
# Ensure your data is in the correct format
ls final_data_for_models/test/subjects/  # Local dataset test subjects
ls final_data_for_models/msseg2016/     # Public dataset (if available)
```

#### Step 4.2: Run Baseline Methods on Both Datasets

Use the scripts in `baselines/` directory:

```bash
# Example: Run SynthSeg on both datasets
cd baselines/SynthSeg

# Local dataset
python run_synthseg.py --input_dir "../../test_data_local" --output_dir "results_local"

# Public dataset
python run_synthseg.py --input_dir "../../test_data_msseg" --output_dir "results_msseg"

# Similar for other baselines (BIANCA, LST methods, etc.)
```

#### Step 4.3: Comprehensive Cross-Dataset Analysis

```python
# Run analysis for both datasets
cd Phase4_comparison_analysis

# Local dataset analysis
python analysis_pipeline.py --our_results "path/to/our/local/results" \
                           --baseline_results "path/to/baseline/local/results" \
                           --dataset_type "local" \
                           --output_dir "analysis_results/local_dataset"

# Public dataset analysis
python analysis_pipeline.py --our_results "path/to/our/msseg/results" \
                           --baseline_results "path/to/baseline/msseg/results" \
                           --dataset_type "msseg2016" \
                           --output_dir "analysis_results/msseg2016_dataset"

# Cross-dataset comparison
python cross_dataset_analysis.py --local_results "analysis_results/local_dataset" \
                                --msseg_results "analysis_results/msseg2016_dataset" \
                                --output_dir "analysis_results/cross_validation"
```

## Input Data Requirements

### FLAIR MRI Specifications
- **Format**: NIfTI (.nii or .nii.gz)
- **Orientation**: Standard radiological orientation
- **Resolution**: Compatible with anisotropic voxels (tested on 0.9 × 0.9 × 6 mm)
- **Scanner**: Any 1.5T or 3T scanner (primary validation on 1.5T TOSHIBA Vantage)
- **Cross-dataset**: Validated on MSSEG2016 (3 different centers)

### File Naming Convention
```
patient_001_FLAIR.nii.gz
patient_002_FLAIR.nii.gz
...
```

### Dataset-Specific Considerations

**Local Clinical Data**:
- Single scanner type (1.5T TOSHIBA Vantage)
- Consistent acquisition parameters
- Expert annotations for 4 classes
- Optimal model: Epoch 19

**Multi-site/Research Data**:
- Multiple scanner types and centers
- Varying acquisition parameters
- May require fine-tuning for optimal performance
- Optimal model: Epoch 28 (fine-tuned)

## Output Interpretation

### Segmentation Results

The model outputs a 4-class segmentation:

- **Class 0**: Background
- **Class 1**: Brain ventricles
- **Class 2**: Normal white matter hyperintensities
- **Class 3**: Abnormal white matter hyperintensities (MS lesions)

### Cross-Dataset Performance Expectations

**Local Dataset Performance**:
- Ventricle Dice: 0.801 ± 0.025
- WMH Dice: 0.624 ± 0.061
- Processing time: ~4 seconds

**Public Dataset Performance (MSSEG2016)**:
- Ventricle Dice: 0.798 ± 0.101
- WMH Dice: 0.484 ± 0.153
- Processing time: ~4 seconds

**Performance Insights**:
- Ventricle segmentation: Consistent across datasets
- WMH segmentation: More variable, dataset-dependent
- Speed advantage: Maintained across all datasets

### Result Files

```
results/
├── local_dataset/
│   ├── patient_001/
│   │   ├── preprocessed_flair.nii.gz
│   │   ├── segmentation_4class.nii.gz
│   │   ├── ventricles_mask.nii.gz
│   │   ├── normal_wmh_mask.nii.gz
│   │   ├── abnormal_wmh_mask.nii.gz
│   │   └── performance_metrics.json
│   └── ...
├── msseg2016_dataset/
│   └── [similar structure]
└── cross_validation/
    ├── performance_comparison.csv
    ├── generalizability_analysis.json
    └── visualization/
```

## Processing Time

**Expected processing times per patient**:
- Preprocessing: ~2.4 seconds
- Inference: ~1.6 seconds
- Total: ~4 seconds
- **Consistency**: Maintained across both datasets

## Clinical Interpretation

### Ventricle Segmentation
- **Volume measurements**: Quantify ventricular enlargement
- **Cross-dataset consistency**: Reliable across different scanners
- **Asymmetry assessment**: Compare left vs right ventricles
- **Longitudinal tracking**: Monitor atrophy progression

### White Matter Hyperintensity Classification
- **Normal WMH**: Age-related changes, periventricular caps
- **Abnormal WMH**: MS lesions, pathological hyperintensities
- **Dataset considerations**: Performance may vary with acquisition protocol
- **Clinical significance**: Distinguish pathology from normal aging

### Cross-Dataset Considerations
- **Scanner variability**: Fine-tuned model better for multi-site data
- **Acquisition differences**: May affect WMH detection sensitivity
- **Validation importance**: Always validate on representative sample

## Performance Monitoring and Validation

### Cross-Dataset Validation Protocol

```python
def cross_dataset_validation(local_model, finetuned_model, test_datasets):
    """
    Compare model performance across different datasets
    """
    results = {}
    
    for dataset_name, dataset in test_datasets.items():
        print(f"Evaluating on {dataset_name}...")
        
        # Test local model
        local_results = evaluate_model(local_model, dataset)
        
        # Test fine-tuned model
        finetuned_results = evaluate_model(finetuned_model, dataset)
        
        results[dataset_name] = {
            'local_model': local_results,
            'finetuned_model': finetuned_results,
            'best_model': 'finetuned' if finetuned_results['dice'] > local_results['dice'] else 'local'
        }
    
    return results
```

### Quality Assessment Metrics

```python
def assess_cross_dataset_quality(results_dict):
    """
    Assess segmentation quality across datasets
    """
    quality_metrics = {}
    
    for dataset, results in results_dict.items():
        quality_metrics[dataset] = {
            'dice_consistency': np.std([r['dice'] for r in results]),
            'boundary_accuracy': np.mean([r['hd95'] for r in results]),
            'clinical_acceptability': calculate_clinical_score(results)
        }
    
    return quality_metrics
```

## Advanced Usage

### Batch Processing with Dataset-Aware Model Selection

```python
def intelligent_batch_processing(input_files, output_dir):
    """
    Process multiple files with automatic model selection
    """
    
    for input_file in input_files:
        # Analyze input characteristics
        dataset_type = detect_dataset_characteristics(input_file)
        
        # Select appropriate model
        if dataset_type == 'clinical_single_site':
            model_path = "pix2pix_generator_4L_epoch19"
        else:
            model_path = "pix2pix_generator_4L_epoch28_finetuned"
        
        # Process with selected model
        result = process_single_file(input_file, model_path, output_dir)
        
        print(f"Processed {input_file} with {model_path}")
```

### Visualization with Cross-Dataset Comparison

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_cross_dataset_results(local_results, msseg_results, slice_idx=None):
    """Visualize results from both datasets for comparison"""
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Local dataset results (top row)
    axes[0, 0].imshow(local_results['flair'][:, :, slice_idx], cmap='gray')
    axes[0, 0].set_title('Local Dataset FLAIR')
    
    axes[0, 1].imshow(local_results['segmentation'][:, :, slice_idx], cmap='viridis')
    axes[0, 1].set_title('Local Dataset Segmentation')
    
    axes[0, 2].imshow(local_results['ventricles'][:, :, slice_idx], cmap='Blues')
    axes[0, 2].set_title('Local Dataset Ventricles')
    
    axes[0, 3].imshow(local_results['wmh'][:, :, slice_idx], cmap='Reds')
    axes[0, 3].set_title('Local Dataset WMH')
    
    # MSSEG results (bottom row)
    axes[1, 0].imshow(msseg_results['flair'][:, :, slice_idx], cmap='gray')
    axes[1, 0].set_title('MSSEG2016 FLAIR')
    
    axes[1, 1].imshow(msseg_results['segmentation'][:, :, slice_idx], cmap='viridis')
    axes[1, 1].set_title('MSSEG2016 Segmentation')
    
    axes[1, 2].imshow(msseg_results['ventricles'][:, :, slice_idx], cmap='Blues')
    axes[1, 2].set_title('MSSEG2016 Ventricles')
    
    axes[1, 3].imshow(msseg_results['wmh'][:, :, slice_idx], cmap='Reds')
    axes[1, 3].set_title('MSSEG2016 WMH')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
```

## Troubleshooting Cross-Dataset Issues

### Common Cross-Dataset Problems

1. **Performance Drop on New Dataset**:
   - Try fine-tuned model (epoch 28)
   - Check preprocessing consistency
   - Validate input data quality

2. **Inconsistent Segmentation Results**:
   - Compare acquisition parameters
   - Check intensity normalization
   - Consider dataset-specific fine-tuning

3. **Model Selection Uncertainty**:
   - Test both models on sample data
   - Compare performance metrics
   - Use clinical validation when possible

### Dataset-Specific Optimization

```python
def optimize_for_dataset(model_path, dataset_samples, validation_samples):
    """
    Fine-tune model for specific dataset characteristics
    """
    
    # Analyze dataset characteristics
    dataset_stats = analyze_dataset_properties(dataset_samples)
    
    # Determine if fine-tuning is needed
    baseline_performance = evaluate_model(model_path, validation_samples)
    
    if baseline_performance['dice'] < 0.7:  # Threshold for fine-tuning
        print("Performance below threshold, fine-tuning recommended")
        fine_tuned_model = fine_tune_model(model_path, dataset_samples)
        return fine_tuned_model
    else:
        print("Baseline performance acceptable")
        return model_path
```

## Next Steps

After completing this tutorial:

1. **Analyze Cross-Dataset Results**: Compare performance across datasets
2. **Clinical Validation**: Have experts review both local and external data results
3. **Method Selection**: Choose optimal model based on your data characteristics
4. **Performance Monitoring**: Track quality across different data sources
5. **Integration**: Implement dataset-aware processing in clinical workflows

## Support

For additional help:
- Check the **Troubleshooting Guide** for dataset-specific issues
- Review cross-validation results in the `results/cross_validation/` directory
- Create GitHub issues for cross-dataset performance problems
- Consult the Clinical Guide for interpretation across different scanner types
