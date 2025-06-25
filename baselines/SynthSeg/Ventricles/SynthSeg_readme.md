# SynthSeg Ventricular Segmentation

## Overview
This document outlines the process for performing ventricular segmentation using SynthSeg, a deep learning-based brain segmentation tool.

## Prerequisites
- SynthSeg tool installed and configured
- FSL (FMRIB Software Library) for image registration
- Input FLAIR images

## Workflow

### Step 1: Brain Segmentation
Perform brain segmentation following the official SynthSeg script instructions.


> **Important**: Enable the `resample` feature to ensure proper registration of outcomes to the main input space.


### Step 2: Ventricle Label Merging
Merge all ventricle-related labels in the output segmentation file to create a unified ventricular mask.


### Step 3: Registration to FLAIR Space
Use FSL to register the segmentation output back to the original FLAIR space for fair comparisons and analyses.


## Output Files
- Raw SynthSeg segmentation
- Merged ventricular mask
- Final ventricular mask in FLAIR space

## Notes
- The resampling feature is crucial for proper spatial alignment
- Ventricle label IDs may vary depending on SynthSeg version
- Use nearest neighbor interpolation to preserve label integrity during registration

## References
- [SynthSeg Official Repository](https://github.com/BBillot/SynthSeg)
- [FSL Documentation](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/)
