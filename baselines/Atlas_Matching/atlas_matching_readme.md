# Atlas Matching Segmentation

## Overview
This document describes the atlas-based segmentation approach using SPM (Statistical Parametric Mapping) normalization and segmentation tools, followed by atlas matching for ventricular segmentation.

## Prerequisites
- SPM12 (Statistical Parametric Mapping) software
- MATLAB or SPM Standalone
- MNI brain template
- MNI ventricular atlas template
- Input FLAIR images

## Workflow

### Step 1: Spatial Normalization to MNI Space
Register (normalize) the FLAIR data to the MNI (Montreal Neurological Institute) standard space using SPM normalization.


### Step 2: Tissue Segmentation
Perform tissue segmentation using SPM to generate CSF probability maps.


### Step 3: Atlas Matching
The atlas matching script requires three input files:

1. **CSF Probability Map**: `c3mni[filename].nii` (generated from Step 2)
2. **Normalized FLAIR Image**: `mni[filename].nii` (from Step 1)
3. **MNI Ventricular Template**: Pre-defined ventricular atlas in MNI space


## Required Files and Naming Convention

### Input Files:
- Original FLAIR image
- SPM tissue probability map template
- MNI ventricular atlas template

### Generated Files:
- Normalized FLAIR (prefix 'mni' from SPM)
- CSF probability map (prefix 'c3' from SPM segmentation)
- Final segmentation output


## Key Steps Summary

1. **Normalization**: Transform FLAIR to MNI space → `mni[filename].nii`
2. **Segmentation**: Generate tissue probability maps → `c3mni[filename].nii`
3. **Atlas Matching**: Combine CSF map with ventricular atlas template


## Parameters and Settings
- **Voxel Size**: 2×2×2 mm (standard MNI resolution)
- **Bounding Box**: [-78 -112 -70; 78 76 85] mm
- **CSF Threshold**: Typically 0.5 (adjust based on data characteristics)
- **Interpolation**: 4th degree B-spline for normalization


## Troubleshooting
- Ensure proper SPM installation and path configuration
- Check image orientations and voxel sizes
- Verify atlas template matches MNI space
- Adjust CSF threshold if segmentation is too conservative/liberal


## References
- [SPM12 Manual](https://www.fil.ion.ucl.ac.uk/spm/doc/manual.pdf)
- [MNI Templates](https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009)
- [Atlas-Based Segmentation Methods](https://doi.org/10.1016/j.neuroimage.2004.07.026)
