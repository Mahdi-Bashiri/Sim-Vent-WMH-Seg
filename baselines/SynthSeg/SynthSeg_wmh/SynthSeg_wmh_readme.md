# SynthSeg-WMH White Matter Hyperintensity Segmentation


## Overview
This document describes the procedure for segmenting White Matter Hyperintensities (WMH) using SynthSeg-WMH, a specialized variant of SynthSeg designed for WMH detection.


## Prerequisites
- SynthSeg-WMH tool installed and configured
- FSL (FMRIB Software Library) for image registration
- Input FLAIR images


## Workflow

### Step 1: WMH Segmentation
Execute WMH segmentation using the official SynthSeg-WMH script.


> **Critical**: The `resample` feature must be enabled to ensure proper registration of outcomes to the main input space.


### Step 2: WMH Label Consolidation
Merge all WMH-related labels in the output segmentation to create a unified WMH mask.


### Step 3: Registration to Original FLAIR Space
Register the WMH segmentation back to the original FLAIR space using FSL for consistent spatial alignment.


## Output Files
- Raw SynthSeg-WMH segmentation
- Consolidated WMH mask
- Final WMH mask in original FLAIR space


## Key Considerations
- **Spatial Consistency**: Resampling ensures alignment with input space
- **Label Accuracy**: Verify WMH label IDs match your SynthSeg-WMH version
- **Registration Quality**: Use appropriate interpolation methods for binary masks
- **Validation**: Always perform visual quality control of segmentation results


## Troubleshooting
- If registration fails, check image orientations and headers
- Ensure FLAIR images have sufficient contrast for WMH detection
- Verify SynthSeg-WMH model is appropriate for your data characteristics


## References
- [SynthSeg-WMH Documentation](https://github.com/BBillot/SynthSeg)
- [FSL Registration Guide](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT)
- [WMH Segmentation Best Practices](https://doi.org/10.1016/j.neuroimage.2019.116017)
