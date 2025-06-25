For SynthSeg ventricular segmentation:
We first perform brain segmentation as instructed by the official script. Of course, we enable resample feature of the script, necessaary for registering the outcomes on the main input space.
Then, we merge any ventricle-related labels in the outcome segmentation file to reach a unified ventricular mask of the brain.
Moreover, by FSL, we register back the desired outcome on the main FLAIR space for further fair comparisons and analyses.