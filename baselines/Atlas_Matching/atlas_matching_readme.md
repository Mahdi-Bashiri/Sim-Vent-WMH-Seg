For Atlas Matching:
We first register (normalize) the FLAIR data on the MNI space, by SPM-Norm.
Then, we segment the outcome by SPM-Seg to reach csf map or mask.
The final file will have a prefix like c3mni... .
This file alongside the main FLAIR file and the MNI template of brain ventricles will be the inputs for Atlas Matching script.