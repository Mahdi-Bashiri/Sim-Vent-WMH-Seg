function [dest_path] = SPM_NORM(flair_file, vent_path, ID, spm_path)

% 

addpath(spm_path)

inputFiles = {};

[vent_path, '\subj_', ID];
mkdir([vent_path, '\subj_', ID])

dest_path = [vent_path, '\subj_', ID];
dest = [dest_path, '\', ID, '.nii.gz'];
source = flair_file;

copyfile(source, dest);
    
gunzip(dest);

inputFiles{1} = [dest(1:end-3), ',1'];

% List of open inputs
inputs = cell(0, 1);

% Initialize the matlabbatch structure
    
% normalizing:
matlabbatch = cell(1, numel(inputs));

matlabbatch{1}.spm.spatial.normalise.estwrite.subj.vol = {inputFiles{1}};
matlabbatch{1}.spm.spatial.normalise.estwrite.subj.resample = {inputFiles{1}};
matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.biasreg = 0.001;
matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.biasfwhm = 60;
matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.tpm = {[spm_path, '/tpm/TPM.nii']};
matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.affreg = 'mni';
matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.fwhm = 0;    
matlabbatch{1}.spm.spatial.normalise.estwrite.eoptions.samp = 3;
matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.bb = [-78 -112 -70
                                                             78 76 85];
matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.vox = [2 2 2];
matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.interp = 4;
matlabbatch{1}.spm.spatial.normalise.estwrite.woptions.prefix = 'mni';

spm('defaults', 'FMRI');
spm_jobman('run', matlabbatch, inputs{:});

% segmenting:
matlabbatch = cell(1, numel(inputs));
flair_m = inputFiles{1};
mni_flair = [flair_m(1:end-12), 'mni', flair_m(end-11:end)];

matlabbatch{1}.spm.spatial.preproc.channel.vols = {mni_flair};
matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;
matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60;
matlabbatch{1}.spm.spatial.preproc.channel.write = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(1).tpm = {[spm_path, '/tpm/TPM.nii,1']};
matlabbatch{1}.spm.spatial.preproc.tissue(1).ngaus = 1;
matlabbatch{1}.spm.spatial.preproc.tissue(1).native = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(1).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(2).tpm = {[spm_path, '/tpm/TPM.nii,2']};
matlabbatch{1}.spm.spatial.preproc.tissue(2).ngaus = 1;
matlabbatch{1}.spm.spatial.preproc.tissue(2).native = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(2).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(3).tpm = {[spm_path, '/tpm/TPM.nii,3']};
matlabbatch{1}.spm.spatial.preproc.tissue(3).ngaus = 2;
matlabbatch{1}.spm.spatial.preproc.tissue(3).native = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(3).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(4).tpm = {[spm_path, '/tpm/TPM.nii,4']};
matlabbatch{1}.spm.spatial.preproc.tissue(4).ngaus = 3;
matlabbatch{1}.spm.spatial.preproc.tissue(4).native = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(4).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(5).tpm = {[spm_path, '/tpm/TPM.nii,5']};
matlabbatch{1}.spm.spatial.preproc.tissue(5).ngaus = 4;
matlabbatch{1}.spm.spatial.preproc.tissue(5).native = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(5).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(6).tpm = {[spm_path, '/tpm/TPM.nii,6']};
matlabbatch{1}.spm.spatial.preproc.tissue(6).ngaus = 2;
matlabbatch{1}.spm.spatial.preproc.tissue(6).native = [0 0];
matlabbatch{1}.spm.spatial.preproc.tissue(6).warped = [0 0];
matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1;
matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0;
matlabbatch{1}.spm.spatial.preproc.warp.samp = 3;
matlabbatch{1}.spm.spatial.preproc.warp.write = [1 1];
matlabbatch{1}.spm.spatial.preproc.warp.vox = NaN;
matlabbatch{1}.spm.spatial.preproc.warp.bb = [NaN NaN NaN
                                              NaN NaN NaN];
spm('defaults', 'FMRI');
spm_jobman('run', matlabbatch, inputs{:});

% claeining

% finding flair image from a subject's files
files = dir([dest_path, '/']);
        
for k = 1:length(files)
    if length(files(k).name) < 6
        continue
    end
    file_name = files(k).name;
        
    if file_name(1) == 'y' | file_name(1) == 'i' | file_name(end-1:end) == 'gk' | file_name(1:2) == 'c4' | file_name(1:2) == 'c5' | file_name(end) == 't'           
        source = [dest_path, '/', file_name];
        delete(source)
    end
end
    
files = dir([dest_path, '/']);
for k = 1:length(files)
    if length(files(k).name) < 6
        continue
    end
    file_name = files(k).name;
    if file_name(end) == 'i'
        source = [dest_path, '/', file_name];
        gzip(source)
        delete(source)
    end
end


end