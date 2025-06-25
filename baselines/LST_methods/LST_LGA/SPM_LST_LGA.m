function [dest_path] = SPM_LST_LGA(flair_file, t1_file, wmh_path, ID, spm_path)
%
addpath(spm_path)
inputFiles = {};

% Create a subject folder under wmh_path
[wmh_path, '\subj_', ID];
mkdir([wmh_path, '\subj_', ID])
dest_path = [wmh_path, '\subj_', ID];

% Copy and unzip the FLAIR file into the subject folder
dest = [dest_path, '\', ID, '.nii.gz'];
source = flair_file;
copyfile(source, dest);
gunzip(dest);
inputFiles{1} = [dest(1:end-3), ',1'];

% Copy and unzip the T1 file into the subject folder
dest = [dest_path, '\', ID, '_t.nii.gz'];
source = t1_file;
copyfile(source, dest);
gunzip(dest);
inputFiles{2} = [dest(1:end-3), ',1'];

% List of open inputs (empty in this case)
inputs = cell(0, 1);

% Initialize the matlabbatch structure for LST-LGA
matlabbatch = cell(1,1);  % one job item
matlabbatch{1}.spm.tools.LST.lga.data_F2 = {inputFiles{1}};
matlabbatch{1}.spm.tools.LST.lga.data_T1 = {inputFiles{2}};
matlabbatch{1}.spm.tools.LST.lga.opts_lga.initial = 0.3;
matlabbatch{1}.spm.tools.LST.lga.opts_lga.mrf = 1;
matlabbatch{1}.spm.tools.LST.lga.opts_lga.maxiter = 100;
matlabbatch{1}.spm.tools.LST.lga.html_report = 1;


% Set defaults and run the job
spm('defaults', 'FMRI');
spm_jobman('run', matlabbatch, inputs{:});

% Cleaning: find and delete intermediate FLAIR files
files = dir([dest_path, '/']);
for k = 1:length(files)
    if length(files(k).name) < 6
        continue
    end
    file_name = files(k).name;
    if file_name(end) == 't' || file_name(10) == 'F' || file_name(1) == 'm' || file_name(1) == '1'
        source = [dest_path, '/', file_name];
        delete(source)
    elseif startsWith(file_name, 'pl') && file_name(end) == 'i'
        source = [dest_path, '/', file_name];
        gzip(source)
        delete(source)
    end
end
end
