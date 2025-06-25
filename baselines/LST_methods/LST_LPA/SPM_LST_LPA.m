function [dest_path] = SPM_LST_LPA(flair_file, wmh_path, ID, spm_path)

% 

addpath(spm_path)

inputFiles = {};

[wmh_path, '\subj_', ID];

mkdir([wmh_path, '\subj_', ID])

dest_path = [wmh_path, '\subj_', ID];
dest = [dest_path, '\', ID, '.nii.gz'];
source = flair_file;

copyfile(source, dest);

gunzip(dest);

inputFiles{1} = [dest(1:end-3), ',1'];

% List of open inputs
inputs = cell(0, 1);

% Initialize the matlabbatch structure
matlabbatch = cell(1, numel(inputs));

matlabbatch{1}.spm.tools.LST.lpa.data_F2 = {inputFiles{1}};
matlabbatch{1}.spm.tools.LST.lpa.data_coreg = {''};
matlabbatch{1}.spm.tools.LST.lpa.html_report = 1;

    
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
        
    if file_name(end) == 't' | file_name(10) == 'F' | file_name(1) == 'm' | file_name(1) == '1'   
        source = [dest_path, '/', file_name];
        delete(source)
        
    elseif file_name(1:2) == 'pl' & file_name(end) == 'i'
        source = [dest_path, '/', file_name];
        gzip(source)
        delete(source)
          
    end
end

end
