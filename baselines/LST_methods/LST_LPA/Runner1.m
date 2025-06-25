% Running this m-file will start the LPA implementation part of the paper2
% It is all fully automatic once you only insert needed paths at the first section below.
%

clear all
close all
clc
% Please, identify the path of the installed SPM toolbox for MATLAB software
spm_path ='C:\Users\SAI\Documents\MATLAB\spm12';

% Please, identify the main data paths. {a folder that contains subject folders that each has a nifti FLAIR file}
flair_path = 'C:\Users\DR-Shamsi-LAB\Desktop\subjects';

% Please, define the saving paths for the following parameters
% {for 3 tissue segmentation, for found WMH, for ventricles segmentation}
wmh_path = 'C:\Users\DR-Shamsi-LAB\Desktop\flair_lpa';

%%
tic;  % Start timing

subjects_fold = dir(flair_path);
j = 1;
for i = 1:length(subjects_fold)

    if length(subjects_fold(i).name) < 6
        continue
    end

%     if j == 1
%         if subjects_fold(i).name == 'case_126494'
%             j = 0;
%         else
%             continue
%         end
% 
%     end



    % finding flair image from a subjects files
    subjects = dir([subjects_fold(i).folder, '/', subjects_fold(i).name]);
    for k = 1:length(subjects)
        if length(subjects(k).name) < 10
            continue
        end
        file_name = subjects(k).name;
        if file_name(end-5) == 'n' % file_name(end-7) == 'n' % & file_name(8:14) == 'AXFLAIR' 
            subj_name = subjects(k).name;
            subj_fold = subjects(k).folder;
            break
        end
    end
    flair_file = [subj_fold, '/', subj_name];

    dest_path = SPM_LST_LPA(flair_file, wmh_path, subj_name(1:6), spm_path);
    
    subject_start_time = toc;
    disp(['Elapsed time for subject: ', num2str(subject_start_time), ' seconds']);

end


total_elapsed_time = toc;
disp(['Total elapsed time: ', num2str(total_elapsed_time), ' seconds']);




