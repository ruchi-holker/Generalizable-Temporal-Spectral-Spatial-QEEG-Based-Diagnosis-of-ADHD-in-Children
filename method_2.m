function feat_vec = method_2(eeg_file)
% Toole, J. M., & Boylan, G. B. (2017). NEURAL: quantitative features for 
% newborn EEG using Matlab. arXiv preprint arXiv:1704.05694.
% Date: 01/04/2021

%%
% close all;
% clear all;
% clc;

%% CONSTANTS
include_bipolar = 1;

data_st.Fs = 256;
data_st.ch_labels_bi = ...
                {{'FP1', 'F7'}, {'F7', 'T7'}, {'T7', 'P7'}, {'P7' 'O1'},... 
                 {'FP1', 'F3'}, {'F3', 'C3'}, {'C3', 'P3'}, {'P3', 'O1'},...
                 {'FP2', 'F4'}, {'F4', 'C4'}, {'C4', 'P4'}, {'P4', 'O2'},...
                 {'FP2', 'F8'}, {'F8', 'T8'}, {'T8', 'P8'}, {'P8', 'O2'},...
                 {'FZ','CZ'}, {'CZ', 'PZ'}, ...
                 };
data_st.ch_labels_ref={'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', ...
                       'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'FZ',...
                       'CZ', 'PZ'};

%% Load data
% PATH = 'G:\EEG_Alcoholic_Controlled\Traning_DATA\AL\';
% eeg_data_ref = csvread([PATH, 'sam1_co2a0000364.rd.000.csv']).';
data_st.eeg_data_ref = csvread(eeg_file).';

%%
if(include_bipolar)
    % generate bi-polar montage:
    [data_st.eeg_data,data_st.ch_labels]=set_bi_montage(data_st.eeg_data_ref, ...
                                                      data_st.ch_labels_ref, ...
                                                      data_st.ch_labels_bi);
end

%% Feature extraction
[~, feat_st]=generate_all_features(data_st,[], [], true);

%% Feature vector
feat_vec = [];
for i=1:length(feat_st)
    feat_vec = [feat_vec; feat_st{i}(:)];
end
