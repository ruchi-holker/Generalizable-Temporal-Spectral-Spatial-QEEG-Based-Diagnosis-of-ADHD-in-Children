% Date: 11/08/2021

%%
close all
clear
clc

%% Constants
fs = 256;   % Sampling frequency
NFFT = 512;
bw = 10;
low_freq = 0;
high_freq = 100;
K = fix(NFFT/2)+1;

%% Filter bank demo
fbank = filter_bank(NFFT, fs, bw, low_freq, high_freq, 'flat');
plot((0:size(fbank,2)-1)/NFFT*fs, fbank.')
xlabel('Frequency (Hz)')
ylabel('Amplitude')

%% Load AD Train
path = '..\Data_ADHD_New';
CTRL_files = dir(fullfile(path, 'CTRL'));
CTRL_files = CTRL_files(3:end-1);
CTRL_files = CTRL_files(1:300);

%% Load ADD 
path = '..\Data_ADHD_New';
ADD_files = dir(fullfile(path, 'ADD'));
ADD_files = ADD_files(3:end-1);
ADD_files = ADD_files(1:300);

%% Load ADHD Train
ADHD_files = dir(fullfile(path, 'ADHD'));
ADHD_files = ADHD_files(3:end-1);
ADHD_files = ADHD_files(1:300);

%% Feature extraction
[p, q] = rat(fs/256);  % Resampling factor calculation
EEG = cell(9, length(CTRL_files)+length(ADD_files)+length(ADHD_files));
labels = zeros(1, length(CTRL_files)+length(ADD_files)+length(ADHD_files));
features = zeros(K, 385, length(CTRL_files));

%% CTRL file processing
for f = 1:length(CTRL_files)
    fcount = fprintf(['Reading Train->CTRL file: ', num2str(f), '/', num2str(length(CTRL_files))]);
    data = csvread(fullfile(path, 'CTRL', CTRL_files(f).name));
%     data = resample(data, 25, 64);   % Resampling to 100 Hz (from paper)
    X = abs(fft(data, NFFT, 1));
    X = X(1:K, :).^2;
    
    features(:, :, f) = X;
    labels(f) = 2;
    fprintf(repmat('\b', 1, fcount))
end
save ADD_beforeCSP_all.mat features labels

%% ADD file processing
% features = zeros(K, 385, length(ADD_files));
for f = 1:length(ADD_files)
    fcount = fprintf(['Reading Train->ADD file: ', num2str(f), '/', num2str(length(ADD_files))]);
    data = csvread(fullfile(path, 'ADD', ADD_files(f).name));
%     data = resample(data, 25, 64);   % Resampling to 100 Hz (from paper)
    X = abs(fft(data, NFFT, 1));
    X = X(1:K, :).^2;
    
    features(:, :, f+length(CTRL_files)) = X;
%     features(:, :, f) = X;
    labels(f+length(CTRL_files)) = 1;
    fprintf(repmat('\b', 1, fcount))
end
save CTRL_beforeCSP_all.mat features labels

%% ADHD file processing
% features = zeros(K, 385, length(ADHD_files));
for f = 1:length(ADHD_files)
    fcount = fprintf(['Reading Train->ADHD file: ', num2str(f), '/', num2str(length(ADHD_files))]);
    data = csvread(fullfile(path, 'ADHD', ADHD_files(f).name));
%     data = resample(data, 25, 64);   % Resampling to 100 Hz (from paper)
    X = abs(fft(data, NFFT, 1));
    X = X(1:K, :).^2;
    
    features(:, :, f+length(CTRL_files)+length(ADD_files)) = X;
%     features(:, :, f) = X;
    labels(f+length(CTRL_files)+length(ADD_files)) = 1;
    fprintf(repmat('\b', 1, fcount))
end

fprintf('Feature extraction is complete.\n')
save ADHD_beforeCSP_all.mat features labels

%% CSP
W = cell(1, size(fbank, 1));
sel_ch = 6;  % Select columns 
Prj_X  = zeros(sel_ch*size(fbank, 1), size(labels, 2));
start = 1;
for bank = 1: size(fbank, 1)
    X = squeeze(sum(features(fbank(bank, :)>0, :, :), 1));
    W{bank} = RegCsp(X, labels, 0, 0); % No regularization
    
    stop = start + sel_ch-1;
%     for i = 1:size(X, 3)
    Prj_X(start:stop, :) = X * W{bank}(1:sel_ch, :).'; % Projection
%     end
    start = stop + 1;
end
fprintf('CSP processing is complete.\n')

%% Linear Discriminant Analysis
fcount = fprintf('Started LDA analysis.\n');
Prj = reshape(Prj_X, [size(Prj_X, 1)*size(Prj_X, 2), size(Prj_X, 3)]);
% [Y, EV, E] = LDA(Prj.', labels);
fprintf(repmat('\b', 1, fcount))
fprintf('LDA analysis complete.\n');

%%
csvwrite('filter_bank_ADHD_large.csv', [Prj; labels].');
fprintf('Data saved.\n')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% END OF Main_routing.m