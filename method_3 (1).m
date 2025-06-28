function feat_vec = method_3(EEG)
% Permutation entropy
m = 3;
Pi = perms(1:m);  % Symbols
PeEn = zeros(size(EEG, 1), 1);
for ch = 1:size(EEG, 1)
    eeg_band_data = real(ifft(EEG(ch, :)));

    Xm = buffer(eeg_band_data, 3, 2);

    Xm = Xm(:, 1:end-m+1);

    [~, I] = sort(Xm, 1);
    p = zeros(1, size(Pi, 1));
    for j=1:size(Pi, 1)
        p(j) = sum(ismember(I.', Pi(1, :), 'rows'))/factorial(3)/size(Xm, 2);
    end
    PeEn(ch) = -p*log(p).';
end    

%% Feature vector
feat_vec = [PeEn(:)];

