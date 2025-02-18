clear
clc
close all

fold = "IRMAS"; % folder with the recordings listed in IRMAS_list.txt

% after processing, the files are saved to IRMAS_five_seconds (folder must
% exist)

files = struct2table(dir(fold));
files = files(~files.isdir, :);
for i = 1:height(files)
    fprintf("file %d/%d: %s\n", i, height(files), files.name{i})

    % load signal
    [signal, fs] = audioread(fullfile(fold, files.name{i}));
    
    % convert to mono
    signal = mean(signal, 2);

    % resample to 16 kHz
    signal = resample(signal, 16e3, fs);

    % take 5 seconds around the middle of the signal
    indices = round(length(signal)/2 - 2.5*16e3) + (1:5*16e3);
    signal = fade(signal(indices), round(0.1*16e3));

    % check that the samples are in [-1, 1]
    signal = signal / max(abs(signal));

    % write audio
    audiowrite( ...
        sprintf("IRMAS_five_seconds/audio_original_example%02d.wav", i-1), ...
        signal, ...
        16e3)
end

function y = fade(x, len)
    
    y = x;
    cosinus = cos(linspace(0, pi/2, len)').^2;
    y(1:len, :) = y(1:len, :) .* (1-cosinus);
    y(end-len+1:end, :) = y(end-len+1:end, :) .* cosinus;

end