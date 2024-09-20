clear
clc
close all

% add path to PEMO-Q package

figures = false;

%% librosa STFT
w = 2048;
a = w/4;
M = w;
r = 110; % dynamic range for spectrograms
win = fftshift(gabwin('hann', a, M));
win = win / max(win);
F = frame('dgtreal', fftshift(win), a, M, 'timeinv');

for signal_id = 0:7

for mask_id = 1:6

outname = sprintf("example%d_mask%d.mat", signal_id, mask_id);
if isfile(outname)
    continue
end

%% load audio and mask
[audio, fs] = audioread("../audio-originals/audio_original_example" + num2str(signal_id) + ".wav");
mask = load("../masks/spectrogram_mask" + num2str(mask_id));
mask = mask.("C" + num2str(mask_id));

% reconstruction settings
margin = round(0.128*fs);
p = round(0.032*fs);
maxit = 5;
algoit = 1e3;

% ensure length compatibility
coefs = framecoef2native(F, librosastft(F, audio));
audio = librosaistft(F, librosastft(F, audio));
time = (0:length(audio)-1)/fs;

%% prepare
numgaps = 5;
allmiss = find(~mask);
allmiss = reshape(allmiss, [], numgaps);

arnames = [ ...
    "extrapolation, ignoring affected samples", ...
    "Janssen, ignoring affected samples", ...
    "Janssen, with TF consistency (approximal, DRaccel)", ...
    "Janssen, with TF consistency (approximal, DR)", ...
    "Janssen, with TF consistency (approximal, PG)", ...
    "Janssen, with TF consistency (ADMM, primal)", ...
    "Janssen, with TF consistency (ADMM, dual)", ...
    "Janssen, with TF consistency (CP)"];
arsigs = zeros(length(audio), numgaps, length(arnames));
hardmask = true(length(audio), numgaps);
times = NaN(numgaps, length(arnames));

%% process
for g = 1:numgaps

    fprintf("Gap %d of %d...\n", g, numgaps)

    % mask for spectrogram
    missframes = allmiss(:, g);
    sgmask = true(size(coefs));
    sgmask(:, missframes) = false;
        
    % mask for audio
    hardmask(1 + (missframes(1)-1)*a - floor(w/2) : (missframes(end)-1)*a + ceil(w/2), g) = false;
    
    % gapped audio
    gapped = librosaistft(F, framenative2coef(F, sgmask .* coefs));
    
    %% extrapolation-based inpainting
    fprintf("Extrapolation-based inpainting... ")
    t = tic;
    toinpaint = gapped;
    toinpaint(~hardmask(:, g)) = NaN;
    arsigs(:, g, 1) = arinpaint(toinpaint, margin, p, "arburg");
    times(g, 1) = toc(t);
    fprintf("%.2f s\n", times(g, 1))
    
    %% janssen
    objectives = NaN(maxit, length(arnames));
    
    % =============================
    % ignoring the affected samples
    % =============================
    fprintf("Janssen inpainting... ")
    t = tic;
    from = find(~hardmask(:, g), 1, "first") - margin;
    to = find(~hardmask(:, g), 1, "last") + margin;
    seggapped = gapped(from:to) .* hardmask(from:to, g);
    segmask = hardmask(from:to, g);
    masks.R = segmask;
    masks.U = false;
    masks.L = false;
    
    arsigs(:, g, 2) = gapped .* hardmask(:, g);
    [arsigs(from:to, g, 2), objectives(:, 2)] = janssen("inpainting", gapped(from:to), masks, 0, p, maxit, "verbose", true);
    
    times(g, 2) = toc(t);
    fprintf("%.2f s\n", times(g, 2))

    % ======================================================
    % consistency in spectrogram, approximal, accelerated DR
    % ======================================================
    fprintf("Janssen inpainting with TF consistency (approximal, DRaccel)... ")
    t = tic;
    sigprox = @(x, t) sgsigprox(x, gapped, from, to, F, sgmask, coefs);
    arsigs(:, g, 3) = gapped;
    [arsigs(from:to, g, 3), objectives(:, 3)] = janssen_sg(gapped(from:to), sigprox, p, maxit, ...
        "algo", "DRaccel", ...
        "algoit", algoit, ...
        "verbose", true);
    times(g, 3) = toc(t);
    fprintf("%.2f s\n", times(g, 3))

    % ==========================================
    % consistency in spectrogram, approximal, DR
    % ==========================================
    fprintf("Janssen inpainting with TF consistency (approximal, DR)... ")
    t = tic;
    sigprox = @(x, t) sgsigprox(x, gapped, from, to, F, sgmask, coefs);
    arsigs(:, g, 4) = gapped;
    [arsigs(from:to, g, 4), objectives(:, 4)] = janssen_sg(gapped(from:to), sigprox, p, maxit, ...
        "algo", "DR", ...
        "algoit", algoit, ...
        "verbose", true);
    times(g, 4) = toc(t);
    fprintf("%.2f s\n", times(g, 4))
    
    % ==========================================
    % consistency in spectrogram, approximal, PG
    % ==========================================
    fprintf("Janssen inpainting with TF consistency (approximal, PG)... ")
    t = tic;
    sigprox = @(x, t) sgsigprox(x, gapped, from, to, F, sgmask, coefs);
    arsigs(:, g, 5) = gapped;
    [arsigs(from:to, g, 5), objectives(:, 5)] = janssen_sg(gapped(from:to), sigprox, p, maxit, ...
        "algo", "PG", ...
        "algoit", algoit, ...
        "verbose", true);
    times(g, 5) = toc(t);
    fprintf("%.2f s\n", times(g, 5))

    % =====================================
    % consistency in spectrogram using ADMM
    % =====================================
    fprintf("Janssen inpainting with TF consistency (ADMM)... ")
    t = tic;
    colstart = missframes(1);
    colend = missframes(end);
    colmargin = 5; % margin as number of spectrogram columns
    colstart = colstart - colmargin;
    colend = colend + colmargin;
    collength = colend - colstart + 1; % length of the selection
    collength = 4*ceil(collength/4); % ensure that length of the selection is divisible by 4
    colend = colstart + collength - 1;
    partcoefs = coefs(:, colstart:colend);
    partmask = sgmask(:, colstart:colend);
    partgapped = librosaistft(F, framenative2coef(F, partcoefs .* partmask));
    coefprox = @(c, t) partcoefs(:) .* partmask(:) + c .* (1-partmask(:));
    lambda = 10;
    % coefprox = @(c, t) (c + lambda * t * partcoefs(:)) .* partmask(:) / (1 + lambda * t) + c .* (1-partmask(:));
    [partsig, objectives(:, 6), ~, z] = janssen_sg(partgapped, coefprox, p, maxit, ...
        "algo", "ADMM", ...
        "algoit", algoit, ...
        "K", @(x) librosastft(F, x), ...
        "K_adj", @(c) librosaistft(F, c), ...
        "verbose", true, ...
        "fun", @(c) lambda * norm(c .* partmask(:) - partcoefs(:) .* partmask(:)));
    arsigs(:, g, 6) = gapped;
    arsigs(:, g, 7) = gapped;
    objectives(:, 7) = objectives(:, 6);
    
    % take only the "gapped" part
    partstart = 1 + (colstart-1)*F.a;
    partend = partstart + length(partgapped) - 1;
    gapstart = find(~hardmask(:, g), 1, 'first');
    gapend = find(~hardmask(:, g), 1, 'last');
    arsigs(gapstart:gapend, g, 6) = partsig(1+gapstart-partstart:1+gapend-partstart);

    zsig = librosaistft(F, z);
    arsigs(gapstart:gapend, g, 7) = zsig(1+gapstart-partstart:1+gapend-partstart);
    
    times(g, 6) = toc(t);
    times(g, 7) = times(g, 6);
    fprintf("%.2f s\n", times(g, 6))

    if figures
        figure %#ok<UNRCH>
        tiledlayout(2, 5)
        nexttile([1, 5])
        plot(gapped, "DisplayName", "gapped")
        hold on
        plot(partstart:partend, partsig, "DisplayName", "restored from Janssen")
        xline([gapstart, gapend], "DisplayName", "considered for reco")
        legend
        
        nexttile
        imagesc(log10(abs(partcoefs .* partmask)))
        clim([-6.5 0])
        colorbar
        title("partcoefs")
        
        z = framecoef2native(F, z);
        nexttile
        imagesc(log10(abs(z)))
        clim([-6.5 0])
        colorbar
        title("ADMM dual variable")
        
        nexttile
        imagesc(log10(abs((z - partcoefs) .* partmask)))
        colorbar
        title("difference of partcoefs and ADMM dual")
        
        recoefs = librosastft(F, partsig);
        recoefs = framecoef2native(F, recoefs);
        nexttile
        imagesc(log10(abs(recoefs)))
        clim([-6.5 0])
        colorbar
        title("analysis of partsig (ADMM primal)")
        
        nexttile
        imagesc(log10(abs((recoefs - partcoefs) .* partmask)))
        colorbar
        title("difference of partcoefs and ADMM primal")
    end
    
    % ===================================
    % consistency in spectrogram using CP
    % ===================================
    fprintf("Janssen inpainting with TF consistency (CP)... ")
    t = tic;
    [partsig, objectives(:, 8), ~] = janssen_sg(partgapped, coefprox, p, maxit, ...
        "algo", "CP", ...
        "algoit", algoit, ...
        "K", @(x) librosastft(F, x), ...
        "K_adj", @(c) librosaistft(F, c), ...
        "verbose", true, ...
        "fun", @(c) lambda * norm(c .* partmask(:) - partcoefs(:) .* partmask(:)));
    arsigs(:, g, 8) = gapped;
    arsigs(gapstart:gapend, g, 8) = partsig(1+gapstart-partstart:1+gapend-partstart);
    times(g, 8) = toc(t);
    fprintf("%.2f s\n", times(g, 8))
end

%% merge solution
fullmask = prod(hardmask, 2);
arsolution = zeros(length(audio), length(arnames));
artime = sum(times, 1)';
for i = 1:length(arnames)
    arsolution(:, i) = audio .* fullmask + sum((1-hardmask) .* arsigs(:, :, i), 2);
end

%% plot
% plot time domain
if figures
    figure %#ok<UNRCH>
    plot(time, arsolution)
    hold on
    plot(time, audio, "color", 0.15*[1 1 1])
    legend([arnames, "original"])
    
    % plot spectrograms and compute evaluation
    sg = @(x) plotframe(F, librosastft(F, x), 'dynrange', r);
    
    figure
    tiledlayout("flow")
    
    nexttile
    sg(audio)
    title("original")
end

reaudio = resample(audio, 44100, fs);
SNR = NaN(length(arnames), 1);
ODG = NaN(length(arnames), 1);
for i = 1:size(arsolution, 2)

    % spectrogram
    if figures
        nexttile %#ok<UNRCH>
        sg(arsolution(:, i))
        title(arnames(i))
    end

    % SNR
    SNR(i) = snr(audio, audio-arsolution(:, i));

    % Pemo-Q
    resolution = resample(arsolution(:, i), 44100, fs);
    try
        [~, ~, ODG(i), ~] = audioqual(reaudio, resolution, 44100);
    catch
        warning("ODG not computed, PEMO-Q package is not on the search path.")
    end
    
    % output
    fprintf("%50s: SNR = %5.2f dB, ODG = %4.2f\n", arnames(i), SNR(i), ODG(i))

end

%% save
save(outname, "arnames", "arsolution", "artime", "audio", "coefs", "fs", "mask", "ODG", "SNR")

end
end

%% functions
function y = sgsigprox(x, gapped, from, to, F, sgmask, coefs)

    z = gapped;
    z(from:to) = x;
    y = librosaistft(F, framenative2coef(F, framecoef2native(F, librosastft(F, z)) .* (1-sgmask) + coefs .* sgmask)); % projection
    y = y(from:to);

end

function y = librosastft(F, x)

    y = frana(F, x);
    y = framecoef2native(F, y);
    y = y .* exp(1i*pi*(0:F.M/2)'); % phase correction
    y = framenative2coef(F, y);

end

function x = librosaistft(F, y)

    y = framecoef2native(F, y);
    y = y ./ exp(1i*pi*(0:F.M/2)'); % phase correction
    y = framenative2coef(F, y);
    x = frsyn(F, y);
    factor = (length(F.g)/F.a) * norm(F.g)^2;
    x = x / factor;

end