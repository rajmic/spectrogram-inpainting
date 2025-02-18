# spectrogram-inpainting
A repository and [webpage](https://rajmic.github.io/spectrogram-inpainting/) accompanying the paper *Janssen 2.0: Audio Inpainting in the Time-frequency Domain*.

> The paper focuses on inpainting missing parts of an audio signal spectrogram. The autoregression-based Janssen algorithm, the state-of-the-art for the time-domain audio inpainting, is adapted for the time-frequency setting. This novel method, termed Janssen-TF, is compared to the deep-prior neural network approach using both objective metrics and a subjective listening test, proving Janssen-TF to be superior in all the considered measures.

## Contents of the repository

The paper compares a recent method abbreviated DPAI with the newly proposed Janssen-TF approach.
- DPAI codes are not a part of this repository but are available [here](https://github.com/fmiotello/dpai).
- Matlab codes of our method are available in the `Janssen-TF` folder.
- For reproducibility reasons, the codes are set to read the input (uncorrupted) audio files from the `audio-originals` folder.
- The spectrogram masks used in our experiments are read from the `masks` folder.
- Regarding the mid-scale experiment using the IRMAS dataset, the folder `audio-irmas` includes a list of the files used in our experiment and a Matlab script which crops the files to a length of 5 seconds and subsamples them to 16 kHz. The original files can be downloaded [here](https://www.upf.edu/web/mtg/irmas).

Note that there are several autoregression-based methods implemented in the `Janssen-TF` folder;
to exactly reproduce results from the paper, switch to *ADMM, primal*.
This provides the time-domain signal from line 7 of the algorithm in the paper, after the convergence is reached.

## Dependencies

The Matlab codes for Janssen-TF use the [LTFAT](https://ltfat.org/) and the Signal Processing Toolbox. To compute the perceptually-motivated evaluation, we have used the [PEMO-Q package](https://uol.de/en/mediphysics/downloads/pemo-q) (version 1.4.1).
