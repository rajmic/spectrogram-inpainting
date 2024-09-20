# spectrogram-inpainting
A repository and webpage accompanying the paper *Janssen 2.0: Audio Inpainting in the Time-frequency Domain*.

> The paper focuses on inpainting missing parts of an audio signal spectrogram. First, a recent successful approach based on an untrained neural network is revised and its several modifications are proposed, improving the signal-to-noise ratio of the restored audio. Second, the Janssen algorithm, the autoregression-based state-of-the-art for time-domain audio inpainting, is adapted for the time-frequency setting. This novel method, coined Janssen-TF, is compared to the neural network approach using both objective metrics and a subjective listening test, proving Janssen-TF to be superior in all the considered measures.

## Dependencies

The Matlab codes for Janssen-TF use the [LTFAT](https://ltfat.org/) and Signal Processing Toolbox. To compute the perceptually-motivated evaluation, we have used the [PEMO-Q package](https://uol.de/en/mediphysics/downloads/pemo-q) (version 1.4.1), which is not a part of this repository.