# Audio-Denoising-Model

# Audio Source Separation using U-Net

This project implements an audio source separation model using a U-Net architecture. It trains on a dataset of noisy speech to learn to separate clean speech from background noise. The model is trained on Mel-spectrogram representations of the audio signals.

## Project Overview

The goal of this project is to build a neural network capable of estimating a time-frequency mask to extract a desired source (in this case, clean speech) from a mixture of audio signals. The U-Net architecture, commonly used in image segmentation, is adapted here for spectrogram masking.

## Features

- Data preparation pipeline for generating noisy mixtures and corresponding clean sources from the MS-SNSD dataset.
- Implementation of a U-Net model for spectrogram masking.
- Training and evaluation of the model.
- Audio reconstruction using the predicted mask and original mixture phase.
- Visualization of spectrograms and waveforms for comparison.
- Calculation of objective metrics like Mean Squared Error (MSE) and Signal-to-Distortion Ratio (SDR) for evaluation.

## Setup

1.  **Clone the repository:**
