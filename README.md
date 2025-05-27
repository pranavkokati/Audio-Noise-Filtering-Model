# Audio Noise Filtering Model

## Project Description

AudioSep-UNet is a deep learning project that implements a U-Net architecture for the task of audio source separation, specifically targeting the isolation of clean speech from noisy mixtures. The model operates on Mel-spectrogram representations of audio signals, learning to estimate a time-frequency mask that, when applied to the noisy mixture's spectrogram, suppresses the unwanted background noise while preserving the desired speech component. The project leverages the MS-SNSD dataset for training and evaluation, providing a robust framework for developing and assessing speech enhancement techniques.

## Features

*   **U-Net Architecture:** Utilizes a standard U-Net for effective time-frequency masking.
*   **Mel-Spectrogram Processing:** Processes audio data as Mel-spectrograms, a common representation in audio processing.
*   **Data Preparation Pipeline:** Includes scripts to generate noisy mixtures and corresponding clean sources from the MS-SNSD dataset with configurable SNR levels.
*   **Phase Preservation:** Reconstructs the separated audio using the original mixture's phase information for better perceptual quality.
*   **Objective Evaluation:** Incorporates metrics like Mean Squared Error (MSE) and Signal-to-Distortion Ratio (SDR) to quantitatively assess separation performance.
*   **Visualization Tools:** Provides visualizations of spectrograms and waveforms for qualitative analysis.
*   **TensorFlow Implementation:** Built using TensorFlow for efficient model training and deployment.

## Use Cases

This project can be applied in various scenarios where separating speech from noise is crucial:

*   **Speech Enhancement:** Improving the clarity of speech in noisy environments for communication systems, voice assistants, and recordings.
*   **Automatic Speech Recognition (ASR):** Pre-processing audio for ASR systems to improve recognition accuracy in noisy conditions.
*   **Hearing Aids and Assistive Listening Devices:** Developing algorithms to enhance the intelligibility of speech for individuals with hearing impairments.
*   **Audio Forensics:** Isolating specific audio sources from complex recordings.

## Setup

1.  **Clone the repository:**
2.  **Install dependencies:**
    Ensure you have Python installed. Then, install the required libraries using pip:
    \`\`\`bash
    #!/bin/bash
    echo pip install librosa soundfile tensorflow numpy matplotlib scikit-learn scipy IPython
    \`\`\`
