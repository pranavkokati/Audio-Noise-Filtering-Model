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
    Ensure you have Python installed. Then, install the required libraries using pip: !pip install librosa soundfile tensorflow numpy matplotlib scikit-learn scipy IPython

3.  **Dataset:**

    The project uses the MS-SNSD dataset. The provided code includes steps to clone and process the dataset within the execution environment (e.g., Google Colab). If running locally, ensure the dataset is downloaded and the paths     in the code are updated accordingly.

## Usage

To run the project, execute the main Python script (assuming your code is saved in a file named `main.py`):
The script will perform the following actions:

1.  Prepare the training and validation data by creating noisy mixtures and clean source files in the `Dataset` directory.
2.  Build and train the U-Net model. Model checkpoints will be saved in the `model_checkpoints` directory.
3.  Load the trained model.
4.  Prepare a test sample and process it.
5.  Predict the speech mask and reconstruct the separated audio.
6.  Save the separated audio file.
7.  Calculate and print evaluation metrics (MSE, SDR).
8.  Display visualizations and play audio samples (if running in a compatible environment like Colab or Jupyter Notebook).

## Code Structure

*   The main script contains functions for data generation, model definition, training loop, evaluation, and visualization.
*   `Dataset/`: Directory where the prepared mixture and source audio files are stored.
*   `model_checkpoints/`: Directory for saving model weights during training.

## Training Details

*   **Input:** Mel-spectrogram of the noisy mixture.
*   **Output:** A predicted mask of the same dimensions as the input Mel-spectrogram.
*   **Loss Function:** Mean Squared Error (MSE) between the predicted mask and the ground truth mask.
*   **Optimizer:** Adam optimizer with a learning rate scheduler.
*   **Epochs:** Configurable number of training epochs.
*   **Batch Size:** Configurable batch size.

## Evaluation

The project evaluates the model's performance using:

*   **Mean Squared Error (MSE):** Measures the difference between the predicted mask and the ground truth mask.
*   **Signal-to-Distortion Ratio (SDR):** A perceptual metric that quantifies the quality of the separated audio compared to the original clean source.

## Customization

*   You can adjust the model architecture, hyperparameters (e.g., learning rate, batch size), and training configurations in the script.
*   Experiment with different audio feature representations (e.g., different STFT parameters, other spectrogram types).
*   Explore different datasets for training and testing.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

This project utilizes the MS-SNSD dataset, available at [https://github.com/microsoft/MS-SNSD](https://github.com/microsoft/MS-SNSD).
