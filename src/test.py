import os
import random
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import soundfile as sf
from IPython.display import Audio, display
from sklearn.metrics import mean_squared_error
from scipy.ndimage import gaussian_filter

# Your parameters
sr = 16000
snr = 8
n_mels = 128
hop_length = 256
n_fft = 512
save_path = "/content/Dataset"
mix_path = os.path.join(save_path, "mixtures")
sim_path = os.path.join(save_path, "sources")
target_length_train = sr * 5
max_frames = 160

# Paths to MS-SNSD dataset
clean_train_path = "/content/MS-SNSD/clean_train"
noise_train_path = "/content/MS-SNSD/noise_train"
clean_valid_path = "/content/MS-SNSD/clean_test"
noise_valid_path = "/content/MS-SNSD/noise_test"

# Add Noise Function
def add_noise(clean, noise, snr):
    clean_power = np.mean(clean**2)
    noise_power = np.mean(noise**2)
    desired_noise_power = clean_power / (10**(snr / 10))
    noise = noise * np.sqrt(desired_noise_power / (noise_power + 1e-8))
    return clean + noise

# Prepare test data
def prepare_test_data(clean_path, noise_path, output_mix_path, output_sim_path, num_files=1):
    clean_files = [f for f in os.listdir(clean_path) if f.endswith(".wav")]
    noise_files = [f for f in os.listdir(noise_path) if f.endswith(".wav")]
    test_speakers = []
    processed_files = 0

    random.shuffle(clean_files)

    for i, cfile in enumerate(clean_files):
        if processed_files >= num_files:
            break

        clean, _ = librosa.load(os.path.join(clean_path, cfile), sr=sr, mono=True)
        noise_file = random.choice(noise_files)
        noise, _ = librosa.load(os.path.join(noise_path, noise_file), sr=sr, mono=True)

        if len(clean) < target_length_train:
            continue
        clean = clean[:target_length_train]
        noise = noise[:target_length_train]

        noisy = add_noise(clean, noise, snr)

        speaker_id = f"test_spk_{processed_files:04d}"
        sf.write(os.path.join(output_mix_path, f"mix_{speaker_id}.wav"), noisy, sr)
        os.makedirs(os.path.join(output_sim_path, speaker_id), exist_ok=True)
        sf.write(os.path.join(output_sim_path, speaker_id, "s1.wav"), clean, sr)
        test_speakers.append(speaker_id)

        processed_files += 1

    print(f"Test files processed: {processed_files}")
    return test_speakers

# Process test input with padding and phase preservation
def process_test_input(mix_file, sim_file):
    mixture, _ = librosa.load(mix_file, sr=sr, mono=False)
    sim, _ = librosa.load(sim_file, sr=sr, mono=False)

    # Compute STFT and save phase
    mixture_ft = librosa.stft(mixture, n_fft=n_fft, hop_length=512, win_length=512)
    mixture_mel = librosa.feature.melspectrogram(S=np.abs(mixture_ft), sr=sr, n_fft=512, hop_length=512, win_length=None, window='hann')
    mixture_phase = np.angle(mixture_ft)  # Save phase
    if mixture_mel.shape[1] < max_frames:
        pad_width = max_frames - mixture_mel.shape[1]
        mixture_mel = np.pad(mixture_mel, ((0, 0), (0, pad_width)), mode='constant')
        # Pad phase to match max_frames (though STFT might already be truncated)
        mixture_phase = mixture_phase[:, :max_frames]
    mixture_mel = mixture_mel[:, :max_frames]

    sim_ft = librosa.stft(sim, n_fft=512, hop_length=512, win_length=512)
    sim_mel = librosa.feature.melspectrogram(S=np.abs(sim_ft), sr=sr, n_fft=512, hop_length=512, win_length=None, window='hann')
    if sim_mel.shape[1] < max_frames:
        pad_width = max_frames - sim_mel.shape[1]
        sim_mel = np.pad(sim_mel, ((0, 0), (0, pad_width)), mode='constant')
    sim_mel = sim_mel[:, :max_frames]

    input_data = tf.convert_to_tensor(mixture_mel, dtype=tf.float32)[tf.newaxis, ..., tf.newaxis]
    ground_truth_mask = np.abs(sim_mel) / (np.maximum(sim_mel, mixture_mel) + 1e-8)

    return input_data, mixture_ft, mixture_phase, ground_truth_mask

# Reconstruct audio from predicted mask using saved phase
def reconstruct_audio(mixture_ft, predicted_mask, mixture_phase, hop_length=512, win_length=512):
    mixture_magnitude = np.abs(mixture_ft)  # Shape: (257, 157)
    n_frames = mixture_magnitude.shape[1]

    # Interpolate mask to STFT frequency bins
    mel_basis = librosa.filters.mel(sr=sr, n_fft=512, n_mels=n_mels)
    inverse_mel = np.linalg.pinv(mel_basis)
    predicted_mask_stft = np.dot(inverse_mel, predicted_mask)  # (257, 160)
    predicted_mask_stft = predicted_mask_stft[:, :n_frames]  # (257, 157)

    # Smooth and apply adjustable threshold
    predicted_mask_stft = gaussian_filter(predicted_mask_stft, sigma=1)
    predicted_mask_stft = np.clip(predicted_mask_stft, 0, 1)
    predicted_mask_stft = (predicted_mask_stft > 0.3).astype(float)  # Softer threshold

    # Apply mask and debug
    predicted_magnitude = mixture_magnitude * predicted_mask_stft
    print("Predicted Mask STFT Min/Max:", predicted_mask_stft.min(), predicted_mask_stft.max())
    print("Predicted Magnitude Min/Max:", predicted_magnitude.min(), predicted_magnitude.max())

    # Reconstruct using saved phase
    predicted_stft = predicted_magnitude * np.exp(1j * mixture_phase)
    audio = librosa.istft(predicted_stft, hop_length=hop_length, win_length=win_length)
    audio = librosa.util.normalize(audio)
    return audio

# Signal-to-Distortion Ratio (SDR) calculation
def calculate_sdr(reference, estimation):
    min_length = min(len(reference), len(estimation))
    reference = reference[:min_length]
    estimation = estimation[:min_length]
    signal_power = np.sum(reference ** 2)
    error = reference - estimation
    error_power = np.sum(error ** 2)
    sdr = 10 * np.log10(signal_power / (error_power + 1e-8))
    return sdr

# Load the trained model
model_unet = tf.keras.models.load_model('model_unet_final.keras')

# Prepare test data
print("Preparing test data...")
test_speakers = prepare_test_data(clean_valid_path, noise_valid_path, mix_path, sim_path, num_files=1)
test_speaker = test_speakers[0]
mix_file = os.path.join(mix_path, f"mix_{test_speaker}.wav")
sim_file = os.path.join(sim_path, test_speaker, "s1.wav")

# Process test input
input_data, mixture_ft, mixture_phase, ground_truth_mask = process_test_input(mix_file, sim_file)

# Predict the mask
predicted_mask = model_unet.predict(input_data)[0, :, :, 0]

# Reconstruct the separated audio
separated_audio = reconstruct_audio(mixture_ft, predicted_mask, mixture_phase)

# Save the separated audio
output_audio_file = os.path.join(save_path, f"separated_{test_speaker}.wav")
sf.write(output_audio_file, separated_audio, sr)
print(f"Separated audio saved to: {output_audio_file}")

# Load original mixture and ground truth for comparison
mixture_audio, _ = librosa.load(mix_file, sr=sr, mono=True)
ground_truth_audio, _ = librosa.load(sim_file, sr=sr, mono=True)

# Compute separation quality
separated_mel = librosa.feature.melspectrogram(y=separated_audio, sr=sr, n_fft=512, hop_length=512, n_mels=n_mels)
if separated_mel.shape[1] < max_frames:
    pad_width = max_frames - separated_mel.shape[1]
    separated_mel = np.pad(separated_mel, ((0, 0), (0, pad_width)), mode='constant')
separated_mel = separated_mel[:, :max_frames]
mse = mean_squared_error(ground_truth_mask, separated_mel)
print(f"MSE between Ground Truth and Separated Mel Spectrograms: {mse:.4f}")

sdr = calculate_sdr(ground_truth_audio, separated_audio)
print(f"Signal-to-Distortion Ratio (SDR): {sdr:.2f} dB")

# Visualization
plt.figure(figsize=(15, 10))

plt.subplot(3, 2, 1)
plt.title("Mixture Mel Spectrogram")
plt.imshow(input_data[0, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
plt.colorbar()
plt.xlabel("Time Frames")
plt.ylabel("Mel Bins")

plt.subplot(3, 2, 2)
plt.title("Predicted Mask")
plt.imshow(predicted_mask, aspect='auto', origin='lower', cmap='gray')
plt.colorbar()
plt.xlabel("Time Frames")
plt.ylabel("Mel Bins")

plt.subplot(3, 2, 3)
plt.title("Ground Truth Mask")
plt.imshow(ground_truth_mask, aspect='auto', origin='lower', cmap='gray')
plt.colorbar()
plt.xlabel("Time Frames")
plt.ylabel("Mel Bins")

separated_mel_vis = input_data[0, :, :, 0] * predicted_mask
plt.subplot(3, 2, 4)
plt.title("Separated Mel Spectrogram")
plt.imshow(separated_mel_vis, aspect='auto', origin='lower', cmap='viridis')
plt.colorbar()
plt.xlabel("Time Frames")
plt.ylabel("Mel Bins")

plt.subplot(3, 2, 5)
plt.title("Waveforms")
plt.plot(mixture_audio[:target_length_train], label="Mixture", alpha=0.5)
plt.plot(ground_truth_audio[:target_length_train], label="Ground Truth", alpha=0.5)
plt.plot(separated_audio, label="Separated", alpha=0.5)
plt.legend()
plt.xlabel("Samples")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()

# Play audio in Colab
print("Playing Mixture Audio:")
display(Audio(mixture_audio, rate=sr))
print("Playing Ground Truth Audio:")
display(Audio(ground_truth_audio, rate=sr))
print("Playing Separated Audio:")
display(Audio(separated_audio, rate=sr))
