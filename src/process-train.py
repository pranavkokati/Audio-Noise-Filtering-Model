!git clone https://github.com/microsoft/MS-SNSD.git

!pip install librosa soundfile tensorflow numpy

import os
import random
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import soundfile as sf

# Your parameters
sr = 16000  # Sample rate
snr = 8     # Signal-to-noise ratio
n_mels = 128      # Number of Mel filterbanks
hop_length = 256  # Hop length for STFT
n_fft = 512      # FFT size for STFT
save_path = "/content/Dataset"
mix_path = os.path.join(save_path, "mixtures")
sim_path = os.path.join(save_path, "sources")
target_length_train = sr * 5  # 5 seconds = 80,000 samples for training
target_length_valid = 81408   # Max length for 160 frames at hop_length=512 (5.088s)

# Paths to MS-SNSD dataset
clean_train_path = "/content/MS-SNSD/clean_train"
noise_train_path = "/content/MS-SNSD/noise_train"
clean_valid_path = "/content/MS-SNSD/clean_test"
noise_valid_path = "/content/MS-SNSD/noise_test"

# Ensure directories exist
os.makedirs(mix_path, exist_ok=True)
os.makedirs(sim_path, exist_ok=True)

# Add Noise Function
def add_noise(clean, noise, snr):
    clean_power = np.mean(clean**2)
    noise_power = np.mean(noise**2)
    desired_noise_power = clean_power / (10**(snr / 10))
    noise = noise * np.sqrt(desired_noise_power / (noise_power + 1e-8))
    return clean + noise

# Pre-generate mixture and source files
def prepare_data(clean_path, noise_path, output_mix_path, output_sim_path, num_files=6000, is_training=True):
    clean_files = [f for f in os.listdir(clean_path) if f.endswith(".wav")]
    noise_files = [f for f in os.listdir(noise_path) if f.endswith(".wav")]
    speaker_list = []
    processed_files = 0

    random.shuffle(clean_files)

    for i, cfile in enumerate(clean_files):
        if processed_files >= num_files:
            break

        clean, _ = librosa.load(os.path.join(clean_path, cfile), sr=sr, mono=True)
        noise_file = random.choice(noise_files)
        noise, _ = librosa.load(os.path.join(noise_path, noise_file), sr=sr, mono=True)

        if is_training:
            if len(clean) < target_length_train:
                continue
            clean = clean[:target_length_train]
            noise = noise[:target_length_train]
        else:
            min_len = min(len(clean), len(noise))
            max_len = min(min_len, target_length_valid)
            clean = clean[:max_len]
            noise = noise[:max_len]

        noisy = add_noise(clean, noise, snr)

        speaker_id = f"spk_{processed_files:04d}"
        sf.write(os.path.join(output_mix_path, f"mix_{speaker_id}.wav"), noisy, sr)
        os.makedirs(os.path.join(output_sim_path, speaker_id), exist_ok=True)
        sf.write(os.path.join(output_sim_path, speaker_id, "s1.wav"), clean, sr)
        speaker_list.append(speaker_id)

        processed_files += 1
        if processed_files % 1000 == 0:
            print(f"Processed {processed_files} files")

    print(f"Total files processed: {processed_files}")
    return speaker_list

# Prepare data
print("Preparing training data (5 seconds only)...")
train_speakers = prepare_data(clean_train_path, noise_train_path, mix_path, sim_path, is_training=True)
print("Preparing validation data (max 5.088s)...")
valid_speakers = prepare_data(clean_valid_path, noise_valid_path, mix_path, sim_path, num_files=1000, is_training=False)

# Modified train_gen and valid_gen to enforce max_frames
def train_gen():
    max_frames = 160  # Hardcode to match pipeline
    for speaker in train_speakers:
        mixture, sr_local = librosa.load(mix_path + 'mix_' + speaker + '.wav', sr=sr, mono=False)
        mixture_ft = librosa.stft(mixture, n_fft=512, hop_length=512, win_length=512)  # Align hop_length
        mixture_mel = librosa.feature.melspectrogram(S=np.abs(mixture_ft), sr=sr_local, n_fft=512, hop_length=512, win_length=None, window='hann')
        # Truncate to max_frames
        mixture_mel = mixture_mel[:, :max_frames]

        output_masks = []
        for ii in np.arange(1, 2):
            sim, sr_local = librosa.load(sim_path + speaker + '/s' + str(ii) + '.wav', sr=sr, mono=False)
            sim_ft = librosa.stft(sim, n_fft=512, hop_length=512, win_length=512)
            sim_mel = librosa.feature.melspectrogram(S=np.abs(sim_ft), sr=sr_local, n_fft=512, hop_length=512, win_length=None, window='hann')
            sim_mel = sim_mel[:, :max_frames]  # Truncate to match
            source_mask = np.abs(sim_mel) / (np.maximum(sim_mel, mixture_mel) + 1e-8)
            output_masks.append(source_mask)
        output_masks = np.stack(output_masks, axis=-1)

        input_data = tf.convert_to_tensor(mixture_mel, dtype=tf.float32)
        source_mask = tf.convert_to_tensor(output_masks, dtype=tf.float32)

        yield input_data, source_mask

def valid_gen():
    max_frames = 160  # Hardcode to match pipeline
    for speaker in valid_speakers:
        mixture, sr_local = librosa.load(mix_path + 'mix_' + speaker + '.wav', sr=sr, mono=False)
        mixture_ft = librosa.stft(mixture, n_fft=512, hop_length=512, win_length=512)
        mixture_mel = librosa.feature.melspectrogram(S=np.abs(mixture_ft), sr=sr_local, n_fft=512, hop_length=512, win_length=None, window='hann')
        mixture_mel = mixture_mel[:, :max_frames]  # Truncate to max_frames

        output_masks = []
        for ii in np.arange(1, 2):
            sim, sr_local = librosa.load(sim_path + speaker + '/s' + str(ii) + '.wav', sr=sr, mono=False)
            sim_ft = librosa.stft(sim, n_fft=512, hop_length=512, win_length=512)
            sim_mel = librosa.feature.melspectrogram(S=np.abs(sim_ft), sr=sr_local, n_fft=512, hop_length=512, win_length=None, window='hann')
            sim_mel = sim_mel[:, :max_frames]  # Truncate to match
            source_mask = np.abs(sim_mel) / (np.maximum(sim_mel, mixture_mel) + 1e-8)
            output_masks.append(source_mask)
        output_masks = np.stack(output_masks, axis=-1)

        input_data = tf.convert_to_tensor(mixture_mel, dtype=tf.float32)
        source_mask = tf.convert_to_tensor(output_masks, dtype=tf.float32)

        yield input_data, source_mask

# U-Net Model
def build_unet(input_shape):
    inputs = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    bottleneck = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    up4 = layers.UpSampling2D((2, 2))(bottleneck)
    concat4 = layers.Concatenate()([up4, conv3])
    conv4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(concat4)
    up5 = layers.UpSampling2D((2, 2))(conv4)
    concat5 = layers.Concatenate()([up5, conv2])
    conv5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(concat5)
    up6 = layers.UpSampling2D((2, 2))(conv5)
    concat6 = layers.Concatenate()([up6, conv1])
    conv6 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(concat6)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(conv6)
    return models.Model(inputs, outputs)

if __name__ == "__main__":
    # Fix path syntax
    mix_path = mix_path + "/"
    sim_path = sim_path + "/"

    # Set max_frames
    max_frames = 160  # Matches 5s training and truncated validation
    input_shape = (n_mels, max_frames, 1)
    batch_size = 16

    # Output signature
    output_signature = (
        tf.TensorSpec(shape=(n_mels, max_frames), dtype=tf.float32),  # Fixed size
        tf.TensorSpec(shape=(n_mels, max_frames, 1), dtype=tf.float32)
    )

    # Data pipeline
    ds_train = tf.data.Dataset.from_generator(
        train_gen,
        output_signature=output_signature
    )
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(buffer_size=1024)
    ds_train = ds_train.batch(batch_size)  # No padding needed, fixed size
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    ds_train = ds_train.repeat(1)

    ds_valid = tf.data.Dataset.from_generator(
        valid_gen,
        output_signature=output_signature
    )
    ds_valid = ds_valid.batch(batch_size)  # No padding needed, fixed size
    ds_valid = ds_valid.prefetch(tf.data.AUTOTUNE)
    ds_valid = ds_valid.repeat(1)

    # Ensure checkpoint directory exists
    os.makedirs('model_checkpoints', exist_ok=True)

    # Clear previous models
    tf.keras.backend.clear_session()

    # Initialize model
    model_unet = build_unet(input_shape)
    model_unet.summary()

    # Training configuration
    loss_fn = tf.keras.losses.MeanSquaredError()
    initial_learning_rate = 0.003

    def lr_schedule(epoch, lr):
        if epoch % 10 == 0 and epoch != 0:
            return lr * 0.5
        return lr

    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    callback_lr = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='model_checkpoints/model_unet_epoch_{epoch:02d}.weights.h5',
        save_weights_only=True,
        save_best_only=True,
        save_freq='epoch'
    )

    model_unet.compile(optimizer=optimizer, loss=loss_fn, metrics=['mae'])

    # Train
    history = model_unet.fit(ds_train, epochs=150, callbacks=[callback_lr, checkpoint_callback], validation_data=ds_valid)

    # Save final model
    model_unet.save('model_unet_final.keras')

    # Plot training history
    plt.plot(history.history['loss'], label='Learning loss f-value')
    plt.plot(history.history['val_loss'], label='Validation loss f-statement')
    plt.xlabel('Epoch')
    plt.ylabel('Loss F-account')
    plt.title('Change in the training and validation loss f-ratio over epochs')
    plt.legend()
    plt.show()
