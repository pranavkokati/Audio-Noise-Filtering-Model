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
