import numpy as np
import parselmouth
from parselmouth.praat import call
import scipy.signal
import scipy.stats
import librosa
import pandas as pd

def extract_acoustic_features(audio_data, sr, original_audio_data=None):
    """
    Extract comprehensive acoustic features from audio data.

    Args:
        audio_data (array): Preprocessed audio signal
        sr (int): Sampling rate
        original_audio_data (array, optional): Original audio signal before normalization

    Returns:
        dict: Dictionary containing all extracted features
    """
    features = {}
    sound = parselmouth.Sound(audio_data, sr)

    # Voice Quality Features (Jitter, Shimmer, CPPS)
    features.update(_extract_voice_quality_features(sound))

    # Formant Features (Updated)
    features.update(_extract_formant_features(sound))

    # Spectral Features (Updated)
    features.update(_extract_spectral_features(sound, sr))

    # Harmonics-to-Noise Ratio (HNR)
    features.update(_extract_hnr_features(sound))

    # Amplitude Features (Extracted from original audio data)
    if original_audio_data is not None:
        original_sound = parselmouth.Sound(original_audio_data, sr)
        features.update(_extract_amplitude_features(original_sound))
    else:
        features.update(_extract_amplitude_features(sound))

    # Complexity Features (HFD)
    features.update(_extract_complexity_features(audio_data))

    # Pitch Features
    features.update(_extract_pitch_features(sound))

    # Additional Features (e.g., TrajIntra, Asymmetry)
    features.update(_extract_additional_features(audio_data))

    # AVQI HNR_sd Feature
    features.update(_extract_avqi_hnr_sd(audio_data, sr))

    # Amplitude Maximum Difference Mean
    features.update(_extract_amplitude_maximum_difference_mean(audio_data))

    # Amplitude Minimum
    features.update(_extract_amplitude_minimum(audio_data))

    return features

def _extract_voice_quality_features(sound):
    """Extract voice quality features including jitter, shimmer, and CPPS."""
    features = {}

    try:
        # Jitter and Shimmer
        point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
        features.update({
            'jitter_local': call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3),
            'jitter_ppq5': call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3),
            'shimmer_local': call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
            'shimmer_apq5': call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        })

        # CPPS Analysis
        spectrum = sound.to_spectrum()
        power_spectrum = np.abs(spectrum.values[0])**2
        cepstrum = np.fft.ifft(np.log(power_spectrum + 1e-10)).real

        quefrency = np.arange(len(cepstrum)) / sound.sampling_frequency
        valid_range = (quefrency >= 0.002) & (quefrency <= 0.025)

        window_size = int(0.001 * sound.sampling_frequency)
        if window_size % 2 == 0:
            window_size += 1
        smoothed_cepstrum = scipy.signal.savgol_filter(cepstrum, window_size, 2)

        peak_index = np.argmax(smoothed_cepstrum[valid_range])
        background = np.mean(smoothed_cepstrum[valid_range])
        features['CPPS'] = smoothed_cepstrum[valid_range][peak_index] - background

    except Exception as e:
        print(f"Error in voice quality analysis: {e}")
        features.update({
            'jitter_local': 0, 'jitter_ppq5': 0,
            'shimmer_local': 0, 'shimmer_apq5': 0,
            'CPPS': 0
        })
    return features

def _extract_amplitude_features(sound):
    """Extract average amplitude, peak amplitude, and amplitude variance features."""
    features = {}
    try:
        amplitude_values = sound.values[0]

        # Calculate Average Amplitude
        features['average_amplitude'] = np.mean(amplitude_values)

        # Calculate Peak Amplitude
        features['peak_amplitude'] = np.max(np.abs(amplitude_values))

        # Calculate Amplitude Variance
        features['amplitude_variance'] = np.var(amplitude_values)

    except Exception as e:
        print(f"Error in amplitude feature calculation: {e}")
        features['average_amplitude'] = 0
        features['peak_amplitude'] = 0
        features['amplitude_variance'] = 0

    return features

def _extract_formant_features(sound):
    """Extract formant features (F1-F4) and their dynamics."""
    features = {}
    try:
        formant = sound.to_formant_burg(
            time_step=0.01,
            max_number_of_formants=5,
            maximum_formant=6500
        )

        formant_values = {i: [] for i in range(1, 5)}  # F1-F4
        formant_deltas = {i: [] for i in range(1, 5)}  # Derivatives of F1-F4
        f3_b3_values = []  # Bandwidth of F3

        times = np.arange(0, sound.duration, 0.01)

        for t in times:
            for formant_number in range(1, 5):
                try:
                    value = formant.get_value_at_time(formant_number, t)
                    if value > 0:
                        formant_values[formant_number].append(value)
                        if len(formant_values[formant_number]) > 1:
                            # Compute delta (difference between consecutive formant values)
                            delta = value - formant_values[formant_number][-2]
                            formant_deltas[formant_number].append(delta)
                    if formant_number == 3:
                        bandwidth = formant.get_bandwidth_at_time(formant_number, t)
                        if bandwidth > 0:
                            f3_b3_values.append(bandwidth)
                except:
                    continue

        for formant_number, values in formant_values.items():
            if values:
                prefix = f'F{formant_number}'
                features.update({
                    f'{prefix}_mean': np.mean(values),
                    f'{prefix}_std': np.std(values),
                    f'{prefix}_range': np.max(values) - np.min(values),
                    f'{prefix}_median': np.median(values),
                    f'{prefix}_skewness': scipy.stats.skew(values),
                    f'{prefix}_kurtosis': scipy.stats.kurtosis(values)
                })
                if formant_number == 4:  # Additional F4 features
                    features[f'{prefix}_coefficient_of_variation'] = np.std(values) / np.mean(values)
                # Formant delta statistics
                deltas = formant_deltas[formant_number]
                if deltas:
                    features.update({
                        f'{prefix}_delta_mean': np.mean(deltas),
                        f'{prefix}_delta_std': np.std(deltas),
                        f'{prefix}_delta_range': np.max(deltas) - np.min(deltas)
                    })
            else:
                prefix = f'F{formant_number}'
                features.update({
                    f'{prefix}_mean': 0,
                    f'{prefix}_std': 0,
                    f'{prefix}_range': 0,
                    f'{prefix}_median': 0,
                    f'{prefix}_skewness': 0,
                    f'{prefix}_kurtosis': 0,
                    f'{prefix}_delta_mean': 0,
                    f'{prefix}_delta_std': 0,
                    f'{prefix}_delta_range': 0
                })
                if formant_number == 4:
                    features[f'{prefix}_coefficient_of_variation'] = 0

        # F3 Bandwidth (F3_B3)
        features['F3_B3'] = np.mean(f3_b3_values) if f3_b3_values else 0

        # F1 Standard Deviation (F1_sd)
        if formant_values[1]:
            features['F1_sd'] = np.std(formant_values[1])
        else:
            features['F1_sd'] = 0

    except Exception as e:
        print(f"Error in formant analysis: {e}")
        for i in range(1, 5):
            prefix = f'F{i}'
            features.update({
                f'{prefix}_mean': 0,
                f'{prefix}_std': 0,
                f'{prefix}_range': 0,
                f'{prefix}_median': 0,
                f'{prefix}_skewness': 0,
                f'{prefix}_kurtosis': 0,
                f'{prefix}_delta_mean': 0,
                f'{prefix}_delta_std': 0,
                f'{prefix}_delta_range': 0
            })
            if i == 4:
                features[f'{prefix}_coefficient_of_variation'] = 0
        features['F3_B3'] = 0
        features['F1_sd'] = 0

    return features

def _extract_complexity_features(audio_data):
    """Extract complexity features including Higuchi Fractal Dimension."""
    features = {}
    try:
        window_sizes = [128, 256, 512, 1024]
        hfd_values = []

        for window_size in window_sizes:
            num_windows = len(audio_data) // window_size
            for i in range(num_windows):
                start_idx = i * window_size
                end_idx = start_idx + window_size
                window_data = audio_data[start_idx:end_idx]
                hfd = _calculate_hfd(window_data)
                if not np.isnan(hfd):
                    hfd_values.append(hfd)

        if hfd_values:
            features.update({
                'HFD_mean': np.mean(hfd_values),
                'HFD_max': np.max(hfd_values),
                'HFD_min': np.min(hfd_values),
                'HFD_std': np.std(hfd_values),
                'HFD_var': np.var(hfd_values)
            })
        else:
            features.update({
                'HFD_mean': 0, 'HFD_max': 0,
                'HFD_min': 0, 'HFD_std': 0,
                'HFD_var': 0
            })

    except Exception as e:
        print(f"Error in complexity analysis: {e}")
        features.update({
            'HFD_mean': 0, 'HFD_max': 0,
            'HFD_min': 0,
            'HFD_std': 0,
            'HFD_var': 0
        })

    return features

def _calculate_hfd(signal, kmax=10):
    """
    Calculate the Higuchi Fractal Dimension of a signal.

    Args:
        signal (array): Input signal
        kmax (int): Maximum delay/lag (default=10)

    Returns:
        float: Higuchi Fractal Dimension
    """
    N = len(signal)
    L = np.zeros((kmax,))
    x = np.zeros((kmax,))

    for k in range(1, kmax + 1):
        Lk = 0
        for m in range(k):
            indices = np.arange(1, int((N-m)/k))
            Lmk = np.abs(signal[m + k*indices] - signal[m + k*(indices-1)]).sum()
            Lmk = (Lmk * (N - 1) / (((N-m)/k)*k)) / k
            Lk += Lmk

        L[k-1] = Lk / k
        x[k-1] = np.log(1.0 / k)

    p = np.polyfit(x, np.log(L), 1)
    return p[0]

def _extract_spectral_features(sound, sr):
    """Extract spectral features including MFCCs, spectral slope, centroid, flux, roll-off, zero-crossing rate, and energy entropy."""
    features = {}
    try:
        # Convert sound to spectrum
        spectrum = sound.to_spectrum()
        spectral_values = spectrum.values[0]
        frequencies = np.linspace(0, sr / 2, len(spectral_values))
        
        # Spectral Slope Calculation (Original)
        slope, intercept = np.polyfit(frequencies, spectral_values, 1)
        features['spectral_slope'] = slope

        # Spectral Centroid Calculation (Original)
        spectral_centroid = np.sum(frequencies * np.abs(spectral_values)) / np.sum(np.abs(spectral_values))
        features['spectral_centroid'] = spectral_centroid

        # MFCCs (using librosa)
        audio_signal = sound.values[0]
        n_mfcc = 30  # Increased number of MFCCs
        mfccs = librosa.feature.mfcc(y=audio_signal, sr=sr, n_mfcc=n_mfcc, hop_length=512)
        # Delta coefficients
        delta_mfccs = librosa.feature.delta(mfccs)
        # Delta-Delta coefficients
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Collect statistics for MFCCs and their deltas
        for i in range(mfccs.shape[0]):
            # MFCCs
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])
            # Delta MFCCs
            features[f'delta_mfcc_{i+1}_mean'] = np.mean(delta_mfccs[i])
            features[f'delta_mfcc_{i+1}_std'] = np.std(delta_mfccs[i])
            # Delta-Delta MFCCs
            features[f'delta2_mfcc_{i+1}_mean'] = np.mean(delta2_mfccs[i])
            features[f'delta2_mfcc_{i+1}_std'] = np.std(delta2_mfccs[i])
        
        # Spectral Flux
        spectral_flux = librosa.onset.onset_strength(y=audio_signal, sr=sr, hop_length=512)
        features['spectral_flux_mean'] = np.mean(spectral_flux)
        features['spectral_flux_std'] = np.std(spectral_flux)
        
        # Spectral Roll-off
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_signal, sr=sr, hop_length=512)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Zero-Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_signal, hop_length=512)
        features['zero_crossing_rate_mean'] = np.mean(zero_crossing_rate)
        features['zero_crossing_rate_std'] = np.std(zero_crossing_rate)
        
        # Energy Entropy
        stft = np.abs(librosa.stft(audio_signal, n_fft=2048, hop_length=512))
        energy = np.sum(stft ** 2, axis=0)
        energy_norm = energy / np.sum(energy)
        energy_entropy = -np.sum(energy_norm * np.log2(energy_norm + 1e-12))  # Add epsilon to avoid log(0)
        features['energy_entropy'] = energy_entropy
        
    except Exception as e:
        print(f"Error in spectral analysis: {e}")
        # Initialize all features to zero in case of error
        feature_names = ['spectral_slope',
                         'spectral_centroid',
                         'spectral_flux_mean', 'spectral_flux_std',
                         'spectral_rolloff_mean', 'spectral_rolloff_std',
                         'zero_crossing_rate_mean', 'zero_crossing_rate_std',
                         'energy_entropy']
        feature_names += [f'mfcc_{i+1}_mean' for i in range(n_mfcc)]
        feature_names += [f'mfcc_{i+1}_std' for i in range(n_mfcc)]
        feature_names += [f'delta_mfcc_{i+1}_mean' for i in range(n_mfcc)]
        feature_names += [f'delta_mfcc_{i+1}_std' for i in range(n_mfcc)]
        feature_names += [f'delta2_mfcc_{i+1}_mean' for i in range(n_mfcc)]
        feature_names += [f'delta2_mfcc_{i+1}_std' for i in range(n_mfcc)]
        features.update({name: 0 for name in feature_names})
    return features

def _extract_hnr_features(sound):
    """Extract Harmonics-to-Noise Ratio (HNR) using autocorrelation method with required parameters."""
    features = {}
    pitch_floor = 60  # Adjusted pitch floor for elderly speakers
    silence_threshold = 0.1  # Typical silence threshold value
    periods_per_window = 3.0  # Periods per window

    try:
        # Use autocorrelation method with silence threshold and periods per window
        hnr = call(sound, "To Harmonicity (ac)", 0.01, pitch_floor, silence_threshold, periods_per_window)

        if hnr:
            frame_count = hnr.get_number_of_frames()

            if frame_count > 0:
                hnr_mean = call(hnr, "Get mean", 0, 0)

                if hnr_mean is None or isinstance(hnr_mean, float) and np.isnan(hnr_mean):
                    hnr_mean = 0
                features['hnr_mean'] = hnr_mean
            else:
                features['hnr_mean'] = 0
        else:
            features['hnr_mean'] = 0

    except Exception as e:
        features['hnr_mean'] = 0

    return features

def _extract_pitch_features(sound):
    """Extract pitch features including mean and standard deviation of pitch."""
    features = {}
    try:
        pitch = call(sound, "To Pitch", 0.01, 75, 500)
        features['pitch_mean'] = call(pitch, "Get mean", 0, 0, "Hertz")
        features['pitch_std'] = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    except Exception as e:
        print(f"Error in pitch analysis: {e}")
        features['pitch_mean'] = 0
        features['pitch_std'] = 0
    return features

def _extract_additional_features(audio_data):
    """Extract additional features such as Asymmetry and TrajIntra."""
    features = {}
    try:
        # Asymmetry Feature Calculation
        features['Asymmetry'] = np.mean((audio_data - np.mean(audio_data))**3) / (np.std(audio_data)**3)

        # TrajIntra Feature Calculation
        features['TrajIntra'] = np.mean(np.abs(np.diff(audio_data)))

    except Exception as e:
        print(f"Error in additional feature calculation: {e}")
        features['Asymmetry'] = 0
        features['TrajIntra'] = 0
    return features

def _extract_avqi_hnr_sd(audio_data, sr):
    """Extract AVQI HNR_sd feature using RMS of the audio signal."""
    features = {}
    try:
        # Calculate RMS energy using librosa
        rms = librosa.feature.rms(y=audio_data)
        features['AVQI_HNR_sd'] = np.std(rms)
    except Exception as e:
        print(f"Error in AVQI HNR_sd calculation: {e}")
        features['AVQI_HNR_sd'] = 0
    return features

def _extract_amplitude_maximum_difference_mean(audio_data):
    """Extract Amplitude Maximum Difference Mean."""
    features = {}
    try:
        features['Amplitude_Maximum_Difference_mean'] = np.mean(np.abs(np.diff(audio_data)))
    except Exception as e:
        print(f"Error in Amplitude Maximum Difference Mean calculation: {e}")
        features['Amplitude_Maximum_Difference_mean'] = 0
    return features

def _extract_amplitude_minimum(audio_data):
    """Extract Amplitude Minimum."""
    features = {}
    try:
        features['Amplitude_Minimum'] = np.min(audio_data)
    except Exception as e:
        print(f"Error in Amplitude Minimum calculation: {e}")
        features['Amplitude_Minimum'] = 0
    return features
