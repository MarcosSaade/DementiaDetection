import numpy as np
import librosa
import noisereduce as nr
from matplotlib.patches import Patch
import matplotlib.pyplot as plt


def load_audio(file_path, target_sr=16000):
    """Load audio file, convert to mono, and resample to supported sample rate."""
    audio_data, sr = librosa.load(file_path, sr=target_sr, mono=True)
    return audio_data, target_sr

def normalize_audio(audio_data):
    """Normalize audio to a consistent RMS."""
    rms_target = 0.1
    current_rms = np.sqrt(np.mean(audio_data**2))
    scaling_factor = rms_target / current_rms if current_rms > 0 else 1
    return audio_data * scaling_factor

def reduce_noise(audio_data, sr):
    """Reduce background noise in the audio."""
    reduced_audio = nr.reduce_noise(y=audio_data, sr=sr)
    return reduced_audio

def remove_extreme_peaks(audio_data, k=7, reduction_ratio=0.99):
    """
    Remove extreme peaks from audio data.
    
    Parameters:
    - audio_data: 1D NumPy array of audio samples.
    - k: Multiplier for standard deviation to set the peak threshold.
    - reduction_ratio: Proportion to reduce the amplitude of detected peaks.

    Returns:
    - Processed audio data with extreme peaks reduced.
    """
    # Calculate standard deviation (mean is 0)
    std_dev = np.std(audio_data)
    
    # Define threshold for peak detection
    threshold = k * std_dev

    # Create a copy of the audio data to modify
    processed_audio = np.copy(audio_data)
    
    # Identify peaks above the threshold
    peaks = np.abs(processed_audio) > threshold
    
    # Reduce the amplitude of peaks
    processed_audio[peaks] *= (1 - reduction_ratio)

    return processed_audio

def filter_silence_segments(silence_segments, speech_segments):
    """Filter out initial and final silence segments."""
    if not silence_segments or not speech_segments:
        return []

    filtered_silence_segments = []
    for silence_start, silence_end in silence_segments:
        if silence_end <= speech_segments[0][0]:
            continue
        if silence_start >= speech_segments[-1][1]:
            continue
        filtered_silence_segments.append((silence_start, silence_end))

    return filtered_silence_segments

def plot_audio_segments(audio_data, sr, silence_segments, speech_segments, title="Audio Segments Visualization", save_path=None):
    """Plot audio waveform with color-coded speech and silence segments."""
    filtered_silence_segments = filter_silence_segments(silence_segments, speech_segments)

    plt.figure(figsize=(15, 8))
    time = np.arange(len(audio_data)) / sr
    plt.plot(time, audio_data, color='lightgray', alpha=0.5, label='Waveform')

    for start, end in filtered_silence_segments:
        start_idx = int(start * sr)
        end_idx = min(int(end * sr), len(audio_data))
        plt.plot(time[start_idx:end_idx], audio_data[start_idx:end_idx], color='red', alpha=0.7)

    for start, end in speech_segments:
        start_idx = int(start * sr)
        end_idx = min(int(end * sr), len(audio_data))
        plt.plot(time[start_idx:end_idx], audio_data[start_idx:end_idx], color='green', alpha=0.7)

    ymin, ymax = plt.ylim()
    for start, end in filtered_silence_segments:
        plt.axvspan(start, end, color='red', alpha=0.1)
    for start, end in speech_segments:
        plt.axvspan(start, end, color='green', alpha=0.1)

    plt.title(title)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend(handles=[Patch(facecolor='green', alpha=0.3, label='Speech'),
                        Patch(facecolor='red', alpha=0.3, label='Silence')],
               loc='upper right')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)

    # Save plot if save_path is provided
    if save_path:
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')

    plt.close()
