import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa
import webrtcvad
import scipy.signal
import scipy.stats

def extract_prosodic_features(audio_data, sr):
    """
    Extract comprehensive prosodic features from audio data.

    Args:
        audio_data (array): Audio signal
        sr (int): Sampling rate

    Returns:
        dict: Dictionary containing all extracted features
    """

    features = {}
    sound = parselmouth.Sound(audio_data, sr)
        
    # 1. Timing and Speech/Silence Features
    features.update(_extract_timing_features(audio_data, sr))

    # 2. Rhythm Features
    features.update(_extract_rhythm_features(sound))

    # 3. Tempo and Beat Features
    features.update(_extract_tempo_beat_features(audio_data, sr))

    # 4. Energy-Based Temporal Features
    features.update(_extract_energy_temporal_features(audio_data, sr))

    return features

def _extract_timing_features(audio_data, sr):
    """Extract timing-related features using VAD."""
    features = {}

    # Prepare audio for VAD
    vad_audio = librosa.resample(audio_data, orig_sr=sr, target_sr=16000) if sr != 16000 else audio_data

    # Get speech and silence segments
    silence_segments, speech_segments = extract_silences(vad_audio, 16000)

    # Filter silence segments between speech
    filtered_silence_segments = [
        (start, end) for start, end in silence_segments
        if speech_segments and start > speech_segments[0][0] and end < speech_segments[-1][1]
    ]

    # Calculate pause durations
    pause_durations = [end - start for start, end in filtered_silence_segments]

    # Calculate speech segment durations
    speech_durations = [end - start for start, end in speech_segments]

    total_duration = len(audio_data) / sr
    total_speech_duration = sum(speech_durations)

    features.update({
        # Existing timing features
        'total_duration': total_duration,
        'silence_count': len(filtered_silence_segments),
        'speech_segment_count': len(speech_segments),
        'total_silence_duration': sum(pause_durations),
        'total_speech_duration': total_speech_duration,
        'speech_rate': len(speech_segments) / total_duration,
        'articulation_rate': len(speech_segments) / total_speech_duration if total_speech_duration > 0 else 0,

        # Pause-related features
        'mean_pause_duration': np.mean(pause_durations) if pause_durations else 0,
        'std_pause_duration': np.std(pause_durations) if pause_durations else 0,
        'max_pause_duration': np.max(pause_durations) if pause_durations else 0,
        'min_pause_duration': np.min(pause_durations) if pause_durations else 0,
        'pause_rate': len(pause_durations) / total_speech_duration if total_speech_duration > 0 else 0,
        'pause_ratio': sum(pause_durations) / total_speech_duration if total_speech_duration > 0 else 0,

        # Speech features
        'mean_speech_duration': np.mean(speech_durations) if speech_durations else 0,
        'std_speech_duration': np.std(speech_durations) if speech_durations else 0,
        'max_speech_duration': np.max(speech_durations) if speech_durations else 0,
        'speech_duration_range': (np.max(speech_durations) - np.min(speech_durations)) if speech_durations else 0,
        'speech_duration_coefficient_of_variation': (np.std(speech_durations) / np.mean(speech_durations)) if speech_durations and np.mean(speech_durations) > 0 else 0,
        'speech_to_pause_ratio': total_speech_duration / sum(pause_durations) if pause_durations and sum(pause_durations) > 0 else float('inf')
    })

    return features

def _extract_rhythm_features(sound):
    """Extract rhythm-related features including PVI measures."""
    features = {}
    try:
        intensity = sound.to_intensity()
        intensity_values = intensity.values[0]
        time_step = intensity.get_time_step()

        peaks, _ = scipy.signal.find_peaks(
            intensity_values,
            height=np.mean(intensity_values),
            distance=int(0.06 / time_step)
        )

        if len(peaks) > 1:
            intervals = np.diff(peaks) * time_step
            differences = np.abs(np.diff(intervals))
            means = np.mean([intervals[:-1], intervals[1:]], axis=0)

            features.update({
                'rPVI': np.mean(differences),
                'nPVI': np.mean((differences / means) * 100)
            })
        else:
            features.update({'rPVI': 0, 'nPVI': 0})

    except Exception as e:
        print(f"Error in rhythm analysis: {e}")
        features.update({'rPVI': 0, 'nPVI': 0})

    return features

def _extract_tempo_beat_features(audio_data, sr):
    """Extract tempo and beat-related features."""
    features = {}
    try:
        # Estimate tempo and beat frames
        tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sr, hop_length=512, trim=False)

        # Convert beat frames to time
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)

        # Number of beats
        num_beats = len(beat_frames)
        features['tempo_bpm'] = tempo
        features['num_beats'] = num_beats

        if num_beats > 1:
            # Inter-beat intervals (IBI)
            ibi = np.diff(beat_times)
            features['ibi_mean'] = np.mean(ibi)
            features['ibi_std'] = np.std(ibi)
            features['ibi_variance'] = np.var(ibi)
            features['ibi_skewness'] = scipy.stats.skew(ibi)
            features['ibi_kurtosis'] = scipy.stats.kurtosis(ibi)
        else:
            features.update({
                'ibi_mean': 0,
                'ibi_std': 0,
                'ibi_variance': 0,
                'ibi_skewness': 0,
                'ibi_kurtosis': 0
            })

    except Exception as e:
        print(f"Error in tempo and beat analysis: {e}")
        features.update({
            'tempo_bpm': 0,
            'num_beats': 0,
            'ibi_mean': 0,
            'ibi_std': 0,
            'ibi_variance': 0,
            'ibi_skewness': 0,
            'ibi_kurtosis': 0
        })

    return features

def _extract_energy_temporal_features(audio_data, sr):
    """Extract energy-based temporal features."""
    features = {}
    try:
        # Compute Short-Time Energy (STE)
        frame_length = 1024
        hop_length = 512
        energy = np.array([
            np.sum(np.abs(audio_data[i:i+frame_length]**2))
            for i in range(0, len(audio_data), hop_length)
        ])

        # Mean Energy
        features['energy_mean'] = np.mean(energy) if len(energy) > 0 else 0

        # Standard Deviation of Energy
        features['energy_std'] = np.std(energy) if len(energy) > 0 else 0

        # Energy Variability (Coefficient of Variation)
        features['energy_cv'] = (features['energy_std'] / features['energy_mean']) if features['energy_mean'] > 0 else 0

        # Energy Entropy
        energy_norm = energy / np.sum(energy) if np.sum(energy) > 0 else energy
        energy_entropy = -np.sum(energy_norm * np.log2(energy_norm + 1e-12))  # Add epsilon to avoid log(0)
        features['energy_entropy'] = energy_entropy

    except Exception as e:
        print(f"Error in energy temporal analysis: {e}")
        features.update({
            'energy_mean': 0,
            'energy_std': 0,
            'energy_cv': 0,
            'energy_entropy': 0
        })

    return features

def extract_silences(audio_data, sr, frame_duration=30, aggressiveness=3, min_silence_duration=0.5, min_speech_duration=0.2):
    """
    Perform Voice Activity Detection using WebRTC VAD with minimum duration thresholds.

    Args:
        audio_data: numpy array of audio samples
        sr: sample rate (must be 8000, 16000, 32000, or 48000)
        frame_duration: frame duration in ms (10, 20, or 30)
        aggressiveness: VAD aggressiveness (0-3)
        min_silence_duration: minimum duration in seconds for a silence segment to be counted
        min_speech_duration: minimum duration in seconds for a speech segment to be counted

    Returns:
        silence_segments (list of tuples): List of (start_time, end_time) for silences
        speech_segments (list of tuples): List of (start_time, end_time) for speech
    """
    if sr not in [8000, 16000, 32000, 48000]:
        raise ValueError("Sample rate must be 8000, 16000, 32000, or 48000")
    if frame_duration not in [10, 20, 30]:
        raise ValueError("Frame duration must be 10, 20, or 30 ms")
    if not (0 <= aggressiveness <= 3):
        raise ValueError("Aggressiveness must be between 0 and 3")

    # Convert to int16 if needed
    if audio_data.dtype != np.int16:
        audio_data = convert_to_int16(audio_data)

    vad = webrtcvad.Vad(aggressiveness)
    frame_size = int(sr * frame_duration / 1000)
    num_frames = len(audio_data) // frame_size

    # Use temporary lists to store all segments before filtering
    temp_speech_segments = []
    temp_silence_segments = []
    voiced = False
    start_time = 0

    for i in range(num_frames):
        start_idx = i * frame_size
        end_idx = start_idx + frame_size

        frame = audio_data[start_idx:end_idx]
        if len(frame) != frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)), 'constant')

        try:
            is_speech = vad.is_speech(frame.tobytes(), sr)
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            continue

        current_time = start_idx / sr

        if is_speech != voiced:
            if is_speech:
                start_time = current_time
                if len(temp_silence_segments) == 0 and start_time > 0:
                    temp_silence_segments.append((0, start_time))
            else:
                temp_speech_segments.append((start_time, current_time))
            voiced = is_speech

    # Handle the final segment
    if voiced:
        temp_speech_segments.append((start_time, len(audio_data) / sr))
    elif len(temp_speech_segments) > 0:
        temp_silence_segments.append((temp_speech_segments[-1][1], len(audio_data) / sr))
    elif len(temp_speech_segments) == 0:
        temp_silence_segments.append((0, len(audio_data) / sr))

    # Add intermediate silence segments
    if len(temp_speech_segments) > 1:
        for i in range(1, len(temp_speech_segments)):
            temp_silence_segments.append((temp_speech_segments[i-1][1], temp_speech_segments[i][0]))

    # Filter segments based on duration thresholds
    speech_segments = []
    silence_segments = []

    for start, end in temp_speech_segments:
        duration = end - start
        if duration >= min_speech_duration:
            speech_segments.append((start, end))

    for start, end in temp_silence_segments:
        duration = end - start
        if duration >= min_silence_duration:
            silence_segments.append((start, end))

    # Sort the filtered segments
    speech_segments.sort(key=lambda x: x[0])
    silence_segments.sort(key=lambda x: x[0])

    # Merge adjacent silence segments if any gaps were created by filtering
    if len(silence_segments) > 1:
        merged_silence = []
        current_start, current_end = silence_segments[0]

        for start, end in silence_segments[1:]:
            if start <= current_end:  # Overlapping or adjacent segments
                current_end = max(current_end, end)
            else:
                merged_silence.append((current_start, current_end))
                current_start, current_end = start, end

        merged_silence.append((current_start, current_end))
        silence_segments = merged_silence

    return silence_segments, speech_segments

def convert_to_int16(audio_data):
    """Convert float audio data to 16-bit integers."""
    audio_int16 = np.clip(audio_data * 32768, -32768, 32767).astype(np.int16)
    return audio_int16
