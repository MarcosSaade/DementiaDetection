import argparse
import glob
import os
from typing import Dict, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from accoustic import extract_acoustic_features
from prosodic import extract_prosodic_features, extract_silences
from preprocessing import load_audio, normalize_audio, reduce_noise, remove_extreme_peaks, plot_audio_segments


def extract_all_features(audio_data, sr, original_audio_data=None, reference_text=None):
    """
    Extract acoustic + prosodic features. Linguistic features are optional and meant for natural speech.

    Args:
        audio_data: Preprocessed audio signal (1D array)
        sr: Sampling rate (int)
        original_audio_data: Optional original audio before normalization (1D array)
        reference_text: Optional reference text for reading task (str)

    Returns:
        dict of extracted features
    """

    # Get acoustic features
    features = extract_acoustic_features(audio_data, sr, original_audio_data)

    # Get prosodic features
    prosodic_features = extract_prosodic_features(audio_data, sr)

    # Update features with prosodic features
    features.update(prosodic_features)

    return features


def extract_info_from_filename(filename: str) -> Tuple[Optional[int], Optional[str], Optional[int], Optional[str]]:
    """Extract (age, gender, id, class_label) from '<Class>-<Gender>-<Age>-<ID>.wav'."""
    try:
        base_name = os.path.splitext(os.path.basename(filename))[0]
        parts = base_name.split('-')
        if len(parts) >= 4:
            class_label = parts[0]
            gender = parts[1]
            age = int(parts[2]) if parts[2].isdigit() else None
            # allow non-pure numeric IDs by stripping non-digits
            id_digits = ''.join(ch for ch in parts[3] if ch.isdigit())
            id_number = int(id_digits) if id_digits else None
            return age, gender, id_number, class_label
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
    return None, None, None, None


def process_audio_files(audio_folder: str, output_path: str, plot_dir: Optional[str] = None, plot: bool = False,
                        sample_rate: int = 16000, reference_text: Optional[str] = None) -> None:
    """
    Process all .wav files in a folder and save extracted features to CSV. Optionally save plots.

    Args:
        audio_folder: directory containing .wav files
        output_path: path to output CSV
        plot_dir: directory to save per-file plots (optional)
        plot: whether to generate plots
        sample_rate: target sample rate for processing (default 16kHz)
        reference_text: optional text for linguistic comparisons (disabled by default)
    """
    if plot and plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    audio_files = sorted(glob.glob(os.path.join(audio_folder, "*.wav")))
    if not audio_files:
        print(f"No .wav files found in {audio_folder}")
        pd.DataFrame([]).to_csv(output_path, index=False)
        return

    all_features: list[Dict] = []

    for file in tqdm(audio_files, desc="Processing audio files"):
        filename = os.path.basename(file)
        try:
            audio_data, sr = load_audio(file, target_sr=sample_rate)
            original_audio_data = audio_data.copy()  # store original audio data

            # Preprocessing
            audio_data = normalize_audio(audio_data)
            audio_data = reduce_noise(audio_data, sr)
            audio_data = remove_extreme_peaks(audio_data)

            age, gender, id_number, class_label = extract_info_from_filename(file)
            silence_segments, speech_segments = extract_silences(audio_data, sr)

            # Plot only if 'plot' is True and 'plot_dir' is provided
            if plot and plot_dir:
                plot_path = os.path.join(plot_dir, filename.replace(".wav", ".png"))
                title = (f"Speech Detection Results - {filename}\nClass: {class_label}, "
                         f"Gender: {gender}, Age: {age}, ID: {id_number}")
                plot_audio_segments(audio_data, sr, silence_segments, speech_segments, title, save_path=plot_path)

            # Default reference text can be overridden via CLI
            features = extract_all_features(audio_data, sr, original_audio_data, reference_text)
            features.update({
                'filename': filename,
                'age': age,
                'gender': gender,
                'id': id_number,
                'class_label': class_label
            })
            all_features.append(features)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    pd.DataFrame(all_features).to_csv(output_path, index=False)
    print(f"Saved features for {len(all_features)} files to {output_path}")


# CLI -------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Extract acoustic/prosodic features from wav files.")
    p.add_argument('--input', '-i', required=True, help='Folder with .wav files')
    p.add_argument('--output', '-o', required=True, help='Output CSV path')
    p.add_argument('--plots', '-p', required=False, default=None, help='Folder to save plots (optional)')
    p.add_argument('--plot', type=int, default=0, help='Whether to generate plots (1=yes, 0=no)')
    p.add_argument('--sr', type=int, default=16000, help='Target sample rate')
    p.add_argument('--reference-text', type=str, default=(
        "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo "
        "de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. Una olla de algo más vaca "
        "que carnero, salpicón las más noches, duelos y quebrantos los sábados, lantejas los viernes, algún "
        "palomino de añadidura los domingos, consumían las tres partes de su hacienda."
    ), help='Reference text for linguistic comparison (only used if enabled inside code)')
    return p


if __name__ == '__main__':
    args = _build_arg_parser().parse_args()
    process_audio_files(
        audio_folder=args.input,
        output_path=args.output,
        plot_dir=args.plots,
        plot=bool(args.plot),
        sample_rate=args.sr,
        reference_text=args.reference_text,
    )
