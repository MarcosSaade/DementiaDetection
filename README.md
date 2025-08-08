# Dementia Detection through Machine Learning

Open-source code accompanying our research on automatic dementia detection from speech. This repository includes:

- Feature extraction (acoustic, prosodic, optional linguistic)
- Dataset preparation and visualization utilities
- Classifiers (SVM, ANN, XGBoost) and evaluation scripts

Note: Some modules (e.g., linguistic ASR) are optional and heavy; see installation notes below.

## Repository structure

- FeatureExtraction/
  - preprocessing.py: audio I/O, normalization, noise reduction, VAD helpers and plotting
  - accoustic.py: acoustic feature extraction (Praat/parselmouth, librosa, scipy)
  - prosodic.py: prosodic and timing features (VAD, rhythm, tempo, energy)
  - main.py: batch processing CLI to extract features from a folder of .wav files
- DataVisualization/
  - main.py: plots for class distributions and per-feature distributions
- Classifier/
  - paper.py: train/evaluate SVM, ANN and XGBoost and generate plots
  - missclassified.py: SVM pipeline with misclassification reporting

If you used an earlier version, keep in mind we retain the misspelled file name `accoustic.py` for backward compatibility.

## Installation

Python 3.10+ recommended. Create a virtual environment and install dependencies:

```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional/extra components:

- Linguistic/ASR features (disabled by default): install extras
  - PyTorch (CPU/GPU per your system)
  - transformers, datasets, sentencepiece
  - nltk (download punkt and averaged_perceptron_tagger)

See comments inside `FeatureExtraction/main.py` to enable linguistic features.

## Usage

1) Extract features from a folder with .wav files

```
python FeatureExtraction/main.py \
  --input "path/to/wavs" \
  --output "savedFeatures/features.csv" \
  --plots "Graphs/silence" \
  --plot 0
```

2) Visualize dataset

```
python DataVisualization/main.py \
  --input "savedFeatures/features.csv" \
  --output "Graphs"
```

3) Train and evaluate classifiers (reproduces paper figures)

- SVM + detailed error analysis
```
python Classifier/missclassified.py \
  --features "savedFeatures/features.csv" \
  --graphs "paper/graphs"
```

- SVM, ANN, XGBoost comparison
```
python Classifier/paper.py \
  --features "savedFeatures/features.csv" \
  --graphs "paper/graphs"
```

## Expected filename format

For automatic metadata parsing, audio files are expected to follow:

```
<ClassLabel>-<Gender>-<Age>-<ID>.wav
```

Examples: `AD-M-78-001.wav`, `HC-W-70-012.wav`

Parsed fields are saved as columns in the output CSV.

## Reproducibility

- Random seeds are set where applicable (e.g., 42)
- All plots are saved under the provided graphs directory
- Best feature subset is hardcoded in classifier scripts to match the paper

## Notes

- VAD relies on WebRTC (`webrtcvad`) which expects 16kHz (or 8/32/48kHz) and 16-bit PCM frames. The preprocessing pipeline resamples and converts as needed.
- Parselmouth/Praat is used for acoustic measures (jitter, shimmer, formants, HNR). Make sure `praat-parselmouth` is installed from PyPI.

## Data availability

Due to dataset usage policies, we cannot redistribute the raw audio/data in this repository. Researchers can request access to DementiaBank datasets via: https://talkbank.org/dementia/

## License

The license for this repository will be set via GitHub’s License feature. See the repository’s License tab for details.

## Citation

If you use this code, please cite the accompanying paper.
