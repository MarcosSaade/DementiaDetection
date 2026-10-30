# Detecting Alzheimer's Disease from Spanish Speech

Code for my paper on screening for Alzheimer's disease from short recordings of Spanish read speech. The pipeline extracts 80+ interpretable acoustic, timing and complexity features, balances the classes with SMOTE-ENN, and classifies with an RBF SVM.

**Result:** 80.1% +/- 2.2% accuracy, 81.2% weighted F1 and ROC AUC 0.90 in 5-fold cross-validation on 361 recordings.

Paper (PDF): [marcossaade.github.io/dementiaDetectionML.pdf](https://marcossaade.github.io/dementiaDetectionML.pdf) · Project page: [marcossaade.github.io/dementia](https://marcossaade.github.io/dementia)

## Data

361 recordings of the same paragraph read aloud in Spanish, from the Ivanova corpus in [DementiaBank](https://talkbank.org/dementia/): 74 with Alzheimer's disease and 287 without (197 healthy controls, 90 with mild cognitive impairment), ages 50 to 96. The task is Alzheimer's vs. non-Alzheimer's.

The dataset can't be redistributed. Researchers can request access through DementiaBank.

## Method

**Preprocessing.** Mono, 16 kHz, amplitude normalization, noise reduction, and WebRTC voice activity detection to separate speech from pauses.

**Features (80+).**
- *Spectral:* 13 MFCCs (mean and standard deviation), formants F1 to F4, spectral centroid and slope.
- *Voice quality:* pitch, jitter, shimmer, harmonics-to-noise ratio, cepstral peak prominence.
- *Timing and rhythm:* total and speech duration, pause counts and lengths, speech-to-pause ratio, articulation rate, rhythm indices (nPVI, rPVI).
- *Complexity:* Higuchi fractal dimension.

**Model.** Recursive feature elimination reduced the features to 15. The final pipeline is SMOTE-ENN, then scaling, then an RBF SVM (C = 10, gamma = scale) tuned by grid search. SMOTE-ENN runs inside the pipeline, so synthetic samples are only created from the training part of each cross-validation fold and never leak into the test fold.

## Results

5-fold stratified cross-validation:

| Model | Accuracy |
|---|---|
| **RBF SVM** | **80.1% +/- 2.2%** (weighted F1 81.2%, ROC AUC 0.90, PR AUC 0.69) |
| Neural network (MLP) | 78% |
| XGBoost | 76% |

The most important features by permutation importance were the mean of MFCC 8, the range of F2, the mean of MFCC 2 and the spectral centroid.

**Limitations.** The Alzheimer's class is small (74 recordings), and precision on that class is 0.62, so the model produces a fair number of false positives. It is a screening aid to study, not a diagnostic tool. An end-to-end deep learning model on raw audio was tried first and performed near chance, which is why the final model uses engineered features.

## Repository structure

```
FeatureExtraction/
  preprocessing.py     audio loading, normalization, noise reduction, VAD
  acoustic.py          spectral and voice-quality features (Praat/parselmouth, librosa)
  prosodic.py          timing, rhythm and pause features
  main.py              batch feature extraction from a folder of .wav files
DataVisualization/
  main.py              class and per-feature distribution plots
Classifier/
  paper.py             SVM, MLP and XGBoost comparison; reproduces the paper's figures
  misclassified.py     SVM pipeline with per-recording error analysis
```

## Usage

Python 3.10+:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# 1. Extract features
python FeatureExtraction/main.py --input path/to/wavs --output savedFeatures/features.csv

# 2. Plot the dataset (optional)
python DataVisualization/main.py --input savedFeatures/features.csv --output Graphs

# 3. Train and evaluate (reproduces the paper's figures)
python Classifier/paper.py --features savedFeatures/features.csv --graphs paper/graphs
python Classifier/misclassified.py --features savedFeatures/features.csv --graphs paper/graphs
```

Audio files should be named `<Label>-<Gender>-<Age>-<ID>.wav` (for example `AD-M-78-001.wav` or `HC-W-70-012.wav`); these fields are parsed into the output CSV.

Notes:
- The 15 selected features are hardcoded in the classifier scripts to match the paper, and random seeds are fixed.
- WebRTC VAD requires 16-bit PCM at 8, 16, 32 or 48 kHz; the preprocessing step resamples as needed.
- Install Praat bindings from PyPI as `praat-parselmouth`.

## License

MIT, see [LICENSE](LICENSE).

## Citation

If you use this code, please cite the paper:

> Marcos Saade Romano. *Alzheimer's Disease Detection Through Automated Speech Analysis in Spanish-Speaking Populations through Feature Engineering and Machine Learning.* Tecnológico de Monterrey.
