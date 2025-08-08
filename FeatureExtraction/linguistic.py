import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re
from collections import Counter
import nltk
from nltk.util import ngrams
import librosa


# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class SpeechLinguisticAnalyzer:
    def __init__(self):
        # Initialize Wav2Vec2 model for Spanish
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-spanish")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-spanish")

    def transcribe_audio(self, audio_data, sample_rate):
        """
        Transcribe audio using Wav2Vec2 model.
        """
        # Resample if needed (Wav2Vec2 expects 16kHz)
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)

        # Prepare input
        inputs = self.processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)

        # Get logits
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        # Get predicted ids and decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]

        return transcription

    def extract_linguistic_features(self, transcription, reference_text=None):
        """
        Extract linguistic features from transcription.
        """
        features = {}

        # Clean transcription
        cleaned_transcription = self._clean_text(transcription)
        words = cleaned_transcription.split()

        # Basic word statistics
        features['total_words'] = len(words)

        # Word repetitions
        word_counts = Counter(words)
        repeated_words = {word: count for word, count in word_counts.items() if count > 1}
        features['number_repeated_words'] = len(repeated_words)
        features['total_word_repetitions'] = sum(count-1 for count in repeated_words.values())

        # Phrase repetitions (2-4 word sequences)
        for n in range(2, 5):
            phrase_counts = Counter(self._get_ngrams(words, n))
            repeated_phrases = {' '.join(phrase): count
                              for phrase, count in phrase_counts.items()
                              if count > 1}
            features[f'repeated_{n}word_phrases'] = len(repeated_phrases)

        # Hesitations and fillers
        hesitation_patterns = [
            r'\b(eh|em|mm|um|uh|este|pues|como|o sea)\b',
            r'(\.\.\.|…)',
            r'\([^\)]*hesitation[^\)]*\)',
            r'\([^\)]*pause[^\)]*\)'
        ]

        features['hesitation_count'] = sum(
            len(re.findall(pattern, transcription, re.IGNORECASE))
            for pattern in hesitation_patterns
        )

        # False starts and repairs
        false_start_patterns = [
            r'\b\w+\-\s',  # Words cut off with hyphen
            r'\([^\)]*false start[^\)]*\)',
            r'\([^\)]*repair[^\)]*\)'
        ]

        features['false_starts'] = sum(
            len(re.findall(pattern, transcription, re.IGNORECASE))
            for pattern in false_start_patterns
        )

        # Unintelligible words
        unintelligible_patterns = [
            r'\*+',  # Asterisks often used for unintelligible speech
            r'\[unintelligible\]',
            r'\[unclear\]',
            r'\([^\)]*unintelligible[^\)]*\)'
        ]

        features['unintelligible_words'] = sum(
            len(re.findall(pattern, transcription, re.IGNORECASE))
            for pattern in unintelligible_patterns
        )

        # If reference text is provided, calculate reading accuracy
        if reference_text:
            features.update(self._calculate_reading_accuracy(cleaned_transcription, reference_text))

        return features

    def _clean_text(self, text):
        """Clean text by removing special characters and extra whitespace."""
        text = re.sub(r'\([^\)]*\)', '', text)  # Remove parenthetical notes
        text = re.sub(r'[^\w\s]', '', text)     # Remove punctuation
        text = re.sub(r'\s+', ' ', text)        # Normalize whitespace
        return text.lower().strip()

    def _get_ngrams(self, words, n):
        """Generate n-grams from word list."""
        return list(ngrams(words, n))

    def _calculate_reading_accuracy(self, transcription, reference):
        """
        Calculate reading accuracy metrics by comparing with reference text.
        """
        features = {}

        # Clean reference text
        reference = self._clean_text(reference)

        # Split into words
        trans_words = transcription.split()
        ref_words = reference.split()

        # Calculate Levenshtein distance
        features['levenshtein_distance'] = self._levenshtein_distance(trans_words, ref_words)

        # Calculate Word Error Rate (WER)
        features['word_error_rate'] = features['levenshtein_distance'] / len(ref_words)

        # Calculate correct words
        correct_words = sum(1 for t, r in zip(trans_words, ref_words) if t == r)
        features['correct_words'] = correct_words
        features['word_accuracy'] = correct_words / len(ref_words) if ref_words else 0

        return features

    def _levenshtein_distance(self, s1, s2):
        """Calculate Levenshtein distance between two word lists."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]