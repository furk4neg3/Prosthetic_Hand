#!/usr/bin/env python3
"""
prepare_samples_for_pi.py
=========================

Run this in Google Colab AFTER training your DANN model.

What it does:
  1. Loads NINAPRO DB1 data for Subject 1 (or any subject you choose)
  2. Loads the trained StandardScaler
  3. For each of the 29 movements, selects 10 random RAW sEMG windows
     (after bandpass filtering, but BEFORE StandardScaler normalization)
  4. Saves everything into a single compact .npz file:
       - samples:        (29, 10, 50, 10)  raw sEMG windows
       - encoded_labels:  (29,)             model class indices (0-28)
       - original_labels: (29,)             NINAPRO original labels
       - scaler_mean:     (10,)             StandardScaler mean per channel
       - scaler_scale:    (10,)             StandardScaler scale per channel
  5. Saves to Google Drive for download to the Pi

Usage (in Colab):
  %run prepare_samples_for_pi.py

Or copy cells into a notebook and run sequentially.
"""

# =============================================================================
# CELL 1: Mount Drive & Imports
# =============================================================================
from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt
import os
import glob
import re
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

print("✅ Imports complete.")

# =============================================================================
# CELL 2: Configuration — MUST match training config exactly
# =============================================================================
class Config:
    DATASET_PATH = '/content/drive/MyDrive/NINAPRO'  # Path to NINAPRO data
    SAMPLING_RATE = 100
    NUM_CHANNELS = 10
    USE_CHANNELS = list(range(10))  # All 10 channels

    # Excluded labels (wrist B9-B17 + some E3 movements)
    EXCLUDED_LABELS = list(range(21, 30)) + [36, 37] + list(range(41, 53))

    WINDOW_SIZE = 50     # 500ms at 100Hz
    OVERLAP = 0.80
    LOWCUT = 20.0
    HIGHCUT = 45.0
    SEED = 42

    # Number of samples to keep per movement
    SAMPLES_PER_MOVEMENT = 10

    # Which subject's data to use
    SUBJECT_ID = 1

    # Paths
    MODEL_DIR = '/content/drive/MyDrive/models_grad_project'
    SCALER_PATH = os.path.join(MODEL_DIR, 'scaler_subject_01_29.pkl')
    OUTPUT_DIR = os.path.join(MODEL_DIR, 'pi_deployment')
    OUTPUT_FILE = 'pi_samples.npz'

config = Config()
np.random.seed(config.SEED)

print(f"Config loaded:")
print(f"  Subject: {config.SUBJECT_ID}")
print(f"  Channels: {config.NUM_CHANNELS}")
print(f"  Window size: {config.WINDOW_SIZE} ({config.WINDOW_SIZE / config.SAMPLING_RATE * 1000:.0f}ms)")
print(f"  Bandpass: {config.LOWCUT}–{config.HIGHCUT} Hz")
print(f"  Samples per movement: {config.SAMPLES_PER_MOVEMENT}")
print(f"  Excluded labels: {config.EXCLUDED_LABELS}")

# =============================================================================
# CELL 3: Data Loading Classes (same as training notebook)
# =============================================================================
class NinaProLoader:
    """Loads raw NINAPRO DB1 .mat files for a given subject."""
    def __init__(self, path, use_channels=list(range(10))):
        self.path = path
        self.use_channels = use_channels
        self.subjects = self._find_subjects()
        print(f'Found {len(self.subjects)} subjects')

    def _find_subjects(self):
        files = glob.glob(os.path.join(self.path, '**', '*.mat'), recursive=True)
        subjects = set()
        for f in files:
            name = os.path.basename(f).upper()
            match = re.match(r'^S(\d+)_A1_E[123]\.MAT$', name)
            if match:
                subjects.add(int(match.group(1)))
        return sorted(subjects)

    def _find_exact_file(self, subj_id, ex):
        target_names = [
            f"S{subj_id}_A1_{ex}.mat",
            f"s{subj_id}_a1_{ex.lower()}.mat",
            f"S{subj_id:02d}_A1_{ex}.mat",
            f"s{subj_id:02d}_a1_{ex.lower()}.mat",
        ]
        for target in target_names:
            matches = glob.glob(os.path.join(self.path, '**', target), recursive=True)
            if matches:
                return sorted(matches)[0]
        return None

    def load_subject(self, subj_id):
        offsets = {'E1': 0, 'E2': 12, 'E3': 29}
        emg_all, stim_all = [], []

        for ex in ['E1', 'E2', 'E3']:
            file_path = self._find_exact_file(subj_id, ex)
            if file_path is None:
                print(f"⚠️ Missing file for Subject {subj_id}, {ex}")
                continue

            mat = scipy.io.loadmat(file_path)
            emg = mat['emg'][:, self.use_channels]
            stim = mat['stimulus'].flatten().copy()
            stim[stim > 0] += offsets[ex]

            emg_all.append(emg)
            stim_all.append(stim)

        if not emg_all:
            return None

        return {
            'id': subj_id,
            'emg': np.vstack(emg_all),
            'stim': np.hstack(stim_all)
        }


class Preprocessor:
    """
    Preprocesses raw EMG data into windows.

    IMPORTANT: For Pi deployment, we apply bandpass filtering
    but NOT StandardScaler normalization. The scaler params
    are saved separately so the Pi applies them at inference time.
    """
    def __init__(self, fs=100, lowcut=20, highcut=45, win_size=50, overlap=0.80,
                 excluded_labels=None, apply_scaler=False, scaler=None):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.win_size = win_size
        self.step = int(win_size * (1 - overlap))
        self.excluded_labels = excluded_labels or []
        self.apply_scaler = apply_scaler
        self.scaler = scaler

    def bandpass(self, emg):
        nyq = self.fs / 2
        b, a = butter(4, [self.lowcut / nyq, self.highcut / nyq], btype='band')
        return np.array([
            filtfilt(b, a, emg[:, ch])
            for ch in range(emg.shape[1])
        ]).T

    def process(self, data):
        emg = np.nan_to_num(data['emg'])
        emg = self.bandpass(emg)

        # Optionally apply scaler (for training pipeline compatibility)
        if self.apply_scaler and self.scaler is not None:
            emg = self.scaler.transform(emg)

        labels = data['stim']
        n_win = (len(emg) - self.win_size) // self.step + 1

        windows, win_labels = [], []
        for i in range(n_win):
            s = i * self.step
            e = s + self.win_size

            vals, counts = np.unique(labels[s:e], return_counts=True)
            label = vals[np.argmax(counts)]

            # Skip rest (0) and excluded labels
            if label == 0 or label in self.excluded_labels:
                continue

            windows.append(emg[s:e])
            win_labels.append(label)

        return np.array(windows), np.array(win_labels)

print("✅ Loader and Preprocessor defined.")

# =============================================================================
# CELL 4: Load Subject Data & Extract Windows
# =============================================================================
loader = NinaProLoader(config.DATASET_PATH, use_channels=config.USE_CHANNELS)
prep = Preprocessor(
    config.SAMPLING_RATE, config.LOWCUT, config.HIGHCUT,
    config.WINDOW_SIZE, config.OVERLAP,
    excluded_labels=config.EXCLUDED_LABELS,
    apply_scaler=False  # ← We save RAW (bandpass-filtered only) windows
)

print(f"\nLoading Subject {config.SUBJECT_ID}...")
data = loader.load_subject(config.SUBJECT_ID)
if data is None:
    raise ValueError(f"Could not load data for Subject {config.SUBJECT_ID}!")

windows, labels = prep.process(data)
print(f"✅ Extracted {len(windows)} windows, shape: {windows.shape}")
print(f"   Unique labels: {sorted(np.unique(labels))}")

# =============================================================================
# CELL 5: Encode Labels & Select Samples
# =============================================================================
le = LabelEncoder()
encoded = le.fit_transform(labels)
n_classes = len(le.classes_)

print(f"\n=== Label Mapping ===")
print(f"Number of classes: {n_classes}")
print(f"Original labels: {list(le.classes_)}")
print(f"Encoded range: 0–{n_classes - 1}")

# Select 10 random windows per movement
samples = np.zeros((n_classes, config.SAMPLES_PER_MOVEMENT,
                     config.WINDOW_SIZE, config.NUM_CHANNELS), dtype=np.float32)

for class_idx in range(n_classes):
    mask = (encoded == class_idx)
    class_windows = windows[mask]
    original_label = le.classes_[class_idx]

    if len(class_windows) < config.SAMPLES_PER_MOVEMENT:
        print(f"⚠️ Class {class_idx} (label {original_label}): "
              f"only {len(class_windows)} windows, need {config.SAMPLES_PER_MOVEMENT}")
        # If not enough, repeat with replacement
        indices = np.random.choice(len(class_windows), config.SAMPLES_PER_MOVEMENT, replace=True)
    else:
        indices = np.random.choice(len(class_windows), config.SAMPLES_PER_MOVEMENT, replace=False)

    samples[class_idx] = class_windows[indices].astype(np.float32)
    print(f"  Class {class_idx:2d} (label {original_label:2d}): "
          f"selected {config.SAMPLES_PER_MOVEMENT} from {len(class_windows)} windows")

print(f"\n✅ Samples array shape: {samples.shape}")

# =============================================================================
# CELL 6: Load Scaler & Extract Parameters
# =============================================================================
print(f"\nLoading scaler from: {config.SCALER_PATH}")
with open(config.SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

scaler_mean = scaler.mean_.astype(np.float32)
scaler_scale = scaler.scale_.astype(np.float32)

print(f"✅ Scaler loaded:")
print(f"   Mean shape: {scaler_mean.shape}, values: {scaler_mean}")
print(f"   Scale shape: {scaler_scale.shape}, values: {scaler_scale}")

# Verify scaler works correctly on a sample
test_window = samples[0, 0].copy()
# Manual transform
manual_result = (test_window - scaler_mean) / scaler_scale
# Sklearn transform
sklearn_result = scaler.transform(test_window)
# They should be identical
assert np.allclose(manual_result, sklearn_result, atol=1e-5), \
    "Scaler verification failed! Manual transform != sklearn transform"
print("✅ Scaler verification passed.")

# =============================================================================
# CELL 7: Save .npz File
# =============================================================================
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(config.OUTPUT_DIR, config.OUTPUT_FILE)

np.savez_compressed(
    output_path,
    samples=samples,                              # (29, 10, 50, 10)
    encoded_labels=np.arange(n_classes),           # (29,)
    original_labels=le.classes_.astype(np.int32),  # (29,)
    scaler_mean=scaler_mean,                       # (10,)
    scaler_scale=scaler_scale,                     # (10,)
)

file_size = os.path.getsize(output_path)
print(f"\n✅ Saved to: {output_path}")
print(f"   File size: {file_size / 1024:.1f} KB")
print(f"\n📋 Contents:")
print(f"   samples:         {samples.shape} (raw bandpass-filtered sEMG windows)")
print(f"   encoded_labels:  {np.arange(n_classes).shape}")
print(f"   original_labels: {le.classes_.shape}")
print(f"   scaler_mean:     {scaler_mean.shape}")
print(f"   scaler_scale:    {scaler_scale.shape}")

# =============================================================================
# CELL 8: Verification — Load back and check
# =============================================================================
print("\n--- Verification ---")
loaded = np.load(output_path)
for key in loaded.files:
    print(f"  {key}: shape={loaded[key].shape}, dtype={loaded[key].dtype}")

# Quick check: manual scaler transform on loaded data should match
test_sample = loaded['samples'][0, 0]
test_mean = loaded['scaler_mean']
test_scale = loaded['scaler_scale']
result = (test_sample - test_mean) / test_scale
assert np.allclose(result, sklearn_result, atol=1e-5), "Verification failed!"
print("✅ Verification passed!")

print(f"""
{'=' * 60}
📋 NEXT STEPS:
{'=' * 60}
1. Download these files from Google Drive to your Raspberry Pi:
   - {output_path}
   - {os.path.join(config.MODEL_DIR, 'dann_inference_all_new_29.tflite')}

2. Place them in a directory on your Pi, e.g.:
   /home/pi/prosthetic_hand/
     ├── dann_inference_all_new_29.tflite
     ├── pi_samples.npz
     └── prosthetic_hand.py

3. Install dependencies:
   pip install tflite-runtime numpy scipy adafruit-circuitpython-servokit

4. Run:
   python3 prosthetic_hand.py
{'=' * 60}
""")
