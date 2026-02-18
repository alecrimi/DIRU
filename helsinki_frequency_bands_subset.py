"""
Helsinki Neonatal EEG - Raw Time Series with Subband Decomposition

Instead of reducing to power scalars, we decompose the signal into frequency bands
keeping the full time series. Each band becomes additional channels:
- Delta (0.5-4 Hz): 21 channels
- Theta (4-8 Hz): 21 channels
- Alpha (8-13 Hz): 21 channels
- Beta (13-30 Hz): 21 channels
- Gamma (30-50 Hz): 21 channels
Total: 105 channels → RNN

Each DIRU compartment can specialize to one frequency band!
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt, resample
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                            f1_score, precision_score, accuracy_score)
import matplotlib.pyplot as plt
from pathlib import Path
import pyedflib
import gc
import pickle
from tqdm import tqdm

# ============================
# Global config
# ============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 20
BATCH_SIZE = 32
HIDDEN_SIZE = 60
NUM_COMPARTMENTS = 5  # One per frequency band!
WINDOW_SIZE = 3       # seconds
OVERLAP = 0.5
FIRST_N_TIMEPOINTS = 3000
MAX_FILES_PER_FOLD = 15
CHECKPOINT_DIR = "/content/drive/MyDrive/checkpoints"

#  DIRU SPECIFIC
DIRU_HIDDEN_SIZE = 20
DIRU_NUM_COMPARTMENTS = 5
DIRU_DROPOUT = 0.5

# Frequency bands for neonatal EEG
FREQUENCY_BANDS = {
    'delta': (0.5, 4),    # Slow-wave, burst suppression
    'theta': (4, 8),      # Neonatal baseline rhythm
    'alpha': (8, 13),     # Less prominent in neonates
    'beta':  (13, 30),    # Seizure markers, muscle artifact
    'gamma': (30, 50)     # High-frequency oscillations
}

print(f"Device: {DEVICE}")
print(f"SUBBAND DECOMPOSITION (Raw Time Series):")
print(f"  Bands: {list(FREQUENCY_BANDS.keys())}")
print(f"  Compartments: {NUM_COMPARTMENTS} (one per band)")
print(f"  Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, Hidden: {HIDDEN_SIZE}")
print(f"  Window: {WINDOW_SIZE}s, Overlap: {OVERLAP}")
print(f"  First {FIRST_N_TIMEPOINTS} timepoints, Max files: {MAX_FILES_PER_FOLD}")

Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)


# ============================
# Subband Decomposition
# ============================

def decompose_into_subbands(data, fs=256, bands=FREQUENCY_BANDS):
    """
    Filter signal into each frequency band keeping full time series.

    Input:  (channels, samples) e.g. (21, 768)
    Output: (channels × bands, samples) e.g. (105, 768)

    Each DIRU compartment can specialize to one frequency band:
        Channels   0- 20: delta-filtered EEG (0.5-4 Hz)
        Channels  21- 41: theta-filtered EEG (4-8 Hz)
        Channels  42- 62: alpha-filtered EEG (8-13 Hz)
        Channels  63- 83: beta-filtered EEG  (13-30 Hz)
        Channels  84-104: gamma-filtered EEG (30-50 Hz)
    """
    band_signals = []

    for band_name, (low, high) in bands.items():
        b, a = butter(4, [low, high], fs=fs, btype='band')
        band_data = filtfilt(b, a, data, axis=1)  # (channels, samples)
        band_signals.append(band_data)

    # Stack all bands along channel dimension
    decomposed = np.vstack(band_signals)  # (channels × bands, samples)

    return decomposed


# ============================
# Model Definitions
# ============================

class DIRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_compartments=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_comp = num_compartments

        # input_size should be divisible by num_compartments
        assert input_size % num_compartments == 0
        self.band_size = input_size // num_compartments  # 105 // 5 = 21

        # Each compartment has its own W_in seeing only its band's channels
        self.W_in = nn.ModuleList([
            nn.Linear(self.band_size, hidden_size) for _ in range(num_compartments)
        ])
        # W_rec still sees full hidden state
        self.W_rec = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_compartments)
        ])
        # Gate still sees all compartments combined
        self.gate = nn.Linear(hidden_size * num_compartments, hidden_size)

    def forward(self, x, h):
        # x shape: (batch, input_size) e.g. (32, 105)
        # Split x into band-specific chunks
        # x_bands[0] = delta channels (0-20)
        # x_bands[1] = theta channels (21-41) etc.
        x_bands = torch.chunk(x, self.num_comp, dim=1)  # 5 × (batch, 21)

        # Each compartment processes its own band
        comp_outputs = []
        for i in range(self.num_comp):
            local_out = torch.tanh(self.W_in[i](x_bands[i]) + self.W_rec[i](h))
            comp_outputs.append(local_out)

        # Stack for gating
        comp = torch.stack(comp_outputs, dim=1)  # (batch, num_comp, hidden_size)
        comp_flat = comp.view(comp.size(0), -1)  # (batch, num_comp * hidden_size)

        # Active gate - KEY difference from Tractable
        g = torch.sigmoid(self.gate(comp_flat))  # (batch, hidden_size)

        # Gated summation - KEY difference from Tractable
        h_new = torch.sum(comp, dim=1) * g  # (batch, hidden_size)

        return h_new


class DIRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_compartments=5, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = DIRUCell(input_size, hidden_size, num_compartments)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t], h)
        h = self.dropout(h)
        return self.fc(h)


class TractableDendriticCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_compartments=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_comp = num_compartments
        self.comp_size = hidden_size // num_compartments
        assert hidden_size % num_compartments == 0

        self.W_in = nn.ModuleList([
            nn.Linear(input_size, self.comp_size) for _ in range(num_compartments)
        ])
        self.W_rec = nn.ModuleList([
            nn.Linear(hidden_size, self.comp_size) for _ in range(num_compartments)
        ])
        self.integration = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        comp_outputs = []
        for i in range(self.num_comp):
            local_out = torch.tanh(self.W_in[i](x) + self.W_rec[i](h))
            comp_outputs.append(local_out)
        combined = torch.cat(comp_outputs, dim=1)
        h_new = torch.tanh(self.integration(combined))
        return h_new


class TractableDendriticRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_compartments=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = TractableDendriticCell(input_size, hidden_size, num_compartments)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t], h)
        h = self.dropout(h)
        return self.fc(h)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1]
        h = self.dropout(h)
        return self.fc(h)


# ============================
# Memory Efficient Dataset with Subband Decomposition
# ============================

class HelsinkiSubbandDataset(Dataset):
    """
    Memory-efficient dataset with subband decomposition.
    Stores only metadata, loads and decomposes data on-demand.
    Input to RNN: (samples, channels×bands) e.g. (768, 105)
    """

    def __init__(self, data_path, annotations_path, window_size=3, overlap=0.5,
                 fold='train', first_n_timepoints=3000, max_files=15):
        self.data_path = Path(data_path)
        self.annotations_path = Path(annotations_path)
        self.window_size = window_size
        self.overlap = overlap
        self.fold = fold
        self.first_n = first_n_timepoints
        self.fs = 256
        self.max_files = max_files

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations not found: {annotations_path}")

        print(f"\nLoading annotations from {annotations_path}...")
        self.annotations_df = pd.read_csv(annotations_path, header=None)
        print(f"✓ Annotations shape: {self.annotations_df.shape}")

        self.window_index = self._build_window_index()

        print(f"\n{fold.upper()} SET:")
        print(f"Total windows: {len(self.window_index)}")
        if len(self.window_index) > 0:
            seizure_count = sum(1 for w in self.window_index if w['label'] == 1)
            print(f"Seizure windows: {seizure_count}")
            print(f"Non-seizure windows: {len(self.window_index) - seizure_count}")
            print(f"Class ratio: {seizure_count / len(self.window_index) * 100:.1f}% seizure")

    def _build_window_index(self):
        window_index = []

        edf_files = sorted(list(self.data_path.glob("eeg*.edf")),
                          key=lambda x: int(''.join(filter(str.isdigit, x.stem))))

        print(f"\nFound {len(edf_files)} EDF files")

        n_total = min(len(edf_files), 79)  # 79 files
        n_train = int(0.70 * n_total)      # 55 files for train
        n_val = int(0.30 * n_total)        # 12 files for val
        # test = remaining files           # 12 files for test (implicitly)

        if self.fold == 'train':
            edf_files = edf_files[:n_train]               # 0-54
        elif self.fold == 'val':
            edf_files = edf_files[n_train:n_train+n_val]  # 55-66
        else:  # test
            edf_files = edf_files[n_train+n_val:n_total]  # 67-78

            print(f"Using {len(edf_files)} files for {self.fold} set")

        window_samples = int(self.window_size * self.fs)
        step_samples = int(window_samples * (1 - self.overlap))

        files_processed = 0
        files_skipped = 0

        for idx, edf_file in enumerate(edf_files):
            file_num = int(''.join(filter(str.isdigit, edf_file.stem)))
            column_idx = file_num - 1

            if column_idx >= self.annotations_df.shape[1]:
                files_skipped += 1
                continue

            try:
                test_edf = pyedflib.EdfReader(str(edf_file))
                test_edf.close()
            except Exception as e:
                print(f"  Skipping corrupted file: {edf_file.name} ({e})")
                files_skipped += 1
                continue

            annotations = self.annotations_df[column_idx].values[:self.first_n]

            for start_idx in range(0, len(annotations) - window_samples, step_samples):
                end_idx = start_idx + window_samples
                window_annotations = annotations[start_idx:end_idx]
                seizure_ratio = window_annotations.sum() / len(window_annotations)
                label = 1 if seizure_ratio > 0.3 else 0

                window_index.append({
                    'file_path': str(edf_file),
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'label': label,
                    'seizure_ratio': seizure_ratio
                })

            files_processed += 1

        print(f"Processed: {files_processed} files, Skipped: {files_skipped} corrupted files")

        # Balance training set
        if self.fold == 'train':
            seizure_windows = [w for w in window_index if w['label'] == 1]
            non_seizure_windows = [w for w in window_index if w['label'] == 0]

            print(f"Train set - Before balancing:")
            print(f"  Seizure: {len(seizure_windows)}, Non-seizure: {len(non_seizure_windows)}")

            if len(seizure_windows) > 0:
                max_non_seizure = min(len(seizure_windows) * 4, len(non_seizure_windows))
                non_seizure_windows = np.random.choice(
                    non_seizure_windows, size=max_non_seizure, replace=False
                ).tolist()

            window_index = seizure_windows + non_seizure_windows
            np.random.shuffle(window_index)

            print(f"Train set - After balancing:")
            print(f"  Seizure: {len(seizure_windows)}, Non-seizure: {len(non_seizure_windows)}")
            print(f"  Total: {len(window_index)}")
        else:
            seizure_count = sum(1 for w in window_index if w['label'] == 1)
            print(f"{self.fold.upper()} set - Keeping all windows:")
            print(f"  Seizure: {seizure_count}, Non-seizure: {len(window_index) - seizure_count}")
            print(f"  Total: {len(window_index)}")

        return window_index

    def _load_and_preprocess_file(self, file_path):
        """Load, preprocess and decompose into subbands."""
        if not hasattr(self, '_cache'):
            self._cache = {}

        if file_path in self._cache:
            return self._cache[file_path]

        # Load EDF
        try:
            edf = pyedflib.EdfReader(file_path)
            n_channels = edf.signals_in_file
            sample_freq = edf.getSampleFrequency(0)

            data = []
            for i in range(n_channels):
                signal = edf.readSignal(i)
                data.append(signal)
            data = np.array(data)
            edf.close()
        except Exception as e:
            print(f"  ERROR loading {Path(file_path).name}: {e}")
            return None

        # Preprocess
        try:
            if sample_freq != self.fs:
                n_samples_new = int(data.shape[1] * self.fs / sample_freq)
                data = resample(data, n_samples_new, axis=1)

            # Bandpass filter
            b, a = butter(4, [0.5, 50], fs=self.fs, btype='band')
            data = filtfilt(b, a, data, axis=1)

            # Notch filter
            b_notch, a_notch = butter(4, [49, 51], fs=self.fs, btype='bandstop')
            data = filtfilt(b_notch, a_notch, data, axis=1)

            # Common average reference
            data = data - data.mean(axis=0, keepdims=True)

            # Normalize each channel
            data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)

            # Truncate
            data = data[:, :self.first_n]

            # Decompose into subbands - KEY STEP
            # (21 channels, samples) → (105 channels, samples)
            data = decompose_into_subbands(data, fs=self.fs)

        except Exception as e:
            print(f"  ERROR preprocessing {Path(file_path).name}: {e}")
            return None

        # Cache (limit to 3 files)
        if len(self._cache) > 3:
            self._cache.pop(next(iter(self._cache)))

        self._cache[file_path] = data
        return data

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        """Load window on-demand with subband decomposition."""
        window_info = self.window_index[idx]

        data = self._load_and_preprocess_file(window_info['file_path'])

        # data is now (105 channels, samples)
        n_channels = len(FREQUENCY_BANDS) * 21  # fallback for dummy

        if data is None:
            dummy_window = np.zeros((int(self.window_size * self.fs), n_channels))
            return torch.FloatTensor(dummy_window), torch.FloatTensor([window_info['label']])

        start = window_info['start_idx']
        end = window_info['end_idx']

        if end > data.shape[1]:
            end = data.shape[1]
            start = max(0, end - int(self.window_size * self.fs))

        window = data[:, start:end].T  # (samples, 105 channels)

        # Pad if needed
        expected_length = int(self.window_size * self.fs)
        if window.shape[0] < expected_length:
            padding = np.zeros((expected_length - window.shape[0], window.shape[1]))
            window = np.vstack([window, padding])

        label = window_info['label']

        return torch.FloatTensor(window), torch.FloatTensor([label])


# ============================
# Training
# ============================

def train_model(model, train_loader, val_loader, criterion, num_epochs=20, device='cpu',
                model_name='model', checkpoint_dir='/content/drive/MyDrive/checkpoints',
                skip_training=False):
    """Training with checkpoint loading, progress bars, and final model saving."""

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    model.to(device)

    best_model_path = Path(checkpoint_dir) / f"{model_name}_best.pkl"
    final_model_path = Path(checkpoint_dir) / f"{model_name}_final.pkl"

    # If final model exists, load and skip training
    if final_model_path.exists():
        print(f"  ⚡ Found final model: {final_model_path.name}")
        try:
            with open(final_model_path, 'rb') as f:
                final_checkpoint = pickle.load(f)

            if isinstance(final_checkpoint, dict) and 'model_state' in final_checkpoint:
                model.load_state_dict(final_checkpoint['model_state'])
                epoch = final_checkpoint.get('epoch', '?')
                val_loss = final_checkpoint.get('val_loss', float('inf'))
                print(f"  ✓ Loaded final model from epoch {epoch+1}, val_loss: {val_loss:.4f}")
            else:
                model.load_state_dict(final_checkpoint)
                print(f"  ✓ Loaded final model (legacy format)")

            print(f"  ⏭ Skipping training!")
            return {'train_loss': [], 'val_loss': []}
        except Exception as e:
            print(f"  ⚠ Error loading final model: {e}")
            print(f"  Falling through to normal training...")

    # Skip training mode - just load best and return
    if skip_training:
        if best_model_path.exists():
            print(f"  ⚡ Skipping training - loading best model")
            try:
                with open(best_model_path, 'rb') as f:
                    best_checkpoint = pickle.load(f)
                model.load_state_dict(best_checkpoint['model_state'])
                print(f"  ✓ Loaded from epoch {best_checkpoint['epoch']+1}, val_loss: {best_checkpoint['val_loss']:.4f}")
                return {'train_loss': [], 'val_loss': []}
            except Exception as e:
                print(f"  ⚠ Error loading: {e}, training from scratch")
        else:
            print(f"  ⚠ Best model not found, training from scratch")

    # Check if we should resume from best checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if best_model_path.exists():
        print(f"  Found existing checkpoint: {best_model_path.name}")
        try:
            with open(best_model_path, 'rb') as f:
                checkpoint = pickle.load(f)

            if isinstance(checkpoint, dict):
                if 'model_state' in checkpoint:
                    model.load_state_dict(checkpoint['model_state'])
                    start_epoch = checkpoint.get('epoch', 0) + 1
                    best_val_loss = checkpoint.get('val_loss', float('inf'))
                    print(f"  ✓ Resuming from epoch {start_epoch}, best val_loss: {best_val_loss:.4f}")
                elif any(k.startswith(('cell.', 'fc.', 'lstm.', 'dropout.')) for k in checkpoint.keys()):
                    model.load_state_dict(checkpoint)
                    print(f"  ✓ Loaded old format checkpoint")
                else:
                    print(f"  ⚠ Unknown checkpoint format, starting fresh")
            else:
                model.load_state_dict(checkpoint)
                print(f"  ✓ Loaded legacy format checkpoint")
        except Exception as e:
            print(f"  ⚠ Error loading checkpoint: {e}")
            print(f"  Starting from scratch")
            start_epoch = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    history = {'train_loss': [], 'val_loss': []}
    patience = 7
    patience_counter = 0
    last_epoch = start_epoch
    last_val_loss = float('inf')

    print(f"  Starting training...")
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"    Training", leave=False, ncols=100)

        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0

        pbar = tqdm(val_loader, desc=f"    Validation", leave=False, ncols=100)

        with torch.no_grad():
            for batch_x, batch_y in pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        val_loss /= len(val_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"  Epoch {epoch+1}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        last_epoch = epoch
        last_val_loss = val_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            with open(best_model_path, 'wb') as f:
                pickle.dump({
                    'model_state': model.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, f)
            print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  🛑 Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()

    # Load best model
    if best_model_path.exists():
        with open(best_model_path, 'rb') as f:
            best_checkpoint = pickle.load(f)
        model.load_state_dict(best_checkpoint['model_state'])
        print(f"  ✓ Loaded best model from epoch {best_checkpoint['epoch']+1}")

    # Save final model (best weights, but marked as final/complete)
    with open(final_model_path, 'wb') as f:
        pickle.dump({
            'model_state': model.state_dict(),
            'epoch': last_epoch,
            'val_loss': last_val_loss
        }, f)
    print(f"  ✓ Final model saved: {final_model_path.name}")

    return history


# ============================
# Evaluation
# ============================

def evaluate_model(model, val_loader, device):
    """Evaluate model and return metrics including probs and labels for ROC curves."""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            labels = batch_y.numpy().flatten()

            all_probs.extend(probs)
            all_labels.extend(labels)

    print(f"  Total samples: {len(all_labels)}")
    print(f"  Positive samples: {sum(all_labels)} ({sum(all_labels)/len(all_labels)*100:.1f}%)")

    if len(set(all_labels)) < 2:
        print(f"  ⚠ WARNING: Only one class present!")
        return {
            'auc': 0.0, 'accuracy': 0.0,
            'sensitivity': 0.0, 'specificity': 0.0, 'f1': 0.0,
            'probs': np.array(all_probs), 'labels': np.array(all_labels)
        }

    preds = (np.array(all_probs) > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()

    print(f"  Confusion: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    return {
        'accuracy': accuracy_score(all_labels, preds),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'auc': roc_auc_score(all_labels, all_probs),
        'f1': f1_score(all_labels, preds),
        'probs': np.array(all_probs),
        'labels': np.array(all_labels)
    }


def plot_roc_curves(results, save_path=None):
    """Plot ROC curves for all three models."""

    plt.figure(figsize=(10, 8))

    for model_name, color in [('diru', 'blue'), ('tractable', 'green'), ('lstm', 'red')]:
        metrics = results[model_name]
        fpr, tpr, _ = roc_curve(metrics['labels'], metrics['probs'])
        auc = metrics['auc']
        plt.plot(fpr, tpr, color=color, lw=2,
                label=f'{model_name.upper()} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Neonatal Seizure Detection', fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ ROC curve saved: {save_path}")

    plt.show()


# ============================
# Main
# ============================

def run_subband_comparison(data_path, num_epochs=20, batch_size=32, clean_checkpoints=False):
    """Run 3-way comparison with subband decomposition."""

    if clean_checkpoints:
        print("\n🧹 Cleaning old checkpoints...")
        checkpoint_files = [
            'diru_best.pkl', 'tractable_best.pkl', 'lstm_best.pkl',
            'diru_final.pkl', 'tractable_final.pkl', 'lstm_final.pkl',
        ]
        for ckpt_name in checkpoint_files:
            ckpt_path = Path(CHECKPOINT_DIR) / ckpt_name
            if ckpt_path.exists():
                ckpt_path.unlink()
                print(f"  Deleted: {ckpt_name}")
        print("✓ Cleanup complete\n")

    annotations_path = Path(data_path) / "annotations_2017_A_fixed.csv"

    print("\n" + "="*80)
    print("HELSINKI NEONATAL EEG - SUBBAND DECOMPOSITION")
    print("Raw Time Series decomposed into 5 frequency bands")
    print("21 channels × 5 bands = 105 input channels → RNN")
    print("Each DIRU compartment specializes to one frequency band!")
    print("="*80)

    train_dataset = HelsinkiSubbandDataset(
        data_path, annotations_path,
        window_size=WINDOW_SIZE, overlap=OVERLAP,
        fold='train', first_n_timepoints=FIRST_N_TIMEPOINTS,
        max_files=MAX_FILES_PER_FOLD
    )

    val_dataset = HelsinkiSubbandDataset(
        data_path, annotations_path,
        window_size=WINDOW_SIZE, overlap=OVERLAP,
        fold='val', first_n_timepoints=FIRST_N_TIMEPOINTS,
        max_files=MAX_FILES_PER_FOLD
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("ERROR: No data loaded!")
        return None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    sample_x, _ = next(iter(train_loader))
    num_channels = sample_x.shape[2]
    print(f"\nInput shape: {sample_x.shape}")
    print(f"Channels: {num_channels} ({21} electrodes × {len(FREQUENCY_BANDS)} bands)")

    seizure_count = sum(1 for w in train_dataset.window_index if w['label'] == 1)
    pos_ratio = seizure_count / len(train_dataset)
    pos_weight = torch.FloatTensor([(1 - pos_ratio) / pos_ratio]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Seizure ratio: {pos_ratio*100:.1f}%, pos_weight: {pos_weight.item():.2f}")

    print(f"\nInitializing models (input={num_channels}, hidden={HIDDEN_SIZE})...")
    diru = DIRU(num_channels, DIRU_HIDDEN_SIZE, 1, DIRU_NUM_COMPARTMENTS, dropout=DIRU_DROPOUT).to(DEVICE)
    tractable = TractableDendriticRNN(num_channels, HIDDEN_SIZE, 1, NUM_COMPARTMENTS).to(DEVICE)
    lstm = LSTMModel(num_channels, HIDDEN_SIZE, 1).to(DEVICE)

    print(f"  DIRU:      {sum(p.numel() for p in diru.parameters()):,} params")
    print(f"  Tractable: {sum(p.numel() for p in tractable.parameters()):,} params")
    print(f"  LSTM:      {sum(p.numel() for p in lstm.parameters()):,} params")

    print("\n" + "-"*80)
    print("Training DIRU...")
    print("-"*80)
    diru_hist = train_model(diru, train_loader, val_loader, criterion, num_epochs, DEVICE,
                           model_name='diru', checkpoint_dir=CHECKPOINT_DIR)

    print("\n" + "-"*80)
    print("Training Tractable...")
    print("-"*80)
    tractable_hist = train_model(tractable, train_loader, val_loader, criterion, num_epochs, DEVICE,
                                 model_name='tractable', checkpoint_dir=CHECKPOINT_DIR)

    print("\n" + "-"*80)
    print("Training LSTM...")
    print("-"*80)
    lstm_hist = train_model(lstm, train_loader, val_loader, criterion, num_epochs, DEVICE,
                           model_name='lstm', checkpoint_dir=CHECKPOINT_DIR)

    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    print("\nEvaluating DIRU:")
    diru_metrics = evaluate_model(diru, val_loader, DEVICE)

    print("\nEvaluating Tractable:")
    tractable_metrics = evaluate_model(tractable, val_loader, DEVICE)

    print("\nEvaluating LSTM:")
    lstm_metrics = evaluate_model(lstm, val_loader, DEVICE)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"DIRU:      Sens: {diru_metrics['sensitivity']:.2%} | AUC: {diru_metrics['auc']:.3f}")
    print(f"Tractable: Sens: {tractable_metrics['sensitivity']:.2%} | AUC: {tractable_metrics['auc']:.3f}")
    print(f"LSTM:      Sens: {lstm_metrics['sensitivity']:.2%} | AUC: {lstm_metrics['auc']:.3f}")

    improvement = (diru_metrics['sensitivity'] - lstm_metrics['sensitivity']) * 100
    print(f"\nDIRU vs LSTM: {improvement:+.1f} percentage points")

    print("\n🧠 Interpretation:")
    print("Each DIRU compartment specializes to one frequency band:")
    print("  Comp 1 → Delta channels  (0.5-4 Hz):  Slow-wave seizures")
    print("  Comp 2 → Theta channels  (4-8 Hz):    Baseline disruption")
    print("  Comp 3 → Alpha channels  (8-13 Hz):   Arousal changes")
    print("  Comp 4 → Beta channels   (13-30 Hz):  Ictal discharges")
    print("  Comp 5 → Gamma channels  (30-50 Hz):  High-frequency markers")

    results = {
        'diru': diru_metrics,
        'tractable': tractable_metrics,
        'lstm': lstm_metrics,
        'improvement': improvement
    }

    results_path = Path(CHECKPOINT_DIR) / 'subband_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Results saved: {results_path}")

    print("\n" + "="*80)
    print("ROC CURVES")
    print("="*80)
    roc_path = Path(CHECKPOINT_DIR) / 'roc_curves.png'
    plot_roc_curves(results, save_path=roc_path)

    return results


if __name__ == "__main__":
    DATA_PATH = "/content/drive/MyDrive/eeg_cache"

    # Set to True to delete old checkpoints and start fresh
    CLEAN_CHECKPOINTS = False

    results = run_subband_comparison(
        DATA_PATH,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        clean_checkpoints=CLEAN_CHECKPOINTS
    )

    if results:
        print("\n" + "="*80)
        print("COMPLETED!")
        print("="*80)
        
        
        
        
        
        
        
import pickle
from pathlib import Path

CHECKPOINT_DIR= '/content/drive/MyDrive/checkpoints'

# Load saved results
results_path = Path(CHECKPOINT_DIR) / 'subband_results.pkl'
with open(results_path, 'rb') as f:
    results = pickle.load(f)

# Print full report for each model
for model_name in ['diru', 'tractable', 'lstm']:
    metrics = results[model_name]
    print(f"\n{model_name.upper()} Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")  # same as recall
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    print(f"  AUC:         {metrics['auc']:.4f}")
