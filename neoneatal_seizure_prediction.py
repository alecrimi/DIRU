"""
Helsinki Neonatal EEG - MEMORY EFFICIENT VERSION
Loads data on-the-fly instead of storing all in RAM

Key changes:
1. Store only file paths and window indices, not actual data
2. Load windows on-demand in __getitem__
3. Process fewer files or smaller windows if needed
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt, resample
from sklearn.metrics import (confusion_matrix, roc_auc_score, 
                            f1_score, precision_score, accuracy_score)
import matplotlib.pyplot as plt
from pathlib import Path
import pyedflib
import gc
import pickle

# ============================
# Global config - FAST VERSION WITH CHECKPOINTING
# ============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 20
BATCH_SIZE = 32
HIDDEN_SIZE = 48
NUM_COMPARTMENTS = 4
WINDOW_SIZE = 3  # seconds
OVERLAP = 0.75
FIRST_N_TIMEPOINTS = 5000
MAX_FILES_PER_FOLD = 20
CHECKPOINT_EVERY = 5
CHECKPOINT_DIR = "/content/drive/MyDrive/checkpoints"

# Task configuration
TASK = "prediction"  # "detection" or "prediction"
PREDICTION_HORIZON = 60  # For prediction: seconds before seizure to predict (1-5 minutes)

print(f"Device: {DEVICE}")
print(f"TASK: {TASK.upper()}")
if TASK == "prediction":
    print(f"  Prediction horizon: {PREDICTION_HORIZON}s ({PREDICTION_HORIZON/60:.1f} minutes before seizure)")
print(f"Configuration:")
print(f"  Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}")
print(f"  Hidden: {HIDDEN_SIZE}, Compartments: {NUM_COMPARTMENTS}")
print(f"  Window: {WINDOW_SIZE}s, Overlap: {OVERLAP}, First {FIRST_N_TIMEPOINTS} points")
print(f"  Max files: {MAX_FILES_PER_FOLD}")
print(f"  Checkpoints: every {CHECKPOINT_EVERY} epochs → {CHECKPOINT_DIR}")

Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)


# ============================
# Model Definitions (same as before)
# ============================

class DIRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_compartments=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_comp = num_compartments
        self.W_in = nn.Linear(input_size, hidden_size * num_compartments)
        self.W_rec = nn.Linear(hidden_size, hidden_size * num_compartments)
        self.gate = nn.Linear(hidden_size * num_compartments, hidden_size)

    def forward(self, x, h):
        comp = torch.tanh(self.W_in(x) + self.W_rec(h))
        g = torch.sigmoid(self.gate(comp))
        comp = comp.view(comp.size(0), self.num_comp, self.hidden_size)
        h_new = torch.sum(comp, dim=1) * g
        return h_new


class DIRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_compartments=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = DIRUCell(input_size, hidden_size, num_compartments)
        self.dropout = nn.Dropout(0.3)  # Add dropout for regularization
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t], h)
        h = self.dropout(h)  # Apply dropout
        return self.fc(h)


class TractableDendriticCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_compartments=4):
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
    def __init__(self, input_size, hidden_size, output_size, num_compartments=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = TractableDendriticCell(input_size, hidden_size, num_compartments)
        self.dropout = nn.Dropout(0.3)  # Add dropout
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t], h)
        h = self.dropout(h)  # Apply dropout
        return self.fc(h)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                           batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        self.dropout = nn.Dropout(0.3)  # Add dropout after LSTM
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1]
        h = self.dropout(h)  # Apply dropout
        return self.fc(h)


# ============================
# MEMORY EFFICIENT Dataset
# ============================

class HelsinkiNeonatalDatasetMemoryEfficient(Dataset):
    """
    Memory-efficient version: stores only metadata, loads data on-demand.
    """
    
    def __init__(self, data_path, annotations_path, window_size=3, overlap=0.75, 
                 fold='train', first_n_timepoints=5000, max_files=20, 
                 task='detection', prediction_horizon=60):
        """
        Args:
            task: 'detection' or 'prediction'
            prediction_horizon: For prediction task, seconds before seizure to predict
        """
        self.data_path = Path(data_path)
        self.annotations_path = Path(annotations_path)
        self.window_size = window_size
        self.overlap = overlap
        self.fold = fold
        self.first_n = first_n_timepoints
        self.fs = 256
        self.max_files = max_files
        self.task = task
        self.prediction_horizon = prediction_horizon  # in seconds
        
        # Check paths
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations not found: {annotations_path}")
        
        # Load annotations (NO HEADERS)
        print(f"\nLoading annotations from {annotations_path}...")
        self.annotations_df = pd.read_csv(annotations_path, header=None)
        print(f"✓ Annotations shape: {self.annotations_df.shape}")
        
        # Build index of windows (but don't load data yet!)
        self.window_index = self._build_window_index()
        
        print(f"\n{fold.upper()} SET:")
        print(f"Total windows: {len(self.window_index)}")
        if len(self.window_index) > 0:
            seizure_count = sum(1 for w in self.window_index if w['label'] == 1)
            print(f"Seizure windows: {seizure_count}")
            print(f"Non-seizure windows: {len(self.window_index) - seizure_count}")
            print(f"Class ratio: {seizure_count / len(self.window_index) * 100:.1f}% seizure")
    
    def _build_window_index(self):
        """Build index of windows without loading data."""
        window_index = []
        
        # Get EDF files
        edf_files = sorted(list(self.data_path.glob("eeg*.edf")), 
                          key=lambda x: int(''.join(filter(str.isdigit, x.stem))))
        
        print(f"\nFound {len(edf_files)} EDF files")
        
        # Split into train/val/test
        n_total = min(len(edf_files), self.max_files * 3)  # Limit total files
        n_train = int(0.6 * n_total)
        n_val = int(0.2 * n_total)
        
        if self.fold == 'train':
            edf_files = edf_files[:n_train]
        elif self.fold == 'val':
            edf_files = edf_files[n_train:n_train+n_val]
        else:
            edf_files = edf_files[n_train+n_val:n_total]
        
        edf_files = edf_files[:self.max_files]  # Limit files per fold
        
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
            
            # Quick validation: try to open the file
            try:
                test_edf = pyedflib.EdfReader(str(edf_file))
                test_edf.close()
            except Exception as e:
                print(f"  Skipping corrupted file: {edf_file.name} ({e})")
                files_skipped += 1
                continue
            
            # Get annotations
            annotations = self.annotations_df[column_idx].values[:self.first_n]
            
            # Create window indices
            for start_idx in range(0, len(annotations) - window_samples, step_samples):
                end_idx = start_idx + window_samples
                window_annotations = annotations[start_idx:end_idx]
                
                if self.task == 'detection':
                    # DETECTION: Label = 1 if window contains seizure
                    seizure_ratio = window_annotations.sum() / len(window_annotations)
                    label = 1 if seizure_ratio > 0.3 else 0
                    
                elif self.task == 'prediction':
                    # PREDICTION: Label = 1 if seizure occurs within horizon AFTER window
                    horizon_samples = int(self.prediction_horizon * self.fs)
                    
                    # Check if window itself has seizure (exclude these)
                    if window_annotations.sum() > 0:
                        continue  # Skip windows during seizure
                    
                    # Check if seizure starts within prediction horizon
                    horizon_start = end_idx
                    horizon_end = min(end_idx + horizon_samples, len(annotations))
                    
                    if horizon_end - horizon_start < horizon_samples * 0.5:
                        continue  # Not enough data for horizon
                    
                    horizon_annotations = annotations[horizon_start:horizon_end]
                    
                    # Label as pre-ictal if seizure starts in horizon
                    # Find seizure onset (transition from 0 to 1)
                    label = 0
                    for i in range(len(horizon_annotations) - 1):
                        if horizon_annotations[i] == 0 and horizon_annotations[i+1] == 1:
                            label = 1
                            break
                    
                    seizure_ratio = horizon_annotations.sum() / len(horizon_annotations)
                
                # Store metadata only (not actual data!)
                window_index.append({
                    'file_path': str(edf_file),
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'label': label,
                    'seizure_ratio': seizure_ratio if self.task == 'detection' else 0
                })
            
            files_processed += 1
        
        print(f"Processed: {files_processed} files, Skipped: {files_skipped} corrupted files")
        print(f"Raw windows before balancing: {len(window_index)}")
        
        # Balance training set ONLY
        if self.fold == 'train':
            seizure_windows = [w for w in window_index if w['label'] == 1]
            non_seizure_windows = [w for w in window_index if w['label'] == 0]
            
            print(f"Train set - Before balancing:")
            print(f"  Seizure: {len(seizure_windows)}, Non-seizure: {len(non_seizure_windows)}")
            
            if len(seizure_windows) > 0:
                # Keep 4:1 ratio for training
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
            # Validation/Test: Keep ALL windows for proper evaluation
            seizure_count = sum(1 for w in window_index if w['label'] == 1)
            print(f"{self.fold.upper()} set - Keeping all windows:")
            print(f"  Seizure: {seizure_count}, Non-seizure: {len(window_index) - seizure_count}")
            print(f"  Total: {len(window_index)}")
        
        return window_index
    
    def _load_and_preprocess_file(self, file_path):
        """Load and preprocess a single file (cached per file)."""
        # Check if already cached
        if not hasattr(self, '_cache'):
            self._cache = {}
        
        if file_path in self._cache:
            return self._cache[file_path]
        
        # Load EDF with error handling
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
            print(f"  Skipping this file...")
            # Return None to signal error
            return None
        
        # Preprocess
        try:
            if sample_freq != self.fs:
                n_samples_new = int(data.shape[1] * self.fs / sample_freq)
                data = resample(data, n_samples_new, axis=1)
            
            b, a = butter(4, [0.5, 50], fs=self.fs, btype='band')
            data = filtfilt(b, a, data, axis=1)
            
            b_notch, a_notch = butter(4, [49, 51], fs=self.fs, btype='bandstop')
            data = filtfilt(b_notch, a_notch, data, axis=1)
            
            data = data - data.mean(axis=0, keepdims=True)
            data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
            
            data = data[:, :self.first_n]
        except Exception as e:
            print(f"  ERROR preprocessing {Path(file_path).name}: {e}")
            return None
        
        # Cache it (but limit cache size)
        if len(self._cache) > 3:  # Keep only 3 files in cache
            self._cache.pop(next(iter(self._cache)))
        
        self._cache[file_path] = data
        return data
    
    def __len__(self):
        return len(self.window_index)
    
    def __getitem__(self, idx):
        """Load window on-demand."""
        window_info = self.window_index[idx]
        
        # Load file (with caching and error handling)
        data = self._load_and_preprocess_file(window_info['file_path'])
        
        # If file is corrupted, return zeros (will be caught by DataLoader)
        if data is None:
            # Return dummy data - PyTorch DataLoader will skip if collate_fn handles it
            # Or we return zeros
            dummy_window = np.zeros((int(self.window_size * self.fs), 21))  # Assume 21 channels
            return torch.FloatTensor(dummy_window), torch.FloatTensor([window_info['label']])
        
        # Extract window
        start = window_info['start_idx']
        end = window_info['end_idx']
        
        # Check bounds
        if end > data.shape[1]:
            end = data.shape[1]
            start = max(0, end - int(self.window_size * self.fs))
        
        window = data[:, start:end].T  # (samples, channels)
        
        # Pad if too short
        expected_length = int(self.window_size * self.fs)
        if window.shape[0] < expected_length:
            padding = np.zeros((expected_length - window.shape[0], window.shape[1]))
            window = np.vstack([window, padding])
        
        label = window_info['label']
        
        return torch.FloatTensor(window), torch.FloatTensor([label])


# ============================
# Training (simplified)
# ============================

def train_model(model, train_loader, val_loader, criterion, num_epochs=20, device='cpu', 
                model_name='model', checkpoint_dir='/content/drive/MyDrive/checkpoints'):
    """Training with checkpointing and early stopping."""
    import pickle
    from pathlib import Path
    
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Added weight decay
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience = 7  # Early stopping patience
    patience_counter = 0
    start_epoch = 0  # Initialize start_epoch first
    epoch = 0  # Initialize epoch variable
    
    # Try to resume from checkpoint
    checkpoint_path = Path(checkpoint_dir) / f"{model_name}_checkpoint.pkl"
    best_model_path = Path(checkpoint_dir) / f"{model_name}_best.pkl"
    
    # Prefer best model if both exist and checkpoint suggests overfitting
    if checkpoint_path.exists() and best_model_path.exists():
        print(f"  Found both checkpoint and best model")
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            with open(best_model_path, 'rb') as f:
                best_checkpoint = pickle.load(f)
            
            # If checkpoint is overfitting (val_loss worse than best), use best model
            checkpoint_epoch = checkpoint['epoch']
            best_epoch = best_checkpoint['epoch']
            
            if checkpoint['best_val_loss'] < best_checkpoint['val_loss'] * 1.05:
                # Checkpoint is still good, resume from it
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                history = checkpoint['history']
                start_epoch = checkpoint_epoch + 1
                epoch = start_epoch
                best_val_loss = checkpoint['best_val_loss']
                patience_counter = checkpoint.get('patience_counter', 0)
                print(f"  Resumed from checkpoint at epoch {start_epoch}")
            else:
                # Checkpoint is overfitting, resume from best model but continue training
                model.load_state_dict(best_checkpoint['model_state'])
                # Don't load optimizer state - fresh start from best model
                history = checkpoint['history']  # Keep history
                start_epoch = best_epoch + 1
                epoch = start_epoch
                best_val_loss = best_checkpoint['val_loss']
                patience_counter = checkpoint_epoch - best_epoch  # Calculate patience from gap
                print(f"  ⚠ Checkpoint (epoch {checkpoint_epoch}) is overfitting")
                print(f"  Loaded best model from epoch {best_epoch+1}, continuing training")
                
        except Exception as e:
            print(f"  Error loading checkpoints: {e}")
            print(f"  Starting from scratch")
            start_epoch = 0
            epoch = 0
    elif checkpoint_path.exists():
        print(f"  Found checkpoint! Resuming from {checkpoint_path}")
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            history = checkpoint['history']
            start_epoch = checkpoint['epoch'] + 1
            epoch = start_epoch
            best_val_loss = checkpoint['best_val_loss']
            patience_counter = checkpoint.get('patience_counter', 0)
            print(f"  Resumed from epoch {start_epoch}")
        except Exception as e:
            print(f"  Error loading checkpoint: {e}")
            print(f"  Starting from scratch")
            start_epoch = 0
            epoch = 0
    
    for epoch in range(start_epoch, num_epochs):
        # Training with dropout enabled
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation with dropout disabled
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"  Epoch {epoch+1}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            best_model_path = Path(checkpoint_dir) / f"{model_name}_best.pkl"
            with open(best_model_path, 'wb') as f:
                pickle.dump({
                    'model_state': model.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, f)
            print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ⚠ No improvement for {patience_counter}/{patience} epochs")
            
            if patience_counter >= patience:
                print(f"\n  🛑 Early stopping triggered at epoch {epoch+1}")
                print(f"  Best validation loss was: {best_val_loss:.4f}")
                break
        
        # Save checkpoint every CHECKPOINT_EVERY epochs
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'history': history,
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter
            }
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        
        # Clear cache
        if (epoch + 1) % 10 == 0:
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()
    
    # Load best model before returning
    best_model_path = Path(checkpoint_dir) / f"{model_name}_best.pkl"
    if best_model_path.exists():
        with open(best_model_path, 'rb') as f:
            best_checkpoint = pickle.load(f)
        model.load_state_dict(best_checkpoint['model_state'])
        print(f"  ✓ Loaded best model from epoch {best_checkpoint['epoch']+1}")
    
    # Final checkpoint
    final_checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'history': history,
        'best_val_loss': best_val_loss
    }
    
    final_path = Path(checkpoint_dir) / f"{model_name}_final.pkl"
    with open(final_path, 'wb') as f:
        pickle.dump(final_checkpoint, f)
    
    print(f"  ✓ Final checkpoint saved: {final_path}")
    
    return history


def evaluate_model(model, val_loader, device):
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
    
    # Diagnostics
    print(f"  Total samples: {len(all_labels)}")
    print(f"  Positive samples: {sum(all_labels)} ({sum(all_labels)/len(all_labels)*100:.1f}%)")
    print(f"  Prediction stats:")
    print(f"    Min prob: {min(all_probs):.3f}, Max prob: {max(all_probs):.3f}")
    print(f"    Mean prob: {np.mean(all_probs):.3f}, Median prob: {np.median(all_probs):.3f}")
    
    if len(set(all_labels)) < 2:
        print(f"  ⚠ WARNING: Only one class present!")
        return {'auc': 0.0, 'accuracy': 0.0, 'sensitivity': 0.0, 'specificity': 0.0}
    
    preds = (np.array(all_probs) > 0.5).astype(int)
    print(f"    Predictions: {sum(preds)} positive, {len(preds)-sum(preds)} negative")
    
    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()
    
    print(f"  Confusion Matrix:")
    print(f"    True Positives: {tp}, False Positives: {fp}")
    print(f"    False Negatives: {fn}, True Negatives: {tn}")
    
    return {
        'accuracy': accuracy_score(all_labels, preds),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'auc': roc_auc_score(all_labels, all_probs),
        'f1': f1_score(all_labels, preds),
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


# ============================
# Main
# ============================

def run_helsinki_comparison(data_path, num_epochs=20, batch_size=32):
    annotations_path = Path(data_path) / "annotations_2017_A_fixed.csv"
    
    print("\n" + "="*80)
    if TASK == 'detection':
        print("HELSINKI NEONATAL EEG - SEIZURE DETECTION")
    else:
        print(f"HELSINKI NEONATAL EEG - SEIZURE PREDICTION ({PREDICTION_HORIZON/60:.1f} min ahead)")
    print("3-Way Comparison: DIRU vs Tractable Dendritic vs LSTM")
    print("="*80)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = HelsinkiNeonatalDatasetMemoryEfficient(
        data_path, annotations_path, 
        window_size=WINDOW_SIZE, overlap=OVERLAP, 
        fold='train', first_n_timepoints=FIRST_N_TIMEPOINTS,
        max_files=MAX_FILES_PER_FOLD,
        task=TASK, prediction_horizon=PREDICTION_HORIZON
    )
    
    val_dataset = HelsinkiNeonatalDatasetMemoryEfficient(
        data_path, annotations_path,
        window_size=WINDOW_SIZE, overlap=OVERLAP,
        fold='val', first_n_timepoints=FIRST_N_TIMEPOINTS,
        max_files=MAX_FILES_PER_FOLD,
        task=TASK, prediction_horizon=PREDICTION_HORIZON
    )
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("ERROR: No data loaded!")
        return None
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Get dimensions
    sample_x, _ = next(iter(train_loader))
    num_channels = sample_x.shape[2]
    print(f"\nChannels: {num_channels}")
    
    # Class weights
    seizure_count = sum(1 for w in train_dataset.window_index if w['label'] == 1)
    pos_ratio = seizure_count / len(train_dataset)
    pos_weight = torch.FloatTensor([(1 - pos_ratio) / pos_ratio]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Seizure ratio: {pos_ratio*100:.1f}%, pos_weight: {pos_weight.item():.2f}")
    
    # Models
    print(f"\nInitializing models...")
    diru = DIRU(num_channels, HIDDEN_SIZE, 1, NUM_COMPARTMENTS).to(DEVICE)
    tractable = TractableDendriticRNN(num_channels, HIDDEN_SIZE, 1, NUM_COMPARTMENTS).to(DEVICE)
    lstm = LSTMModel(num_channels, HIDDEN_SIZE, 1).to(DEVICE)
    
    # Train
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
    
    # Evaluate
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    diru_metrics = evaluate_model(diru, val_loader, DEVICE)
    tractable_metrics = evaluate_model(tractable, val_loader, DEVICE)
    lstm_metrics = evaluate_model(lstm, val_loader, DEVICE)
    
    print(f"\nDIRU:      Sens: {diru_metrics['sensitivity']:.2%} | AUC: {diru_metrics['auc']:.3f}")
    print(f"Tractable: Sens: {tractable_metrics['sensitivity']:.2%} | AUC: {tractable_metrics['auc']:.3f}")
    print(f"LSTM:      Sens: {lstm_metrics['sensitivity']:.2%} | AUC: {lstm_metrics['auc']:.3f}")
    
    improvement = (diru_metrics['sensitivity'] - lstm_metrics['sensitivity']) * 100
    print(f"\nDIRU vs LSTM: {improvement:+.1f} percentage points")
    
    # Save final results
    results = {
        'diru': diru_metrics, 
        'tractable': tractable_metrics, 
        'lstm': lstm_metrics,
        'diru_history': diru_hist,
        'tractable_history': tractable_hist,
        'lstm_history': lstm_hist,
        'improvement': improvement
    }
    
    results_path = Path(CHECKPOINT_DIR) / 'final_results.pkl'
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✓ Final results saved: {results_path}")
    
    return results


if __name__ == "__main__":
    DATA_PATH = "/content/drive/MyDrive/eeg_cache"
    
    results = run_helsinki_comparison(DATA_PATH, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    
    if results:
        print("\n" + "="*80)
        print("COMPLETED!")
        print("="*80)
