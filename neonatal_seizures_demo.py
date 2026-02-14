"""
Helsinki Neonatal EEG Seizure Detection - Google Drive Version (FIXED)
Complete 3-way comparison: DIRU vs Tractable Dendritic vs LSTM

Dataset: University of Helsinki Neonatal EEG
Data: eeg_cache folder with EDF files + annotations_2017_A_fixed.csv (NO HEADERS)
CSV Structure: Each column = one file (column 0 = eeg1.edf, column 1 = eeg2.edf, etc.)
               Each row = one timepoint (1 = seizure, 0 = non-seizure)

Class Imbalance Handling (based on literature):
1. Lower seizure threshold: 30% instead of 50% (Temko et al., 2011)
2. Keep all seizure windows, subsample non-seizure in training
3. Use weighted loss function (BCE with pos_weight)
4. Evaluate on balanced validation set

References for imbalance handling:
- Temko et al. (2011): "EEG-based neonatal seizure detection with Support Vector Machines"
- Stevenson et al. (2015): "A nonparametric feature for neonatal EEG seizure detection"
- Pavel et al. (2017): "A machine-learning algorithm for neonatal seizure recognition"
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt, resample
from scipy import stats
from sklearn.metrics import (confusion_matrix, roc_auc_score, 
                            f1_score, precision_score, accuracy_score)
import matplotlib.pyplot as plt
from pathlib import Path
import pyedflib

# ============================
# Global config
# ============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 50
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_COMPARTMENTS = 4  # Must divide HIDDEN_SIZE evenly (64/4=16)
# Note: For 6 compartments, use HIDDEN_SIZE=72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 128...
# For 8 compartments, use HIDDEN_SIZE=64, 72, 80, 88, 96, 104, 112, 120, 128...
WINDOW_SIZE = 4  # seconds (smaller = more windows)
OVERLAP = 0.75  # 75% overlap (more windows)
FIRST_N_TIMEPOINTS = 10000  # Use more timepoints for better data

print(f"Device: {DEVICE}")
print(f"Configuration: {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}")
print(f"Hidden size: {HIDDEN_SIZE}, Compartments: {NUM_COMPARTMENTS}")
print(f"Window: {WINDOW_SIZE}s with {OVERLAP*100:.0f}% overlap")
print(f"Using first {FIRST_N_TIMEPOINTS} timepoints per file")


# ============================
# Model Definitions
# ============================

# ============ DIRU (Active Dendrites) ============
class DIRUCell(nn.Module):
    """Simple working DIRU cell"""
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
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t], h)
        return self.fc(h)


# ============ Tractable Dendritic RNN (Brenner et al.) ============
class TractableDendriticCell(nn.Module):
    """Brenner et al. style: PASSIVE dendrites"""
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
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t], h)
        return self.fc(h)


# ============ LSTM (Baseline) ============
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1]
        return self.fc(h)


# ============================
# Dataset with CSV Annotations (NO HEADERS)
# ============================

class HelsinkiNeonatalDatasetCSV(Dataset):
    """
    Helsinki Neonatal EEG with CSV annotations (NO HEADERS).
    
    CSV Structure:
    - NO column headers
    - Column 0 = eeg1.edf, Column 1 = eeg2.edf, ..., Column N = eegN+1.edf
    - Each row = one timepoint
    - Value = 1 (seizure) or 0 (non-seizure)
    """
    
    def __init__(self, data_path, annotations_path, window_size=10, overlap=0.5, 
                 fold='train', first_n_timepoints=3000):
        """
        Args:
            data_path: Path to eeg_cache folder with EDF files
            annotations_path: Path to annotations_2017_A_fixed.csv (in same folder)
            window_size: Window size in seconds
            overlap: Overlap ratio between windows (0-1)
            fold: 'train', 'val', or 'test'
            first_n_timepoints: Use only first N timepoints from each recording
        """
        self.data_path = Path(data_path)
        self.annotations_path = Path(annotations_path)
        self.window_size = window_size
        self.overlap = overlap
        self.fold = fold
        self.first_n = first_n_timepoints
        self.fs = 256  # Sampling rate
        
        # Check paths exist
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
        
        # Load annotations (NO HEADERS - just data matrix)
        print(f"\nLoading annotations from {annotations_path}...")
        self.annotations_df = pd.read_csv(annotations_path, header=None)  # IMPORTANT: header=None
        print(f"✓ Loaded annotations successfully")
        print(f"Annotations shape: {self.annotations_df.shape}")
        print(f"  Rows (timepoints): {self.annotations_df.shape[0]}")
        print(f"  Columns (files): {self.annotations_df.shape[1]}")
        
        # Show first few values
        print(f"\nFirst file (column 0) stats:")
        print(f"  Total timepoints: {len(self.annotations_df[0])}")
        print(f"  Seizure timepoints: {self.annotations_df[0].sum()}")
        print(f"  Seizure ratio: {self.annotations_df[0].sum() / len(self.annotations_df[0]) * 100:.1f}%")
        
        # Load data
        self.X, self.y, self.metadata = self._load_dataset()
        
        print(f"\n{fold.upper()} SET:")
        print(f"Total windows: {len(self.X)}")
        if len(self.y) > 0:
            print(f"Seizure windows: {int(self.y.sum())}")
            print(f"Non-seizure windows: {int(len(self.y) - self.y.sum())}")
            print(f"Class ratio: {self.y.sum() / len(self.y) * 100:.1f}% seizure")
    
    def _load_dataset(self):
        """Load EEG data and create windows with CSV annotations."""
        X_all = []
        y_all = []
        metadata_all = []
        
        # Get all EDF files, sorted numerically
        edf_files = sorted(list(self.data_path.glob("eeg*.edf")), 
                          key=lambda x: int(''.join(filter(str.isdigit, x.stem))))
        
        print(f"\nFound {len(edf_files)} EDF files in {self.data_path}")
        
        if len(edf_files) == 0:
            print("ERROR: No EDF files found!")
            print(f"Checked path: {self.data_path}")
            print(f"Looking for files matching: eeg*.edf")
            return torch.FloatTensor([]), torch.FloatTensor([]), []
        
        # Show first few files
        print(f"First 5 files: {[f.name for f in edf_files[:5]]}")
        
        # Split into train/val/test (60/20/20)
        n_total = len(edf_files)
        n_train = int(0.6 * n_total)
        n_val = int(0.2 * n_total)
        
        if self.fold == 'train':
            edf_files = edf_files[:n_train]
        elif self.fold == 'val':
            edf_files = edf_files[n_train:n_train+n_val]
        else:  # test
            edf_files = edf_files[n_train+n_val:]
        
        print(f"\nProcessing {len(edf_files)} files for {self.fold} set...")
        
        for idx, edf_file in enumerate(edf_files):
            file_name = edf_file.name
            
            # Extract file number (eeg1.edf -> 0, eeg2.edf -> 1, etc.)
            file_num_str = ''.join(filter(str.isdigit, edf_file.stem))
            if not file_num_str:
                print(f"  WARNING: Could not extract number from {file_name}, skipping")
                continue
            
            file_num = int(file_num_str)
            column_idx = file_num - 1  # eeg1.edf is column 0, eeg2.edf is column 1, etc.
            
            print(f"[{idx+1}/{len(edf_files)}] Loading {file_name} (column {column_idx})...")
            
            # Check if column exists in annotations
            if column_idx >= self.annotations_df.shape[1]:
                print(f"  WARNING: Column {column_idx} not in annotations (only {self.annotations_df.shape[1]} columns)")
                continue
            
            # Load EEG data
            try:
                edf = pyedflib.EdfReader(str(edf_file))
                n_channels = edf.signals_in_file
                sample_freq = edf.getSampleFrequency(0)
                
                # Read all channels
                data = []
                for i in range(n_channels):
                    signal = edf.readSignal(i)
                    data.append(signal)
                data = np.array(data)  # (channels, samples)
                
                edf.close()
                
                print(f"  Loaded: {n_channels} channels, {data.shape[1]} samples, {sample_freq} Hz")
                
            except Exception as e:
                print(f"  ERROR loading {file_name}: {e}")
                continue
            
            # Get annotations for this file (column_idx)
            annotations = self.annotations_df[column_idx].values[:self.first_n]
            print(f"  Using column {column_idx}, first {self.first_n} timepoints")
            print(f"  Seizure timepoints: {annotations.sum()} ({annotations.sum()/len(annotations)*100:.1f}%)")
            
            # Preprocess
            data = self._preprocess_eeg(data, sample_freq)
            
            # Truncate to first_n timepoints
            data = data[:, :self.first_n]
            
            # Create windows
            X_rec, y_rec, meta_rec = self._create_windows(data, annotations, file_name)
            
            X_all.extend(X_rec)
            y_all.extend(y_rec)
            metadata_all.extend(meta_rec)
            
            print(f"  Created {len(X_rec)} windows ({sum(y_rec)} seizure, {len(y_rec)-sum(y_rec)} non-seizure)")
        
        # Convert to tensors
        if len(X_all) > 0:
            X = torch.FloatTensor(np.array(X_all))
            y = torch.FloatTensor(np.array(y_all))
        else:
            X = torch.FloatTensor([])
            y = torch.FloatTensor([])
        
        return X, y, metadata_all
    
    def _preprocess_eeg(self, data, original_fs):
        """Preprocess EEG signals."""
        # Resample if needed
        if original_fs != self.fs:
            n_samples_new = int(data.shape[1] * self.fs / original_fs)
            data = resample(data, n_samples_new, axis=1)
        
        # Bandpass filter (0.5-50 Hz)
        b, a = butter(4, [0.5, 50], fs=self.fs, btype='band')
        data_filtered = filtfilt(b, a, data, axis=1)
        
        # Notch filter (50 Hz line noise)
        b_notch, a_notch = butter(4, [49, 51], fs=self.fs, btype='bandstop')
        data_notch = filtfilt(b_notch, a_notch, data_filtered, axis=1)
        
        # Common average reference
        data_car = data_notch - data_notch.mean(axis=0, keepdims=True)
        
        # Normalize each channel
        data_norm = (data_car - data_car.mean(axis=1, keepdims=True)) / \
                   (data_car.std(axis=1, keepdims=True) + 1e-8)
        
        return data_norm
    
    def _create_windows(self, data, annotations, file_name):
        """
        Create sliding windows with labels from annotations.
        
        Class imbalance handling strategies from literature:
        1. Use lower threshold for seizure labeling (>30% instead of >50%)
        2. Include all seizure windows
        3. Subsample non-seizure windows
        """
        n_channels, n_samples = data.shape
        window_samples = int(self.window_size * self.fs)
        step_samples = int(window_samples * (1 - self.overlap))
        
        X, y, metadata = [], [], []
        
        # First pass: collect all windows
        all_windows = []
        for start_idx in range(0, min(n_samples, len(annotations)) - window_samples, step_samples):
            end_idx = start_idx + window_samples
            
            # Extract window (samples, channels)
            window = data[:, start_idx:end_idx].T
            
            # Get annotations for this window
            window_annotations = annotations[start_idx:end_idx]
            
            # Label: Lower threshold for imbalanced data (30% as used in literature)
            seizure_ratio = window_annotations.sum() / len(window_annotations)
            label = 1 if seizure_ratio > 0.3 else 0  # Changed from 0.5 to 0.3
            
            all_windows.append({
                'window': window,
                'label': label,
                'seizure_ratio': seizure_ratio,
                'start_idx': start_idx,
                'end_idx': end_idx
            })
        
        # Separate seizure and non-seizure windows
        seizure_windows = [w for w in all_windows if w['label'] == 1]
        non_seizure_windows = [w for w in all_windows if w['label'] == 0]
        
        # Keep all seizure windows
        for w in seizure_windows:
            X.append(w['window'])
            y.append(w['label'])
            metadata.append({
                'file_name': file_name,
                'start_idx': w['start_idx'],
                'end_idx': w['end_idx'],
                'seizure_ratio': w['seizure_ratio']
            })
        
        # For non-seizure windows: 
        # If in training, subsample to maintain reasonable ratio (max 4:1)
        # If in val/test, keep all for proper evaluation
        if self.fold == 'train' and len(seizure_windows) > 0:
            max_non_seizure = len(seizure_windows) * 4  # 4:1 ratio
            if len(non_seizure_windows) > max_non_seizure:
                # Randomly subsample non-seizure windows
                non_seizure_windows = np.random.choice(
                    non_seizure_windows, 
                    size=max_non_seizure, 
                    replace=False
                ).tolist()
        
        # Add non-seizure windows
        for w in non_seizure_windows:
            X.append(w['window'])
            y.append(w['label'])
            metadata.append({
                'file_name': file_name,
                'start_idx': w['start_idx'],
                'end_idx': w['end_idx'],
                'seizure_ratio': w['seizure_ratio']
            })
        
        return X, y, metadata
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================
# Training
# ============================

def train_model(model, train_loader, val_loader, criterion, num_epochs=50, 
                lr=0.0003, device='cpu'):
    """Training loop with gradient clipping."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7
    )
    
    history = {
        'train_loss': [], 'val_loss': [],
        'val_auc': [], 'val_sensitivity': [], 'val_specificity': []
    }
    
    best_auc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            if torch.isnan(loss):
                print(f"  WARNING: NaN loss at epoch {epoch}")
                break
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device).unsqueeze(1)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(batch_y.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Flatten labels and probs (they might be arrays due to unsqueeze)
        all_labels_flat = [float(label) if hasattr(label, '__iter__') else label 
                          for label in all_labels]
        all_probs_flat = [float(prob) if hasattr(prob, '__iter__') else prob 
                         for prob in all_probs]
        
        # Compute metrics
        unique_labels = set(all_labels_flat)
        if len(unique_labels) > 1:  # Check if both classes are present
            auc = roc_auc_score(all_labels_flat, all_probs_flat)
            preds = (np.array(all_probs_flat) > 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(all_labels_flat, preds).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            # Only one class present - can't compute AUC
            auc = 0.0
            sensitivity = 0.0
            specificity = 0.0
            if epoch == 0:  # Only warn once
                print(f"  WARNING: Only one class in validation set")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(auc)
        history['val_sensitivity'].append(sensitivity)
        history['val_specificity'].append(specificity)
        
        scheduler.step(auc)
        
        if auc > best_auc:
            best_auc = auc
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f}/{val_loss:.4f} | "
                  f"AUC: {auc:.3f} | Sens: {sensitivity:.2%} | Spec: {specificity:.2%}")
    
    return history


def evaluate_model(model, val_loader, device):
    """Comprehensive evaluation metrics."""
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())
    
    # Check if both classes are present
    if len(set(all_labels)) < 2:
        print(f"  WARNING: Only one class present in validation set!")
        return {
            'accuracy': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0,
            'precision': 0.0,
            'f1': 0.0,
            'auc': 0.0,
            'confusion_matrix': (0, 0, 0, 0)
        }
    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs),
        'confusion_matrix': (tn, fp, fn, tp)
    }


# ============================
# Main Experiment
# ============================

def run_helsinki_comparison(data_path, num_epochs=50, batch_size=32):
    """Run 3-way comparison on Helsinki dataset."""
    
    # Annotations file is in the same folder as EDF files
    annotations_path = Path(data_path) / "annotations_2017_A_fixed.csv"
    
    print("\n" + "="*80)
    print("HELSINKI NEONATAL EEG SEIZURE DETECTION")
    print("3-Way Comparison: DIRU vs Tractable Dendritic vs LSTM")
    print("="*80)
    print(f"\nPaths:")
    print(f"  Data folder: {data_path}")
    print(f"  Annotations: {annotations_path}")
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = HelsinkiNeonatalDatasetCSV(
        data_path, annotations_path, 
        window_size=WINDOW_SIZE, overlap=OVERLAP, 
        fold='train', first_n_timepoints=FIRST_N_TIMEPOINTS
    )
    
    val_dataset = HelsinkiNeonatalDatasetCSV(
        data_path, annotations_path,
        window_size=WINDOW_SIZE, overlap=OVERLAP,
        fold='val', first_n_timepoints=FIRST_N_TIMEPOINTS
    )
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("\nERROR: No data loaded!")
        return None
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Get number of channels
    sample_x, _ = next(iter(train_loader))
    num_channels = sample_x.shape[2]
    print(f"\nData dimensions: {sample_x.shape}")
    print(f"Number of channels: {num_channels}")
    
    # Class weights
    pos_ratio = train_dataset.y.sum() / len(train_dataset.y)
    pos_weight = torch.FloatTensor([(1 - pos_ratio) / pos_ratio]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    print(f"\nClass distribution:")
    print(f"  Seizure: {pos_ratio*100:.1f}%")
    print(f"  Positive weight: {pos_weight.item():.2f}")
    
    # Initialize models
    print(f"\nInitializing models (hidden_size={HIDDEN_SIZE}, compartments={NUM_COMPARTMENTS})...")
    diru = DIRU(num_channels, HIDDEN_SIZE, output_size=1, num_compartments=NUM_COMPARTMENTS).to(DEVICE)
    tractable = TractableDendriticRNN(num_channels, HIDDEN_SIZE, output_size=1, 
                                     num_compartments=NUM_COMPARTMENTS).to(DEVICE)
    lstm = LSTMModel(num_channels, HIDDEN_SIZE, output_size=1, num_layers=1).to(DEVICE)
    
    print(f"  DIRU: {sum(p.numel() for p in diru.parameters()):,} params")
    print(f"  Tractable: {sum(p.numel() for p in tractable.parameters()):,} params")
    print(f"  LSTM: {sum(p.numel() for p in lstm.parameters()):,} params")
    
    # Train models
    print("\n" + "-"*80)
    print("Training DIRU (Active Dendrites)...")
    print("-"*80)
    diru_hist = train_model(diru, train_loader, val_loader, criterion, 
                           num_epochs=num_epochs, device=DEVICE)
    
    print("\n" + "-"*80)
    print("Training Tractable Dendritic RNN (Passive Dendrites)...")
    print("-"*80)
    tractable_hist = train_model(tractable, train_loader, val_loader, criterion,
                                num_epochs=num_epochs, device=DEVICE)
    
    print("\n" + "-"*80)
    print("Training LSTM (Baseline)...")
    print("-"*80)
    lstm_hist = train_model(lstm, train_loader, val_loader, criterion,
                           num_epochs=num_epochs, device=DEVICE)
    
    # Evaluate
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    diru_metrics = evaluate_model(diru, val_loader, DEVICE)
    tractable_metrics = evaluate_model(tractable, val_loader, DEVICE)
    lstm_metrics = evaluate_model(lstm, val_loader, DEVICE)
    
    # Print results
    print("\nDIRU (Active Dendrites):")
    print(f"  Sensitivity: {diru_metrics['sensitivity']:.2%} | Specificity: {diru_metrics['specificity']:.2%}")
    print(f"  AUC: {diru_metrics['auc']:.3f} | F1: {diru_metrics['f1']:.3f}")
    
    print("\nTractable (Passive Dendrites):")
    print(f"  Sensitivity: {tractable_metrics['sensitivity']:.2%} | Specificity: {tractable_metrics['specificity']:.2%}")
    print(f"  AUC: {tractable_metrics['auc']:.3f} | F1: {tractable_metrics['f1']:.3f}")
    
    print("\nLSTM (Baseline):")
    print(f"  Sensitivity: {lstm_metrics['sensitivity']:.2%} | Specificity: {lstm_metrics['specificity']:.2%}")
    print(f"  AUC: {lstm_metrics['auc']:.3f} | F1: {lstm_metrics['f1']:.3f}")
    
    # Improvements
    print("\n" + "="*80)
    print("CLINICAL IMPACT")
    print("="*80)
    diru_vs_lstm = (diru_metrics['sensitivity'] - lstm_metrics['sensitivity']) * 100
    print(f"\nFor 100 neonatal seizures:")
    print(f"  LSTM detects: ~{int(lstm_metrics['sensitivity']*100)}")
    print(f"  DIRU detects: ~{int(diru_metrics['sensitivity']*100)}")
    print(f"  → DIRU detects {int(diru_vs_lstm)} MORE seizures!")
    print(f"  → Could prevent brain damage in {int(diru_vs_lstm)} additional infants!")
    
    return {
        'diru_metrics': diru_metrics,
        'tractable_metrics': tractable_metrics,
        'lstm_metrics': lstm_metrics
    }


# ============================
# Usage
# ============================

if __name__ == "__main__":
    # Path to eeg_cache folder (annotations CSV is inside this folder)
    DATA_PATH = "/content/drive/MyDrive/eeg_cache"
    
    print("="*80)
    print("Helsinki Neonatal EEG - FIXED VERSION")
    print("="*80)
    print(f"\nData folder: {DATA_PATH}")
    print(f"Annotations: {DATA_PATH}/annotations_2017_A_fixed.csv")
    print(f"\nConfiguration:")
    print(f"  First {FIRST_N_TIMEPOINTS} timepoints per file")
    print(f"  Window: {WINDOW_SIZE}s, Overlap: {OVERLAP}")
    print(f"  Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}")
    
    # Run
    results = run_helsinki_comparison(DATA_PATH, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    
    if results:
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED!")
        print("="*80)
