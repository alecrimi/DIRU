"""
Helsinki Neonatal EEG - Band Power Spectrum Features  (v6 - LOO-CV)

Changes vs v5
─────────────
1. Leave-One-Out CV: features pre-computed once for all 79 recordings,
   z-score re-fitted inside each fold on the 78 training recordings.

2. DIRU fix: each compartment now receives the FULL 40-feature input
   with its own dedicated W_in weights — same input as LSTM/Tractable.
   The inductive bias is the K separate learned projections, not input
   restriction. Hidden size unchanged (32) — fair comparison.
"""

import numpy as np
import pandas as pd
import torch"""
Nigeria / Guinea-Bissau EEG — Control vs Epileptic Classification  (v1)

Data format
───────────
  - Gzipped CSV files (NOT EDF): EEGs_Guinea-Bissau/signal-{id}.csv.gz
                                  EEGs_Nigeria/signal-{session}.csv.gz
  - First column: sample index (discarded)
  - Columns 1–14: 14 EEG channels @ 128 Hz
  - Metadata CSV: subject.id, Group (Epilepsy/Control), Eyes.condition,
                  recordedPeriod, ...

Data selection (matching notebook logic)
─────────────────────────────────────────
  Only closed-eyes data is used (50% of rows, first or last half depending
  on Eyes.condition first word).  This matches load_eeg_from_gz() in the
  original notebook.

Feature pipeline (identical to Helsinki v6)
────────────────────────────────────────────
  Raw signal (14 ch, 128 Hz)
    → bandpass 1–40  Hz + bipolar referencing + per-channel z-score
    → 5 subbands → sliding 0.5s frame → mean(x²) → log1p
    → (T_seq=11, 70)   [14 ch × 5 bands]
    → z-score per LOO fold

LOO-CV: leave one SUBJECT out per fold.
Label: 0 = Control, 1 = Epilepsy (subject-level, repeated across windows).
"""

import gzip
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import butter, filtfilt, resample
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path
import gc
import pickle
from tqdm import tqdm

# ============================================================
# Config
# ============================================================
DATASET_DIR    = "/content/drive_mount/.shortcut-targets-by-id/1NtgDtnXESKUfllHY7gxV__B9Cb9Eciaz/DATASET" #"/content/drive/MyDrive/DATASET"   # folder with CSV and EEG subdirs
METADATA_CSV   = "metadata_guineabissau.csv"         # relative to DATASET_DIR
EEG_SUBDIR     = "EEGs_Guinea-Bissau"                # subfolder with .csv.gz files
FILE_PATTERN   = "signal-{subject_id}.csv.gz"        # {subject_id} = subject.id column

# For Nigeria, change to:
# METADATA_CSV  = "metadata_nigeria.csv"
# EEG_SUBDIR    = "EEGs_Nigeria"
# FILE_PATTERN  = "signal-{csv_file}"   # nigeria uses session-based filenames in csv.file col
# Nigeria also needs: FILE_COL = "csv.file"  (already contains path fragment)

CHECKPOINT_DIR = "/content/drive/MyDrive/checkpoints"
FEATURE_CACHE  = "/content/drive/MyDrive/checkpoints/guinea_features_v1.pkl"

# Signal
SOURCE_FS    = 128        # sampling frequency of the recordings
TARGET_FS    = 128        # keep as-is (no resample needed)
N_CHANNELS   = 14        # columns 1–14 of each CSV
EEG_COL_SLICE = slice(1, 15)   # iloc[:, 1:15] — skip index column

# Closed-eyes extraction (matching notebook: 50% of rows)
CLOSED_EYES_PERC = 0.5

# Window / feature
WINDOW_SIZE    = 3        # seconds
OVERLAP        = 0.75
FRAME_LEN_SEC  = 0.5
FRAME_STEP_SEC = 0.25

SUBBANDS = [
    ("delta", 0.5,  4.0),
    ("theta", 4.0,  8.0),
    ("alpha", 8.0,  13.0),
    ("beta",  13.0, 30.0),
    ("gamma", 30.0, 45.0),
]
N_BANDS         = len(SUBBANDS)
FULL_INPUT_SIZE = N_BANDS * N_CHANNELS   # 70

# Model
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_SIZE    = 32
N_COMPARTMENTS = 4
DIRU_DROPOUT   = 0.5
NUM_EPOCHS     = 40
BATCH_SIZE     = 32

# Derived
_FL       = int(FRAME_LEN_SEC  * TARGET_FS)
_FS_STEP  = int(FRAME_STEP_SEC * TARGET_FS)
_WIN_SAMP = int(WINDOW_SIZE    * TARGET_FS)
T_SEQ     = 1 + (_WIN_SAMP - _FL) // _FS_STEP   # 11

print(f"Device={DEVICE}  T_seq={T_SEQ}  features={FULL_INPUT_SIZE}")
print(f"hidden={HIDDEN_SIZE} (all models)  epochs={NUM_EPOCHS}  batch={BATCH_SIZE}")
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)


# ============================================================
# Data loading
# ============================================================

def load_closed_eyes(file_path, first_condition, file_rows_count=None):
    """
    Load the closed-eyes portion of a .csv.gz EEG file.
    Returns (14, n_samples) numpy array or None on error.

    first_condition : "closed" → use first 50% of rows
                      "open"   → use last 50% of rows
    """
    try:
        perc = CLOSED_EYES_PERC

        with gzip.open(file_path, 'rt') as f:
            if file_rows_count is not None:
                rows_to_read = int(file_rows_count * perc)
                if first_condition == "closed":
                    df = pd.read_csv(f, nrows=rows_to_read, on_bad_lines='skip')
                else:  # "open" — skip first half
                    skip_rows = file_rows_count - rows_to_read
                    df = pd.read_csv(f, skiprows=range(1, skip_rows + 1),
                                     on_bad_lines='skip')
            else:
                # No row count available: load all, then slice
                df = pd.read_csv(f, on_bad_lines='skip')
                rows_to_read = int(len(df) * perc)
                if first_condition == "closed":
                    df = df.iloc[:rows_to_read]
                else:
                    df = df.iloc[-rows_to_read:]

    except Exception as e:
        print(f"  ERROR loading {Path(file_path).name}: {e}")
        return None

    try:
        eeg = df.iloc[:, EEG_COL_SLICE].values.T.astype(np.float32)  # (14, T)
        if eeg.shape[0] != N_CHANNELS:
            print(f"  WARNING: {Path(file_path).name} has {eeg.shape[0]} ch, expected {N_CHANNELS}")
            return None
    except Exception as e:
        print(f"  ERROR extracting channels from {Path(file_path).name}: {e}")
        return None

    return eeg


def preprocess(data):
    """
    (14, n_samples) raw EEG → (14, n_samples) preprocessed.
    Bandpass 1–45 Hz, notch 49–51 Hz, CAR, per-channel z-score.
    """
    fs = TARGET_FS
    try:
        b, a     = butter(4, [1.0, 45.0], fs=fs, btype='band')
        data     = filtfilt(b, a, data, axis=1)
        b_n, a_n = butter(4, [49, 51],   fs=fs, btype='bandstop')
        data     = filtfilt(b_n, a_n, data, axis=1)
        data    -= data.mean(axis=0, keepdims=True)                 # CAR
        data     = (data - data.mean(axis=1, keepdims=True)) / \
                   (data.std(axis=1, keepdims=True) + 1e-8)         # per-ch z-score
    except Exception as e:
        print(f"  ERROR in preprocess: {e}")
        return None
    return data


# ============================================================
# Band-power feature extraction
# ============================================================

def _bp(data, lo, hi, fs, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [max(lo/nyq, 1e-4), min(hi/nyq, 1-1e-4)], btype='band')
    return filtfilt(b, a, data, axis=1)


def compute_band_power_sequence(data):
    """(N_CHANNELS, n_samp) -> (T_seq, N_BANDS*N_CHANNELS)  log band-power."""
    n_ch, n_samp = data.shape
    starts   = np.arange(0, n_samp - _FL + 1, _FS_STEP)
    n_frames = len(starts)
    power    = np.zeros((N_BANDS, n_ch, n_frames), dtype=np.float32)
    for b_i, (_, lo, hi) in enumerate(SUBBANDS):
        band = _bp(data, lo, hi, TARGET_FS)
        for f_i, s in enumerate(starts):
            power[b_i, :, f_i] = np.mean(band[:, s:s+_FL] ** 2, axis=1)
    power = np.log1p(power)
    power = power.transpose(2, 0, 1)     # (T, B, C)
    return power.reshape(n_frames, N_BANDS * n_ch).astype(np.float32)


# ============================================================
# Pre-compute all features once
# ============================================================

def build_all_features(dataset_dir):
    """
    Reads metadata CSV, loads each subject's closed-eyes EEG,
    extracts band-power windows.

    Returns list of dicts:
      { 'subject_id', 'subject_label': 0/1,
        'X': (n_win, T_seq, 70), 'y': (n_win,) }
    """
    cache = Path(FEATURE_CACHE)
    if cache.exists():
        print(f"Loading cached features: {cache}")
        recs = pickle.load(open(cache, 'rb'))
        print(f"  {len(recs)} subjects  "
              f"(ctrl={sum(r['subject_label']==0 for r in recs)}, "
              f"epi={sum(r['subject_label']==1 for r in recs)})")
        return recs

    dataset_dir = Path(dataset_dir)
    meta = pd.read_csv(dataset_dir / METADATA_CSV)

    # Derive first_condition from Eyes.condition
    meta['first_condition'] = meta['Eyes.condition'].str.split('-').str[0]

    # Build file paths
    meta['file_path'] = meta['subject.id'].apply(
        lambda sid: str(dataset_dir / EEG_SUBDIR /
                        FILE_PATTERN.format(subject_id=sid)))

    # Binary label
    meta['label'] = (meta['Group'].str.strip().str.lower() == 'epilepsy').astype(int)

    win_samp  = _WIN_SAMP
    step_samp = max(1, int(win_samp * (1 - OVERLAP)))
    recs      = []

    for _, row in tqdm(meta.iterrows(), total=len(meta),
                       desc="Pre-computing features"):
        fp = row['file_path']
        if not Path(fp).exists():
            print(f"  MISSING: {Path(fp).name}")
            continue

        # row count if available (speeds up loading)
        row_count = int(row['recordedPeriod'] * SOURCE_FS) \
                    if 'recordedPeriod' in row else None

        raw = load_closed_eyes(fp, row['first_condition'], row_count)
        if raw is None:
            continue

        data = preprocess(raw)
        if data is None:
            continue

        n_samp = data.shape[1]
        X_list = []
        for start in range(0, n_samp - win_samp, step_samp):
            window = data[:, start:start+win_samp]
            ps = compute_band_power_sequence(window)
            ps = ps[:T_SEQ] if ps.shape[0] >= T_SEQ else \
                 np.vstack([ps, np.zeros((T_SEQ - ps.shape[0], FULL_INPUT_SIZE))])
            X_list.append(ps)

        if not X_list:
            continue

        X = np.stack(X_list).astype(np.float32)
        y = np.full(len(X_list), float(row['label']), dtype=np.float32)

        recs.append({
            'subject_id':    row['subject.id'],
            'subject_label': int(row['label']),
            'X': X, 'y': y,
        })

    n_ctrl = sum(1 for r in recs if r['subject_label'] == 0)
    n_epi  = sum(1 for r in recs if r['subject_label'] == 1)
    print(f"\nDone: {len(recs)} subjects  ({n_ctrl} control, {n_epi} epilepsy)")
    pickle.dump(recs, open(cache, 'wb'))
    print(f"Cached: {cache}")
    return recs


# ============================================================
# Models — identical to Helsinki v6
# ============================================================

class DIRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_compartments):
        super().__init__()
        self.num_comp = num_compartments
        self.W_in  = nn.ModuleList([
            nn.Linear(input_size, hidden_size) for _ in range(num_compartments)
        ])
        self.W_rec = nn.Linear(hidden_size, hidden_size * num_compartments)
        self.gate  = nn.Linear(hidden_size * num_compartments, num_compartments)

    def forward(self, x, h):
        rec   = self.W_rec(h).chunk(self.num_comp, dim=1)
        outs  = [torch.tanh(self.W_in[i](x) + rec[i]) for i in range(self.num_comp)]
        stack = torch.stack(outs, dim=1)
        w     = torch.softmax(self.gate(stack.reshape(stack.size(0), -1)), dim=1)
        return (stack * w.unsqueeze(2)).sum(dim=1)


class DIRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_compartments=N_COMPARTMENTS, dropout=DIRU_DROPOUT):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell        = DIRUCell(input_size, hidden_size, num_compartments)
        self.dropout     = nn.Dropout(dropout)
        self.fc          = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t], h)
        return self.fc(self.dropout(h))


class TractableDendriticCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_compartments):
        super().__init__()
        self.num_comp  = num_compartments
        self.comp_size = hidden_size // num_compartments
        assert hidden_size % num_compartments == 0
        self.W_in  = nn.ModuleList([
            nn.Linear(input_size,  self.comp_size) for _ in range(num_compartments)
        ])
        self.W_rec = nn.ModuleList([
            nn.Linear(hidden_size, self.comp_size) for _ in range(num_compartments)
        ])
        self.integration = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        outs = [torch.tanh(self.W_in[i](x) + self.W_rec[i](h))
                for i in range(self.num_comp)]
        return torch.tanh(self.integration(torch.cat(outs, dim=1)))


class TractableDendriticRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_compartments=N_COMPARTMENTS):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell        = TractableDendriticCell(input_size, hidden_size, num_compartments)
        self.dropout     = nn.Dropout(0.5)
        self.fc          = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t], h)
        return self.fc(self.dropout(h))


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                               batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1]))


def make_models():
    return {
        'diru':      DIRU(FULL_INPUT_SIZE, HIDDEN_SIZE, 1).to(DEVICE),
        'tractable': TractableDendriticRNN(FULL_INPUT_SIZE, HIDDEN_SIZE, 1).to(DEVICE),
        'lstm':      LSTMModel(FULL_INPUT_SIZE, HIDDEN_SIZE, 1).to(DEVICE),
    }


# ============================================================
# Per-fold training (identical to Helsinki v6)
# ============================================================

def train_fold(model, X_tr, y_tr, X_val, y_val):
    tr_ds     = TensorDataset(torch.FloatTensor(X_tr),
                               torch.FloatTensor(y_tr).unsqueeze(1))
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)

    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    pw    = torch.FloatTensor([n_neg / max(n_pos, 1)]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    optimizer     = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup_sched  = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=3)
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-5)

    best_val_loss = float('inf')
    best_state    = None
    patience_ctr  = 0

    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(DEVICE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(bx), by).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = nn.BCEWithLogitsLoss()(model(X_val_t), y_val_t).item()

        if epoch < 3:
            warmup_sched.step()
        else:
            plateau_sched.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= 8:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X_val_t)).cpu().numpy().flatten()


# ============================================================
# LOO loop — leave one subject out
# ============================================================

LOO_PROGRESS_CACHE = str(Path(CHECKPOINT_DIR) / 'guinea_loo_progress.pkl')

def run_loo(recordings):
    """
    Leave-one-subject-out CV with fold-level checkpointing.
    If interrupted, re-running resumes from the last completed fold.
    Delete LOO_PROGRESS_CACHE to start fresh.
    """
    n = len(recordings)

    # Load existing progress if available
    progress_path = Path(LOO_PROGRESS_CACHE)
    if progress_path.exists():
        saved = pickle.load(open(progress_path, 'rb'))
        all_probs   = saved['all_probs']
        all_labels  = saved['all_labels']
        start_fold  = saved['next_fold']
        print(f"Resuming LOO from fold {start_fold}/{n} "
              f"({start_fold} folds already done)")
    else:
        all_probs   = {name: [] for name in ('diru', 'tractable', 'lstm')}
        all_labels  = {name: [] for name in ('diru', 'tractable', 'lstm')}
        start_fold  = 0

    for fold_i in tqdm(range(start_fold, n), desc="LOO folds",
                       initial=start_fold, total=n):
        test_rec   = recordings[fold_i]
        train_recs = [recordings[j] for j in range(n) if j != fold_i]

        X_tr = np.concatenate([r['X'] for r in train_recs], axis=0)
        y_tr = np.concatenate([r['y'] for r in train_recs], axis=0)
        X_te = test_rec['X']
        y_te = test_rec['y']

        # z-score: fit on train, apply to test
        flat   = X_tr.reshape(-1, FULL_INPUT_SIZE)
        mu     = flat.mean(axis=0, keepdims=True)
        std    = flat.std( axis=0, keepdims=True)
        X_tr_n = (X_tr - mu[None]) / (std[None] + 1e-8)
        X_te_n = (X_te - mu[None]) / (std[None] + 1e-8)

        models = make_models()
        for name, model in models.items():
            probs = train_fold(model, X_tr_n, y_tr, X_te_n, y_te)
            all_probs[name].extend(probs.tolist())
            all_labels[name].extend(y_te.tolist())
        del models

        # Save progress after every completed fold
        pickle.dump({
            'all_probs':  all_probs,
            'all_labels': all_labels,
            'next_fold':  fold_i + 1,
        }, open(progress_path, 'wb'))

        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    # All folds done — remove progress file
    if progress_path.exists():
        progress_path.unlink()
        print("LOO complete — progress cache cleared")

    return all_probs, all_labels


# ============================================================
# Evaluation & plotting
# ============================================================

def evaluate(all_probs, all_labels):
    print(f"\n{'='*60}\nLOO-CV SUMMARY  (Control=0  Epilepsy=1)\n{'='*60}")
    summary = {}
    for name in ('diru', 'tractable', 'lstm'):
        probs  = np.array(all_probs[name])
        labels = np.array(all_labels[name])

        # Optimal threshold maximising F1
        best_f1, best_thr = 0., 0.5
        for thr in np.linspace(0.05, 0.95, 181):
            p  = (probs > thr).astype(int)
            tp = int(((p==1)&(labels==1)).sum())
            fp = int(((p==1)&(labels==0)).sum())
            fn = int(((p==0)&(labels==1)).sum())
            f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.
            if f1 > best_f1:
                best_f1, best_thr = f1, thr

        preds = (probs > best_thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        sens = tp/(tp+fn) if (tp+fn) else 0.
        spec = tn/(tn+fp) if (tn+fp) else 0.
        auc  = roc_auc_score(labels, probs)
        f1   = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.
        acc  = accuracy_score(labels, preds)

        summary[name] = dict(sens=sens, spec=spec, auc=auc, f1=f1, acc=acc,
                             probs=probs, labels=labels, threshold=best_thr)
        print(f"  {name.upper():12s}  Sens={sens:.3f}  Spec={spec:.3f}  "
              f"AUC={auc:.3f}  F1={f1:.3f}  Acc={acc:.3f}  thr={best_thr:.2f}")

    d_auc = summary['diru']['auc']
    l_auc = summary['lstm']['auc']
    print(f"\n  DIRU vs LSTM  ΔAUC = {(d_auc-l_auc)*100:+.1f} pp")
    return summary


def plot_roc(summary, save_path=None):
    plt.figure(figsize=(8, 7))
    styles = {'diru':      ('royalblue', '-'),
              'tractable': ('seagreen',  '--'),
              'lstm':      ('firebrick', ':')}
    for name, (col, ls) in styles.items():
        m = summary[name]
        fpr, tpr, _ = roc_curve(m['labels'], m['probs'])
        plt.plot(fpr, tpr, color=col, ls=ls, lw=2,
                 label=f"{name.upper()}  AUC={m['auc']:.3f}  Sens={m['sens']:.3f}")
    plt.plot([0,1],[0,1],'k--',lw=1,label='Chance')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate',  fontsize=12)
    plt.title('LOO-CV ROC — Control vs Epileptic (Guinea-Bissau EEG)', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC saved: {save_path}")
    plt.show()


# ============================================================
# Main
# ============================================================

def run(dataset_dir=DATASET_DIR, clear_cache=False):
    if clear_cache:
        p = Path(FEATURE_CACHE)
        if p.exists():
            p.unlink()
            print("Feature cache cleared")

    recordings = build_all_features(dataset_dir)

    models_ex = make_models()
    for name, m in models_ex.items():
        print(f"  {name:12s}: {sum(p.numel() for p in m.parameters()):,} params")
    del models_ex

    n_ctrl = sum(1 for r in recordings if r['subject_label'] == 0)
    n_epi  = sum(1 for r in recordings if r['subject_label'] == 1)
    print(f"\nSubjects: {len(recordings)}  ({n_ctrl} control, {n_epi} epilepsy)")
    print(f"Windows per subject: ~{int(np.mean([len(r['y']) for r in recordings]))}")

    all_probs, all_labels = run_loo(recordings)
    summary = evaluate(all_probs, all_labels)

    rpath = Path(CHECKPOINT_DIR) / 'guinea_loo_results_v1.pkl'
    pickle.dump({'probs': all_probs, 'labels': all_labels, 'summary': summary},
                open(rpath, 'wb'))
    print(f"Results saved: {rpath}")

    plot_roc(summary, save_path=Path(CHECKPOINT_DIR) / 'roc_guinea_v1.png')
    return summary


if __name__ == "__main__":
    CLEAR_CACHE = False
    summary = run(clear_cache=CLEAR_CACHE)
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import butter, filtfilt, resample
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
from pathlib import Path
import pyedflib
import gc
import pickle
from tqdm import tqdm
from collections import defaultdict

# ============================================================
# Config
# ============================================================
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 40
BATCH_SIZE = 32
HIDDEN_SIZE     = 32
DIRU_DROPOUT    = 0.5
N_COMPARTMENTS  = 4

WINDOW_SIZE        = 3
OVERLAP            = 0.75
FIRST_N_TIMEPOINTS = 15417
FRAME_LEN_SEC      = 0.5
FRAME_STEP_SEC     = 0.25

#SELECTED_CHANNELS = [2, 3, 4, 5, 12, 13, 8, 9]
SELECTED_CHANNELS = [1, 2, 3, 4, 11, 12, 7, 8]   # 8 channels, matching original selection minus CZ

N_CHANNELS        = len(SELECTED_CHANNELS)   # 8

SUBBANDS = [
    ("delta", 0.5,  4.0),
    ("theta", 4.0,  8.0),
    ("alpha", 8.0,  13.0),
    ("beta",  13.0, 30.0),
    ("gamma", 30.0, 50.0),
]
N_BANDS         = len(SUBBANDS)
FULL_INPUT_SIZE = N_BANDS * N_CHANNELS   # 40

CHECKPOINT_DIR = "/content/drive/MyDrive/checkpoints"
FEATURE_CACHE  = "/content/drive/MyDrive/checkpoints/loo_features_v6.pkl"

_FS       = 256
_FL       = int(FRAME_LEN_SEC  * _FS)
_FS_STEP  = int(FRAME_STEP_SEC * _FS)
_WIN_SAMP = int(WINDOW_SIZE    * _FS)
T_SEQ     = 1 + (_WIN_SAMP - _FL) // _FS_STEP   # 11

print(f"Device={DEVICE}  T_seq={T_SEQ}  features={FULL_INPUT_SIZE}")
print(f"hidden={HIDDEN_SIZE} (all models)  epochs={NUM_EPOCHS}  batch={BATCH_SIZE}")
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)


# ============================================================
# Signal processing
# ============================================================
# ============================
# Bipolar montage
# ============================
def bipolar_montage(data):
    """
    Convert monopolar EEG to bipolar montage.
    Input:  (channels, samples) → e.g. (21, T)
    Output: (channels-1, samples) → e.g. (20, T)
    """
    return data[1:, :] - data[:-1, :]


def _bp(data, lo, hi, fs, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [max(lo/nyq, 1e-4), min(hi/nyq, 1-1e-4)], btype='band')
    return filtfilt(b, a, data, axis=1)


def compute_band_power_sequence(data, fs=_FS):
    """(n_ch, n_samp) -> (T_seq, 40)  log band-power, unnormalised."""
    n_ch, n_samp = data.shape
    starts   = np.arange(0, n_samp - _FL + 1, _FS_STEP)
    n_frames = len(starts)
    power    = np.zeros((N_BANDS, n_ch, n_frames), dtype=np.float32)

    for b_i, (_, lo, hi) in enumerate(SUBBANDS):
        band = _bp(data, lo, hi, fs)
        for f_i, s in enumerate(starts):
            power[b_i, :, f_i] = np.mean(band[:, s:s+_FL] ** 2, axis=1)

    power = np.log1p(power)
    power = power.transpose(2, 0, 1)           # (T, B, C)
    return power.reshape(n_frames, N_BANDS * n_ch).astype(np.float32)


def load_recording(file_path):
    """Load EDF -> (8, n_samples) preprocessed signal."""
    try:
        edf   = pyedflib.EdfReader(str(file_path))
        sfreq = edf.getSampleFrequency(0)
        data  = np.array([edf.readSignal(i) for i in range(edf.signals_in_file)])
        edf.close()
    except Exception as e:
        print(f"  ERROR loading {Path(file_path).name}: {e}")
        return None
    try:
        if sfreq != _FS:
            data = resample(data, int(data.shape[1] * _FS / sfreq), axis=1)
        b, a     = butter(4, [0.5, 50], fs=_FS, btype='band')
        data     = filtfilt(b, a, data, axis=1)
        #b_n, a_n = butter(4, [49, 51], fs=_FS, btype='bandstop')
        #data     = filtfilt(b_n, a_n, data, axis=1)

        data = bipolar_montage(data)    

        #data    -= data.mean(axis=0, keepdims=True)                     # CAR
        data     = (data - data.mean(axis=1, keepdims=True)) / \
                   (data.std(axis=1, keepdims=True) + 1e-8)             # per-ch z-score
        data     = data[:, :FIRST_N_TIMEPOINTS]
        data     = data[SELECTED_CHANNELS, :]
    except Exception as e:
        print(f"  ERROR preprocessing {Path(file_path).name}: {e}")
        return None
    return data


# ============================================================
# Pre-compute all features once
# ============================================================

def build_all_features(data_path, annotations_path):
    """
    Returns list of dicts, one per valid recording:
      { 'rec_idx', 'file', 'X': (n_win, T_seq, 40), 'y': (n_win,) }
    Raw log-power — z-score applied per LOO fold, not here.
    Cached to disk so second run is instant.
    """
    cache = Path(FEATURE_CACHE)
    if cache.exists():
        print(f"Loading cached features: {cache}")
        recs = pickle.load(open(cache, 'rb'))
        print(f"  {len(recs)} recordings loaded")
        return recs

    ann_df    = pd.read_csv(annotations_path, header=None)
    edf_files = sorted(Path(data_path).glob("eeg*.edf"),
                       key=lambda x: int(''.join(filter(str.isdigit, x.stem))))[:79]

    win_samp  = _WIN_SAMP
    step_samp = max(1, int(win_samp * (1 - OVERLAP)))
    recs      = []

    for edf_file in tqdm(edf_files, desc="Pre-computing features (runs once)"):
        rec_num = int(''.join(filter(str.isdigit, edf_file.stem)))
        col_idx = rec_num - 1
        if col_idx >= ann_df.shape[1]:
            continue

        data = load_recording(edf_file)
        if data is None:
            continue

        annot  = ann_df[col_idx].values[:FIRST_N_TIMEPOINTS]
        X_list, y_list = [], []

        for start in range(0, len(annot) - win_samp, step_samp):
            end    = start + win_samp
            ratio  = float(annot[start:end].sum()) / win_samp
            window = data[:, start:end]
            if window.shape[1] < win_samp:
                window = np.concatenate(
                    [window, np.zeros((window.shape[0], win_samp - window.shape[1]))],
                    axis=1)
            ps = compute_band_power_sequence(window)
            ps = ps[:T_SEQ] if ps.shape[0] >= T_SEQ else \
                 np.vstack([ps, np.zeros((T_SEQ - ps.shape[0], FULL_INPUT_SIZE))])
            X_list.append(ps)
            y_list.append(float(ratio > 0.1))

        if not X_list:
            continue

        recs.append({
            'rec_idx': col_idx,
            'file':    edf_file.name,
            'X':       np.stack(X_list).astype(np.float32),
            'y':       np.array(y_list, dtype=np.float32),
        })

    print(f"Done: {len(recs)} recordings extracted")
    pickle.dump(recs, open(cache, 'wb'))
    print(f"Cached to: {cache}")
    return recs


# ============================================================
# Models — all take (B, T_seq, 40), same hidden_size=32
# ============================================================

class DIRUCell(nn.Module):
    """
    K compartments, each with its own W_in over the FULL input (40 features).
    Inductive bias: K separate learned projections, not input restriction.
    """
    def __init__(self, input_size, hidden_size, num_compartments):
        super().__init__()
        self.num_comp = num_compartments
        self.W_in  = nn.ModuleList([
            nn.Linear(input_size, hidden_size) for _ in range(num_compartments)
        ])
        self.W_rec = nn.Linear(hidden_size, hidden_size * num_compartments)
        # gate outputs K scalars (one attention weight per compartment)
        self.gate  = nn.Linear(hidden_size * num_compartments, num_compartments)


    def forward(self, x, h):
        rec   = self.W_rec(h).chunk(self.num_comp, dim=1)
        outs  = [torch.tanh(self.W_in[i](x) + rec[i]) for i in range(self.num_comp)]
        stack = torch.stack(outs, dim=1)                        # (B, K, H)
        # gate: K attention weights via softmax — output magnitude preserved
        w = torch.softmax(self.gate(stack.reshape(stack.size(0), -1)), dim=1)  # (B, K)
        return (stack * w.unsqueeze(2)).sum(dim=1)              # (B, H)


class DIRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_compartments=N_COMPARTMENTS, dropout=DIRU_DROPOUT):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell        = DIRUCell(input_size, hidden_size, num_compartments)
        self.dropout     = nn.Dropout(dropout)
        self.fc          = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t], h)
        return self.fc(self.dropout(h))


class TractableDendriticCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_compartments):
        super().__init__()
        self.num_comp  = num_compartments
        self.comp_size = hidden_size // num_compartments
        assert hidden_size % num_compartments == 0
        self.W_in  = nn.ModuleList([
            nn.Linear(input_size,  self.comp_size) for _ in range(num_compartments)
        ])
        self.W_rec = nn.ModuleList([
            nn.Linear(hidden_size, self.comp_size) for _ in range(num_compartments)
        ])
        self.integration = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        outs = [torch.tanh(self.W_in[i](x) + self.W_rec[i](h))
                for i in range(self.num_comp)]
        return torch.tanh(self.integration(torch.cat(outs, dim=1)))


class TractableDendriticRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 num_compartments=N_COMPARTMENTS):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell        = TractableDendriticCell(input_size, hidden_size, num_compartments)
        self.dropout     = nn.Dropout(0.5)
        self.fc          = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t], h)
        return self.fc(self.dropout(h))


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                               batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc      = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1]))


def make_models():
    return {
        'diru':      DIRU(FULL_INPUT_SIZE, HIDDEN_SIZE, 1).to(DEVICE),
        'tractable': TractableDendriticRNN(FULL_INPUT_SIZE, HIDDEN_SIZE, 1).to(DEVICE),
        'lstm':      LSTMModel(FULL_INPUT_SIZE, HIDDEN_SIZE, 1).to(DEVICE),
    }


# ============================================================
# Per-fold training
# ============================================================

def train_fold(model, X_tr, y_tr, X_val, y_val):
    """
    Train on (X_tr, y_tr), evaluate on (X_val, y_val).
    No checkpointing — each fold trains from scratch and is fast.
    Returns: probs array for X_val.
    """
    tr_ds     = TensorDataset(torch.FloatTensor(X_tr),
                               torch.FloatTensor(y_tr).unsqueeze(1))
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)

    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    pw    = torch.FloatTensor([n_neg / max(n_pos, 1)]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    optimizer     = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    warmup_epochs = 3
    warmup_sched  = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    plateau_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-5)

    best_val_loss = float('inf')
    best_state    = None
    patience_ctr  = 0
    patience      = 8

    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(DEVICE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        ep_loss = 0.0
        for bx, by in tr_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        ep_loss /= len(tr_loader)

        model.eval()
        with torch.no_grad():
            val_loss = nn.BCEWithLogitsLoss()(model(X_val_t), y_val_t).item()

        if epoch < warmup_epochs:
            warmup_sched.step()
        else:
            plateau_sched.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_val_t)).cpu().numpy().flatten()
    return probs


# ============================================================
# LOO loop
# ============================================================

def run_loo(recordings):
    """
    Leave-one-recording-out CV.
    For each fold i: train on 75 recordings, test on 1.
    z-score fitted on train windows only, applied to test.
    ALL predictions aggregated — including recordings with only one class.
    AUC is only meaningful on the pooled set (both classes present overall).
    """
    n           = len(recordings)
    all_probs   = {name: [] for name in ('diru', 'tractable', 'lstm')}
    all_labels  = {name: [] for name in ('diru', 'tractable', 'lstm')}
    single_class_folds = 0

    for fold_i in tqdm(range(n), desc="LOO folds"):
        test_rec   = recordings[fold_i]
        train_recs = [recordings[j] for j in range(n) if j != fold_i]

        X_tr = np.concatenate([r['X'] for r in train_recs], axis=0)
        y_tr = np.concatenate([r['y'] for r in train_recs], axis=0)
        X_te = test_rec['X']
        y_te = test_rec['y']

        if len(np.unique(y_te)) < 2:
            single_class_folds += 1
            # Still include predictions — they contribute TN/TP to pooled eval

        # z-score: fit on train, apply to test — no leakage
        flat   = X_tr.reshape(-1, FULL_INPUT_SIZE)
        mu     = flat.mean(axis=0, keepdims=True)
        std    = flat.std( axis=0, keepdims=True)
        X_tr_n = (X_tr - mu[None]) / (std[None] + 1e-8)
        X_te_n = (X_te - mu[None]) / (std[None] + 1e-8)

        models = make_models()
        for name, model in models.items():
            probs = train_fold(model, X_tr_n, y_tr, X_te_n, y_te)
            all_probs[name].extend(probs.tolist())
            all_labels[name].extend(y_te.tolist())

        del models
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    print(f"  {single_class_folds} folds had single-class test recordings "
          f"(predictions included in pooled eval)")
    return all_probs, all_labels


# ============================================================
# Evaluation & plotting
# ============================================================

def evaluate(all_probs, all_labels):
    print(f"\n{'='*60}\nLOO-CV SUMMARY\n{'='*60}")
    summary = {}
    for name in ('diru', 'tractable', 'lstm'):
        probs  = np.array(all_probs[name])
        labels = np.array(all_labels[name])

        # Find threshold maximising F1 over all LOO predictions
        thresholds = np.linspace(0.05, 0.95, 181)
        best_f1, best_thr = 0., 0.5
        for thr in thresholds:
            preds_t = (probs > thr).astype(int)
            tp = int(((preds_t == 1) & (labels == 1)).sum())
            fp = int(((preds_t == 1) & (labels == 0)).sum())
            fn = int(((preds_t == 0) & (labels == 1)).sum())
            f1_t = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) else 0.
            if f1_t > best_f1:
                best_f1, best_thr = f1_t, thr

        preds = (probs > best_thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.
        spec = tn / (tn + fp) if (tn + fp) else 0.
        auc  = roc_auc_score(labels, probs)
        f1   = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) else 0.
        acc  = accuracy_score(labels, preds)
        summary[name] = dict(sens=sens, spec=spec, auc=auc, f1=f1, acc=acc,
                             probs=probs, labels=labels, threshold=best_thr)
        print(f"  {name.upper():12s}  Sens={sens:.3f}  Spec={spec:.3f}  "
              f"AUC={auc:.3f}  F1={f1:.3f}  Acc={acc:.3f}  thr={best_thr:.2f}")

    d_auc = summary['diru']['auc']
    l_auc = summary['lstm']['auc']
    print(f"\n  DIRU vs LSTM  ΔAUC = {(d_auc - l_auc)*100:+.1f} pp")
    return summary


def plot_roc(summary, save_path=None):
    plt.figure(figsize=(8, 7))
    styles = {'diru':      ('royalblue', '-'),
              'tractable': ('seagreen',  '--'),
              'lstm':      ('firebrick', ':')}
    for name, (col, ls) in styles.items():
        m = summary[name]
        fpr, tpr, _ = roc_curve(m['labels'], m['probs'])
        plt.plot(fpr, tpr, color=col, ls=ls, lw=2,
                 label=f"{name.upper()}  AUC={m['auc']:.3f}  Sens={m['sens']:.3f}")
    plt.plot([0,1],[0,1],'k--',lw=1,label='Chance')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate',  fontsize=12)
    plt.title('LOO-CV ROC — Neonatal Seizure (Band-Power v6)', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC saved: {save_path}")
    plt.show()


# ============================================================
# Main
# ============================================================

def run(data_path, clear_cache=False):
    if clear_cache:
        p = Path(FEATURE_CACHE)
        if p.exists():
            p.unlink()
            print("Feature cache cleared")

    ann_path   = Path(data_path) / "annotations_2017_A_fixed.csv"
    recordings = build_all_features(data_path, ann_path)

    # Print param counts
    models_ex = make_models()
    for name, m in models_ex.items():
        n_params = sum(p.numel() for p in m.parameters())
        print(f"  {name:12s}: {n_params:,} params")
    del models_ex

    print(f"\nStarting LOO over {len(recordings)} recordings ...")
    all_probs, all_labels = run_loo(recordings)

    summary = evaluate(all_probs, all_labels)

    rpath = Path(CHECKPOINT_DIR) / 'loo_results_v6.pkl'
    pickle.dump({'probs': all_probs, 'labels': all_labels, 'summary': summary},
                open(rpath, 'wb'))
    print(f"Results saved: {rpath}")

    plot_roc(summary, save_path=Path(CHECKPOINT_DIR) / 'roc_loo_v6.png')
    return summary


if __name__ == "__main__":
    DATA_PATH   = "/content/drive/MyDrive/eeg_cache"
    CLEAR_CACHE = False   # set True to re-extract features after pipeline changes

    summary = run(DATA_PATH, clear_cache=CLEAR_CACHE)
