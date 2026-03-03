"""
EEG Seizure Visualizer
======================
Loads real EDF recordings and the annotations_2017_A_fixed.csv from the same
paths used in the training pipeline, then plots any recording with its
seizure and pre-ictal phases clearly marked.

Usage (in Colab, same cell style as your pipeline):
    DATA_PATH  = "/content/drive/MyDrive/eeg_cache"
    REC_NUMBER = 1          # which recording (1-indexed, matches EDF filename)
    PREICTAL_SEC = 8        # seconds before seizure onset to mark as pre-ictal
    run_visualization(DATA_PATH, REC_NUMBER, preictal_sec=PREICTAL_SEC)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt, resample
from pathlib import Path
import pyedflib

# ============================================================
# Config — mirrors your training pipeline exactly
# ============================================================
_FS               = 256
FIRST_N_TIMEPOINTS = 30000
SELECTED_CHANNELS  = [1, 2, 3, 4, 11, 12, 7, 8]   # same as pipeline

CHANNEL_NAMES = [
    "Fp1-F3", "F3-C3", "C3-P3", "P3-O1",
    "Fp2-F4", "F4-C4", "C4-P4", "P4-O2",
]

SUBBANDS = [
    ("delta", 0.5,  4.0),
    ("theta", 4.0,  8.0),
    ("alpha", 8.0,  13.0),
    ("beta",  13.0, 30.0),
    ("gamma", 30.0, 50.0),
]

# Visual style
COLORS = {
    "interictal":  "#a6cee3",   # pastel light blue
    "preictal":   "#f28e2b",   # softer orange 
    "ictal":      "#7b3294",   # deep purple
    "bg":          "#ffffff", #"#0d1117",
    "panel":       "#ffffff", #"#161b22",
    "grid":       "#21262d",
    "text":       "#c9d1d9",
    "subtext":    "#8b949e",
    "axis":       "#30363d",
}

# ============================================================
# Signal processing — identical to your pipeline
# ============================================================

def bipolar_montage(data):
    """(channels, samples) → (channels-1, samples)"""
    return data[1:, :] - data[:-1, :]


def _bp(data, lo, hi, fs, order=4):
    nyq = fs / 2.0
    b, a = butter(order, [max(lo / nyq, 1e-4), min(hi / nyq, 1 - 1e-4)], btype="band")
    return filtfilt(b, a, data, axis=1)


def load_recording(file_path):
    """Load & preprocess EDF exactly as in the training pipeline."""
    try:
        edf   = pyedflib.EdfReader(str(file_path))
        sfreq = edf.getSampleFrequency(0)
        data  = np.array([edf.readSignal(i) for i in range(edf.signals_in_file)])
        edf.close()
    except Exception as e:
        raise RuntimeError(f"Cannot load {file_path}: {e}")

    if sfreq != _FS:
        data = resample(data, int(data.shape[1] * _FS / sfreq), axis=1)

    b, a = butter(4, [0.5, 50], fs=_FS, btype="band")
    data = filtfilt(b, a, data, axis=1)

    data = bipolar_montage(data)
    data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
    data = data[:, :FIRST_N_TIMEPOINTS]
    data = data[SELECTED_CHANNELS, :]
    return data   # (8, FIRST_N_TIMEPOINTS)


def load_annotations(annotations_path, rec_number):
    """
    Load the per-sample seizure labels for a recording.
    rec_number: 1-indexed (matches EDF filename, e.g. eeg1.edf → rec_number=1)
    Returns binary array of length FIRST_N_TIMEPOINTS.
    """
    ann_df  = pd.read_csv(annotations_path, header=None)
    col_idx = rec_number - 1
    if col_idx >= ann_df.shape[1]:
        raise ValueError(f"Recording {rec_number} not found in annotation CSV "
                         f"(only {ann_df.shape[1]} columns).")
    labels = ann_df[col_idx].values[:FIRST_N_TIMEPOINTS].astype(float)
    return labels


# ============================================================
# Phase detection from sample-level labels
# ============================================================

def detect_phases(labels, fs=_FS, preictal_sec=8):
    """
    From a binary sample-level annotation array build three arrays:
      ictal_mask, preictal_mask, interictal_mask  (same length as labels)

    Pre-ictal = `preictal_sec` seconds immediately before each seizure onset.
    """
    n = len(labels)
    ictal      = labels.astype(bool)
    preictal   = np.zeros(n, dtype=bool)
    pre_samp   = int(preictal_sec * fs)

    # Find seizure onset indices (0→1 transitions)
    diff = np.diff(ictal.astype(int), prepend=0)
    onsets = np.where(diff == 1)[0]

    for onset in onsets:
        start = max(0, onset - pre_samp)
        preictal[start:onset] = True

    # Remove any preictal overlap with ictal (shouldn't happen, but safety)
    preictal &= ~ictal
    interictal = ~ictal & ~preictal

    return ictal, preictal, interictal


# ============================================================
# Compute band-power envelope (for the overview panel)
# ============================================================

def band_power_envelope(data, band="delta", fs=_FS, win_sec=1.0):
    """
    Mean band power across all channels, smoothed with a sliding window.
    Returns (time_sec, power) arrays.
    """
    lo, hi = {b[0]: (b[1], b[2]) for b in SUBBANDS}[band]
    filtered = _bp(data, lo, hi, fs)              # (8, T)
    power    = (filtered ** 2).mean(axis=0)       # (T,)
    win      = int(win_sec * fs)
    # cumsum-based fast sliding mean
    cs  = np.cumsum(np.insert(power, 0, 0))
    smoothed = (cs[win:] - cs[:-win]) / win
    t = np.arange(len(smoothed)) / fs + win_sec / 2
    return t, np.log1p(smoothed)


# ============================================================
# Main plot
# ============================================================

def run_visualization(data_path, rec_number, preictal_sec=8,
                      view_start_sec=None, view_dur_sec=None,
                      save_path=None):
    """
    Parameters
    ----------
    data_path     : str  – same as DATA_PATH in your pipeline
    rec_number    : int  – 1-indexed recording number (matches eeg{N}.edf)
    preictal_sec  : int  – seconds before seizure onset to label as pre-ictal
    view_start_sec: float|None  – if None, auto-zooms to the seizure context
    view_dur_sec  : float|None  – seconds of signal to display (default 60 s)
    save_path     : str|None    – if given, saves PNG here
    """
    data_path = Path(data_path)
    ann_path  = data_path / "annotations_2017_A_fixed.csv"

    # ── find EDF ─────────────────────────────────────────────────────────────
    edf_candidates = sorted(
        data_path.glob(f"eeg{rec_number}.edf"),
    )
    if not edf_candidates:
        # Try zero-padded variants
        edf_candidates = list(data_path.glob(f"eeg0*{rec_number}.edf"))
    if not edf_candidates:
        raise FileNotFoundError(
            f"Cannot find eeg{rec_number}.edf in {data_path}\n"
            f"Files present: {[p.name for p in sorted(data_path.glob('*.edf'))[:10]]}"
        )
    edf_path = edf_candidates[0]
    print(f"Loading: {edf_path.name}")

    # ── load data & labels ───────────────────────────────────────────────────
    data   = load_recording(edf_path)          # (8, T)
    labels = load_annotations(ann_path, rec_number)
    n_samp = data.shape[1]
    time   = np.arange(n_samp) / _FS          # seconds

    ictal, preictal, interictal = detect_phases(labels, _FS, preictal_sec)

    seizure_present = ictal.any()
    if not seizure_present:
        print(f"  ⚠  Recording {rec_number} has NO seizure annotations. "
              f"Showing full recording with interictal label.")

    # ── auto-zoom window ─────────────────────────────────────────────────────
    if view_dur_sec is None:
        view_dur_sec = 60#120.0   # wider default: ~1 min interictal + pre-ictal + ictal + recovery

    if view_start_sec is None and seizure_present:
        onset_sec = np.where(ictal)[0][0] / _FS
        # Start far enough back to show a clear interictal stretch before pre-ictal
        interictal_context = preictal_sec + 40   # 40 s of interictal before pre-ictal window
        view_start_sec = max(0.0, onset_sec - interictal_context)
    elif view_start_sec is None:
        view_start_sec = 0.0

    view_end_sec = min(view_start_sec + view_dur_sec, time[-1])

    i0 = int(view_start_sec * _FS)
    i1 = int(view_end_sec   * _FS)
    t_view  = time[i0:i1]
    d_view  = data[:, i0:i1]
    ic_view = ictal[i0:i1]
    pr_view = preictal[i0:i1]
    ii_view = interictal[i0:i1]

    # ── band-power overview ──────────────────────────────────────────────────
    bp_t, bp_v = band_power_envelope(data, band="delta")

    # ── layout ───────────────────────────────────────────────────────────────
    n_ch = len(CHANNEL_NAMES)
    fig  = plt.figure(figsize=(18, 3 + n_ch * 0.95), facecolor=COLORS["bg"])
    fig.suptitle(
        f"EEG Recording {rec_number}  ·  eeg{rec_number}.edf  ·  "
        f"Pre-ictal window = {preictal_sec} s",
        color=COLORS["text"], fontsize=13, fontweight="bold", y=0.99,
        fontfamily="monospace",
    )

    gs = GridSpec(
        n_ch + 2, 1,
        figure=fig,
        hspace=0.0,
        top=0.96, bottom=0.06, left=0.09, right=0.98,
        height_ratios=[1.8, 0.5] + [1.0] * n_ch,
    )

    # ── Panel 0: band-power overview (full recording) ────────────────────────
    ax_bp = fig.add_subplot(gs[0])
    ax_bp.set_facecolor(COLORS["panel"])
    for spine in ax_bp.spines.values():
        spine.set_edgecolor(COLORS["axis"])

    # shade phases on full timeline
    full_time = np.arange(len(labels)) / _FS
    _shade_phases(ax_bp, full_time, ictal, preictal,
                  ymin=0, ymax=1, transform="axes")
    ax_bp.plot(bp_t, bp_v, color="#60aaff", lw=0.8, alpha=0.9)
    ax_bp.set_xlim(0, time[-1])
    ax_bp.set_ylabel("δ power\n(log)", color=COLORS["subtext"],
                     fontsize=8, fontfamily="monospace")
    ax_bp.set_yticks([])
    ax_bp.tick_params(labelbottom=False, bottom=False, colors=COLORS["subtext"])
    # view-window marker
    ax_bp.axvspan(view_start_sec, view_end_sec,
                  color="white", alpha=0.07, zorder=5)
    ax_bp.text(0.01, 0.88, "OVERVIEW (full recording)",
               transform=ax_bp.transAxes, color=COLORS["subtext"],
               fontsize=8, fontfamily="monospace", va="top")

    # ── Panel 1: thin phase-color bar ────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1])
    ax_bar.set_facecolor(COLORS["bg"])
    for spine in ax_bar.spines.values():
        spine.set_visible(False)
    ax_bar.set_xlim(view_start_sec, view_end_sec)
    ax_bar.set_ylim(0, 1)
    ax_bar.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    _shade_phases_xy(ax_bar, t_view, ic_view, pr_view, ymin=0, ymax=1)

    # ── Panels 2…: EEG channels ──────────────────────────────────────────────
    GAIN = 2.5   # amplitude scale (z-scored signal, ±2 units → ±2.5 * CH_HEIGHT)
    axes_ch = []
    for ch in range(n_ch):
        ax = fig.add_subplot(gs[ch + 2])
        ax.set_facecolor(COLORS["panel"])
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["axis"])
        axes_ch.append(ax)

        sig = d_view[ch]

        # Phase-coloured shading behind trace
        _shade_phases_xy(ax, t_view, ic_view, pr_view,
                         ymin=-1, ymax=1, alpha_scale=0.25)

        # Colour trace by phase
        _plot_coloured_trace(ax, t_view, sig, ic_view, pr_view, ii_view)

        # Zero line
        ax.axhline(0, color=COLORS["grid"], lw=0.5, zorder=1)

        ax.set_xlim(view_start_sec, view_end_sec)
        ax.set_ylim(-GAIN, GAIN)
        ax.set_yticks([])
        ax.set_ylabel(CHANNEL_NAMES[ch], color=COLORS["text"],
                      fontsize=8, fontfamily="monospace",
                      rotation=0, labelpad=52, va="center")

        if ch < n_ch - 1:
            ax.tick_params(labelbottom=False, bottom=False, colors=COLORS["subtext"])
        else:
            ax.tick_params(colors=COLORS["subtext"])
            ax.set_xlabel("Time (s)", color=COLORS["subtext"],
                          fontsize=9, fontfamily="monospace")
            for label in ax.get_xticklabels():
                label.set_fontfamily("monospace")
                label.set_fontsize(8)
                label.set_color(COLORS["subtext"])

    # ── Phase onset vertical lines ────────────────────────────────────────────
    diff = np.diff(ictal.astype(int), prepend=0)
    for onset_s in np.where(diff == 1)[0] / _FS:
        if view_start_sec <= onset_s <= view_end_sec:
            for ax in axes_ch + [ax_bp]:
                ax.axvline(onset_s, color=COLORS["ictal"],
                           lw=1.2, ls="--", alpha=0.8, zorder=10)
            axes_ch[0].text(onset_s + 0.2, GAIN * 0.78, "SEIZURE ONSET",
                            color=COLORS["ictal"], fontsize=8,
                            fontfamily="monospace", fontweight="bold")

    diff2 = np.diff(preictal.astype(int), prepend=0)
    for pre_s in np.where(diff2 == 1)[0] / _FS:
        if view_start_sec <= pre_s <= view_end_sec:
            for ax in axes_ch:
                ax.axvline(pre_s, color=COLORS["preictal"],
                           lw=1.0, ls=":", alpha=0.7, zorder=10)
            axes_ch[0].text(pre_s + 0.2, GAIN * 0.78, f"PRE-ICTAL ({preictal_sec}s)",
                            color=COLORS["preictal"], fontsize=8,
                            fontfamily="monospace")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=COLORS["interictal"], alpha=0.7, label="Interictal"),
        mpatches.Patch(color=COLORS["preictal"],   alpha=0.7, label=f"Pre-ictal (−{preictal_sec}s)"),
        mpatches.Patch(color=COLORS["ictal"],      alpha=0.7, label="Ictal / Seizure"),
    ]
    axes_ch[0].legend(
        handles=legend_patches, loc="upper right",
        framealpha=0.25, facecolor=COLORS["bg"],
        edgecolor=COLORS["axis"], fontsize=8,
        prop={"family": "monospace"},
    )

    if save_path:
        plt.savefig(save_path, dpi=180, bbox_inches="tight",
                    facecolor=COLORS["bg"])
        print(f"Saved: {save_path}")

    plt.show()
    return fig


# ============================================================
# Helpers
# ============================================================

def _shade_phases(ax, time, ictal, preictal, ymin=0, ymax=1, transform="data", alpha=0.18):
    """Shade ictal/preictal regions using data-space y coords."""
    _fill_mask(ax, time, ictal,    COLORS["ictal"],    alpha, ymin, ymax)
    _fill_mask(ax, time, preictal, COLORS["preictal"], alpha, ymin, ymax)


def _shade_phases_xy(ax, time, ictal, preictal, ymin=-1, ymax=1,
                     alpha_scale=1.0):
    _fill_mask(ax, time, ictal,    COLORS["ictal"],    0.18 * alpha_scale, ymin, ymax)
    _fill_mask(ax, time, preictal, COLORS["preictal"], 0.18 * alpha_scale, ymin, ymax)


def _fill_mask(ax, time, mask, color, alpha, ymin, ymax):
    """Fill contiguous True-regions of `mask` with a coloured span."""
    if not mask.any():
        return
    diff   = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(diff ==  1)[0]
    ends   = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        s = min(s, len(time) - 1)
        e = min(e, len(time) - 1)
        ax.axvspan(time[s], time[e], ymin=0, ymax=1,
                   color=color, alpha=alpha, zorder=2)


def _plot_coloured_trace(ax, time, sig, ictal, preictal, interictal):
    """Plot EEG trace in three colours according to phase."""
    for mask, color in [
        (interictal, COLORS["interictal"]),
        (preictal,   COLORS["preictal"]),
        (ictal,      COLORS["ictal"]),
    ]:
        if not mask.any():
            continue
        # Plot contiguous segments (avoids ugly cross-phase lines)
        diff   = np.diff(mask.astype(int), prepend=0, append=0)
        starts = np.where(diff ==  1)[0]
        ends   = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            e = min(e + 1, len(time))
            ax.plot(time[s:e], sig[s:e],
                    color=color, lw=0.7, alpha=0.9, zorder=3)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    # ── Edit these three lines ────────────────────────────────────────────────
    DATA_PATH    = "/content/drive/MyDrive/eeg_cache"
    REC_NUMBER   = 3        # which recording to plot (1-indexed)
    PREICTAL_SEC = 8        # seconds before seizure onset to mark as pre-ictal
    # ─────────────────────────────────────────────────────────────────────────

    save_to = Path(DATA_PATH) / f"seizure_viz_rec{REC_NUMBER}.png"
    run_visualization(
        DATA_PATH,
        rec_number   = REC_NUMBER,
        preictal_sec = PREICTAL_SEC,
        #view_dur_sec=180, 
        view_start_sec=10,
        save_path    = str(save_to),
    )
