# DIRU vs LSTM vs Tractable Dendritic RNN - FIXED VERSION
# Fixed bugs: DIRU cell compartment sizing, gradient clipping, variable names

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy import stats

# ============================
# Global config
# ============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 80
BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_SEEDS = 10
SEEDS = list(range(NUM_SEEDS))

# ============================
# Chaotic systems generators
# ============================

def lorenz_system(state, sigma=10.0, rho=28.0, beta=8/3, dt=0.01):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([x + dx*dt, y + dy*dt, z + dz*dt])


def generate_lorenz(T=2000, dt=0.01):
    state = np.random.rand(3)
    traj = []
    for _ in range(T):
        state = lorenz_system(state, dt=dt)
        traj.append(state.copy())
    return np.array(traj)


# ============================
# Dataset
# ============================
    
class LorenzDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=50, prediction_horizon=5):
        self.seq_len = seq_len
        self.pred_h = prediction_horizon
        self.data = []
        self.targets = []

        traj = generate_lorenz(T=5000)
        for _ in range(num_samples):
            i = np.random.randint(0, len(traj)-seq_len-prediction_horizon)
            x = traj[i:i+seq_len]
            y = traj[i+seq_len+prediction_horizon-1]
            self.data.append(x)
            self.targets.append(y)

        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# ============================
# Models - FIXED VERSIONS
# ============================

# ============ ORIGINAL DIRU (from your working 2-way code) ============
class DIRUCellSimple(nn.Module):
    """
    Original simple DIRU from your working code.
    This version works correctly!
    """
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
    """Using the simple, working DIRU cell"""
    def __init__(self, input_size, hidden_size, output_size, num_compartments=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = DIRUCellSimple(input_size, hidden_size, num_compartments)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t], h)
        return self.fc(h)


# ============ DIRU-Detailed (with explicit features) ============
class DIRUCellDetailed(nn.Module):
    """
    DIRU with explicit architectural features for comparison.
    Fixed version with correct compartment sizing.
    
    Key features:
    1. Active per-compartment gating (input+state dependent)
    2. Global neuromodulation
    3. Residual connections
    """
    def __init__(self, input_size, hidden_size, num_compartments=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_comp = num_compartments
        # FIXED: Correct compartment size
        self.comp_size = hidden_size // num_compartments
        
        assert hidden_size % num_compartments == 0, "hidden_size must be divisible by num_compartments"
        
        # Per-compartment computation
        self.W_in = nn.ModuleList([
            nn.Linear(input_size, self.comp_size) for _ in range(num_compartments)
        ])
        self.W_rec = nn.ModuleList([
            nn.Linear(hidden_size, self.comp_size) for _ in range(num_compartments)
        ])
        
        # FEATURE 1: ACTIVE GATING (input + state dependent)
        self.comp_gates = nn.ModuleList([
            nn.Linear(input_size + hidden_size, self.comp_size) for _ in range(num_compartments)
        ])
        
        # Integration layer
        self.integration = nn.Linear(hidden_size, hidden_size)
        
        # FEATURE 2: GLOBAL NEUROMODULATION
        self.global_gate = nn.Linear(input_size + hidden_size, hidden_size)
        
        # FEATURE 3: Output projection for residual
        self.output_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        gate_input = torch.cat([x, h], dim=1)
        
        # Compute each compartment with ACTIVE GATING
        comp_outputs = []
        for i in range(self.num_comp):
            # Local computation
            local_out = torch.tanh(self.W_in[i](x) + self.W_rec[i](h))
            
            # ACTIVE GATE: context-dependent modulation
            gate = torch.sigmoid(self.comp_gates[i](gate_input))
            
            # Gated output
            comp_outputs.append(local_out * gate)
        
        # Integrate compartments
        combined = torch.cat(comp_outputs, dim=1)
        integrated = torch.tanh(self.integration(combined))
        
        # GLOBAL MODULATION: neuromodulatory-like control
        global_mod = torch.sigmoid(self.global_gate(gate_input))
        modulated_state = integrated * global_mod
        
        # RESIDUAL RECURRENCE
        h_new = self.output_proj(modulated_state) + h
        h_new = torch.tanh(h_new)
        
        return h_new


class DIRUDetailed(nn.Module):
    """DIRU with detailed features exposed"""
    def __init__(self, input_size, hidden_size, output_size, num_compartments=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = DIRUCellDetailed(input_size, hidden_size, num_compartments)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t], h)
        return self.fc(h)


# ============ Tractable Dendritic RNN (Brenner et al.) ============
class TractableDendriticCell(nn.Module):
    """
    Brenner et al. style Tractable Dendritic RNN:
    - PASSIVE DENDRITES: Static nonlinear compartments (no gating)
    - NO GLOBAL MODULATION
    - NO RESIDUAL
    """
    def __init__(self, input_size, hidden_size, num_compartments=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_comp = num_compartments
        self.comp_size = hidden_size // num_compartments
        
        assert hidden_size % num_compartments == 0, "hidden_size must be divisible by num_compartments"
        
        # Per-compartment computation (PASSIVE - no gating)
        self.W_in = nn.ModuleList([
            nn.Linear(input_size, self.comp_size) for _ in range(num_compartments)
        ])
        self.W_rec = nn.ModuleList([
            nn.Linear(hidden_size, self.comp_size) for _ in range(num_compartments)
        ])
        
        # Simple integration (no global modulation)
        self.integration = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        # PASSIVE DENDRITES: Static nonlinear compartments
        comp_outputs = []
        for i in range(self.num_comp):
            # Fixed nonlinearity, no adaptive gating
            local_out = torch.tanh(self.W_in[i](x) + self.W_rec[i](h))
            comp_outputs.append(local_out)
        
        # Simple summation (no global modulation)
        combined = torch.cat(comp_outputs, dim=1)
        h_new = torch.tanh(self.integration(combined))
        
        # NO RESIDUAL CONNECTION
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
# Training - WITH GRADIENT CLIPPING
# ============================

def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-3, device="cpu", 
                clip_grad=True, max_grad_norm=1.0):
    """
    Training with optional gradient clipping for stability.
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    hist = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        model.train()
        tl = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"  WARNING: NaN loss detected at epoch {epoch}")
                break
            
            loss.backward()
            
            # GRADIENT CLIPPING for stability
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            opt.step()
            tl += loss.item()

        tl /= len(train_loader)

        model.eval()
        vl = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = loss_fn(pred, y)
                vl += loss.item()
        vl /= len(val_loader)

        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)

    return hist


# ============================
# Multi-seed experiment (3-way comparison)
# ============================

def run_threeway_comparison(dataset_name, train_dataset_fn, val_dataset_fn,
                           input_size, hidden_size=64, num_epochs=80,
                           batch_size=32, device='cpu', seeds=SEEDS,
                           use_detailed_diru=False):

    all_diru_hist = []
    all_tractable_hist = []
    all_lstm_hist = []
    final_metrics = []

    print(f"\n{'='*80}")
    print(f"Running {dataset_name} - {len(seeds)} seeds")
    print(f"DIRU variant: {'Detailed (with explicit features)' if use_detailed_diru else 'Simple (working version)'}")
    print(f"{'='*80}")

    for idx, seed in enumerate(seeds):
        print(f"\nSeed {idx+1}/{len(seeds)} (seed={seed})")
        
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_dataset = train_dataset_fn()
        val_dataset = val_dataset_fn()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize models - choose DIRU variant
        if use_detailed_diru:
            diru = DIRUDetailed(input_size, hidden_size, output_size=input_size, 
                              num_compartments=4).to(device)
        else:
            diru = DIRU(input_size, hidden_size, output_size=input_size, 
                       num_compartments=4).to(device)
        
        tractable = TractableDendriticRNN(input_size, hidden_size, output_size=input_size, 
                                         num_compartments=4).to(device)
        lstm = LSTMModel(input_size, hidden_size, output_size=input_size).to(device)

        # Train all three with gradient clipping
        print("  Training DIRU...")
        diru_hist = train_model(diru, train_loader, val_loader, num_epochs=num_epochs, 
                               device=device, clip_grad=True)
        
        print("  Training Tractable Dendritic RNN...")
        tractable_hist = train_model(tractable, train_loader, val_loader, num_epochs=num_epochs, 
                                    device=device, clip_grad=True)
        
        print("  Training LSTM...")
        lstm_hist = train_model(lstm, train_loader, val_loader, num_epochs=num_epochs, 
                               device=device, clip_grad=True)

        all_diru_hist.append(diru_hist)
        all_tractable_hist.append(tractable_hist)
        all_lstm_hist.append(lstm_hist)

        d_best = min(diru_hist['val_loss'])
        t_best = min(tractable_hist['val_loss'])
        l_best = min(lstm_hist['val_loss'])
        
        improvement_vs_lstm = ((l_best - d_best) / l_best) * 100
        improvement_vs_tractable = ((t_best - d_best) / t_best) * 100

        final_metrics.append({
            "diru_best": d_best,
            "tractable_best": t_best,
            "lstm_best": l_best,
            "improvement_vs_lstm": improvement_vs_lstm,
            "improvement_vs_tractable": improvement_vs_tractable
        })
        
        print(f"  DIRU: {d_best:.6f} | Tractable: {t_best:.6f} | LSTM: {l_best:.6f}")

    def stack(hist_list, key):
        return np.stack([h[key] for h in hist_list], axis=0)

    results = {
        "diru_train_mean": stack(all_diru_hist, 'train_loss').mean(axis=0),
        "diru_train_std":  stack(all_diru_hist, 'train_loss').std(axis=0),
        "diru_val_mean":   stack(all_diru_hist, 'val_loss').mean(axis=0),
        "diru_val_std":    stack(all_diru_hist, 'val_loss').std(axis=0),

        "tractable_train_mean": stack(all_tractable_hist, 'train_loss').mean(axis=0),
        "tractable_train_std":  stack(all_tractable_hist, 'train_loss').std(axis=0),
        "tractable_val_mean":   stack(all_tractable_hist, 'val_loss').mean(axis=0),
        "tractable_val_std":    stack(all_tractable_hist, 'val_loss').std(axis=0),

        "lstm_train_mean": stack(all_lstm_hist, 'train_loss').mean(axis=0),
        "lstm_train_std":  stack(all_lstm_hist, 'train_loss').std(axis=0),
        "lstm_val_mean":   stack(all_lstm_hist, 'val_loss').mean(axis=0),
        "lstm_val_std":    stack(all_lstm_hist, 'val_loss').std(axis=0),

        "final_metrics": final_metrics,
    }

    # Compute statistics
    diru_bests = [m['diru_best'] for m in final_metrics]
    tractable_bests = [m['tractable_best'] for m in final_metrics]
    lstm_bests = [m['lstm_best'] for m in final_metrics]
    
    improvements_vs_lstm = [m['improvement_vs_lstm'] for m in final_metrics]
    improvements_vs_tractable = [m['improvement_vs_tractable'] for m in final_metrics]

    # Statistical significance tests
    _, p_diru_vs_lstm = stats.ttest_rel(diru_bests, lstm_bests)
    _, p_diru_vs_tractable = stats.ttest_rel(diru_bests, tractable_bests)
    _, p_tractable_vs_lstm = stats.ttest_rel(tractable_bests, lstm_bests)

    print(f"\n{'='*80}")
    print(f"FINAL RESULTS - {dataset_name}")
    print(f"{'='*80}")
    print(f"\nBest Validation Loss (mean ± std):")
    print(f"  DIRU :       {np.mean(diru_bests):.6f} ± {np.std(diru_bests):.6f}")
    print(f"  Tractable (Passive Dendrites): {np.mean(tractable_bests):.6f} ± {np.std(tractable_bests):.6f}")
    print(f"  LSTM (Baseline):               {np.mean(lstm_bests):.6f} ± {np.std(lstm_bests):.6f}")
    
    print(f"\nRelative Improvement:")
    print(f"  DIRU vs LSTM:      {np.mean(improvements_vs_lstm):+.2f}% ± {np.std(improvements_vs_lstm):.2f}%")
    print(f"  DIRU vs Tractable: {np.mean(improvements_vs_tractable):+.2f}% ± {np.std(improvements_vs_tractable):.2f}%")
    print(f"  Tractable vs LSTM: {((np.mean(lstm_bests) - np.mean(tractable_bests)) / np.mean(lstm_bests) * 100):+.2f}%")
    
    print(f"\nStatistical Significance (paired t-test):")
    print(f"  DIRU vs LSTM:      p = {p_diru_vs_lstm:.4f} {'***' if p_diru_vs_lstm < 0.001 else '**' if p_diru_vs_lstm < 0.01 else '*' if p_diru_vs_lstm < 0.05 else 'n.s.'}")
    print(f"  DIRU vs Tractable: p = {p_diru_vs_tractable:.4f} {'***' if p_diru_vs_tractable < 0.001 else '**' if p_diru_vs_tractable < 0.01 else '*' if p_diru_vs_tractable < 0.05 else 'n.s.'}")
    print(f"  Tractable vs LSTM: p = {p_tractable_vs_lstm:.4f} {'***' if p_tractable_vs_lstm < 0.001 else '**' if p_tractable_vs_lstm < 0.01 else '*' if p_tractable_vs_lstm < 0.05 else 'n.s.'}")

    results['statistics'] = {
        'p_diru_vs_lstm': p_diru_vs_lstm,
        'p_diru_vs_tractable': p_diru_vs_tractable,
        'p_tractable_vs_lstm': p_tractable_vs_lstm,
        'improvement_vs_lstm_mean': np.mean(improvements_vs_lstm),
        'improvement_vs_lstm_std': np.std(improvements_vs_lstm),
        'improvement_vs_tractable_mean': np.mean(improvements_vs_tractable),
        'improvement_vs_tractable_std': np.std(improvements_vs_tractable),
    }

    return results


# ============================
# Enhanced Plotting - FIXED
# ============================

def visualize_threeway(results, title, save_path=None):
    epochs = np.arange(1, len(results["diru_train_mean"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Training Loss
    axes[0, 0].plot(epochs, results["diru_train_mean"], label="DIRU  ", 
                   color='#2E86AB', linewidth=2)
    axes[0, 0].fill_between(epochs,
                           results["diru_train_mean"]-results["diru_train_std"],
                           results["diru_train_mean"]+results["diru_train_std"], 
                           alpha=0.2, color='#2E86AB')

    axes[0, 0].plot(epochs, results["tractable_train_mean"], label="Tractable   ", 
                   color='#F18F01', linewidth=2)
    axes[0, 0].fill_between(epochs,
                           results["tractable_train_mean"]-results["tractable_train_std"],
                           results["tractable_train_mean"]+results["tractable_train_std"], 
                           alpha=0.2, color='#F18F01')

    axes[0, 0].plot(epochs, results["lstm_train_mean"], label="LSTM", 
                   color='#A23B72', linewidth=2)
    axes[0, 0].fill_between(epochs,
                           results["lstm_train_mean"]-results["lstm_train_std"],
                           results["lstm_train_mean"]+results["lstm_train_std"], 
                           alpha=0.2, color='#A23B72')

    axes[0, 0].set_title("Training Loss (mean±std)", fontsize=13, fontweight='bold')
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("Epoch", fontsize=11)
    axes[0, 0].set_ylabel("MSE Loss (log)", fontsize=11)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)

    # Validation Loss
    axes[0, 1].plot(epochs, results["diru_val_mean"], label="DIRU  ", 
                   color='#2E86AB', linewidth=2)
    axes[0, 1].fill_between(epochs,
                           results["diru_val_mean"]-results["diru_val_std"],
                           results["diru_val_mean"]+results["diru_val_std"], 
                           alpha=0.2, color='#2E86AB')

    axes[0, 1].plot(epochs, results["tractable_val_mean"], label="Tractable   ", 
                   color='#F18F01', linewidth=2)
    axes[0, 1].fill_between(epochs,
                           results["tractable_val_mean"]-results["tractable_val_std"],
                           results["tractable_val_mean"]+results["tractable_val_std"], 
                           alpha=0.2, color='#F18F01')

    axes[0, 1].plot(epochs, results["lstm_val_mean"], label="LSTM", 
                   color='#A23B72', linewidth=2)
    axes[0, 1].fill_between(epochs,
                           results["lstm_val_mean"]-results["lstm_val_std"],
                           results["lstm_val_mean"]+results["lstm_val_std"], 
                           alpha=0.2, color='#A23B72')

    axes[0, 1].set_title("Validation Loss (mean±std)", fontsize=13, fontweight='bold')
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Epoch", fontsize=11)
    axes[0, 1].set_ylabel("MSE Loss (log)", fontsize=11)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3)

    # Performance Comparison (Bar Chart)
    final_metrics = results['final_metrics']
    diru_vals = [m['diru_best'] for m in final_metrics]
    tractable_vals = [m['tractable_best'] for m in final_metrics]
    lstm_vals = [m['lstm_best'] for m in final_metrics]

    x_pos = np.arange(3)
    means = [np.mean(diru_vals), np.mean(tractable_vals), np.mean(lstm_vals)]
    stds = [np.std(diru_vals), np.std(tractable_vals), np.std(lstm_vals)]
    
    bars = axes[1, 0].bar(x_pos, means, yerr=stds, capsize=5,
                         color=['#2E86AB', '#F18F01', '#A23B72'], alpha=0.7)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(['DIRU\n ', 'Tractable\n  ', 'LSTM'], fontsize=10)
    axes[1, 0].set_ylabel('Best Validation Loss', fontsize=11)
    axes[1, 0].set_title('Final Performance Comparison', fontsize=13, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Add significance stars
    if 'statistics' in results:
        y_max = max(means) + max(stds) * 1.2
        
        # DIRU vs LSTM
        p_val = results['statistics']['p_diru_vs_lstm']
        stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        axes[1, 0].plot([0, 2], [y_max, y_max], 'k-', linewidth=1)
        axes[1, 0].text(1, y_max * 1.02, stars, ha='center', fontsize=12, fontweight='bold')
        
        # DIRU vs Tractable
        p_val = results['statistics']['p_diru_vs_tractable']
        stars = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
        axes[1, 0].plot([0, 1], [y_max * 0.9, y_max * 0.9], 'k-', linewidth=1)
        axes[1, 0].text(0.5, y_max * 0.92, stars, ha='center', fontsize=12, fontweight='bold')

    # FIXED: Use correct variable names
    improvements_lstm = [m['improvement_vs_lstm'] for m in final_metrics]
    improvements_tractable = [m['improvement_vs_tractable'] for m in final_metrics]
    
    x_imp = np.arange(2)
    imp_means = [np.mean(improvements_lstm), np.mean(improvements_tractable)]
    imp_stds = [np.std(improvements_lstm), np.std(improvements_tractable)]
    
    axes[1, 1].bar(x_imp, imp_means, yerr=imp_stds, capsize=5,
                  color=['#C73E1D', '#06A77D'], alpha=0.7)
    axes[1, 1].set_xticks(x_imp)
    axes[1, 1].set_xticklabels(['DIRU vs\nLSTM', 'DIRU vs\nTractable'], fontsize=10)
    axes[1, 1].set_ylabel('Improvement (%)', fontsize=11)
    axes[1, 1].set_title('DIRU Advantage', fontsize=13, fontweight='bold')
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].grid(alpha=0.3, axis='y')

    plt.suptitle(title, fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved: {save_path}")

    plt.show()


# ============================
# Main
# ============================

if __name__ == "__main__":
    
    print("="*80)
    print("DIRU vs Tractable Dendritic vs LSTM - FIXED VERSION")
    print("="*80)
    print("\nFixes applied:")
    print("  1. Corrected compartment sizing in DIRU cell")
    print("  2. Added gradient clipping for training stability")
    print("  3. Fixed variable name bug in visualization")
    print("  4. Using simple working DIRU by default")
    print("\nYou can test detailed DIRU by setting use_detailed_diru=True")
    
    results = run_threeway_comparison(
        "Lorenz Attractor (3D Chaos)",
        train_dataset_fn=lambda: LorenzDataset(num_samples=800, seq_len=50, prediction_horizon=5),
        val_dataset_fn=lambda:   LorenzDataset(num_samples=200, seq_len=50, prediction_horizon=5),
        input_size=3,
        hidden_size=HIDDEN_SIZE,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        seeds=SEEDS,
        use_detailed_diru=False  # Use simple working version by default
    )

    visualize_threeway(results, 
                      "Lorenz System: DIRU vs Tractable Dendritic vs LSTM (10 seeds) - FIXED",
                      "lorenz_threeway_fixed.png")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
