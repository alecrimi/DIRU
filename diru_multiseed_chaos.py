# DIRU vs LSTM Multi-seed Chaotic Dynamics Benchmark
# Full self-contained script
# Alessandro-ready scientific version

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

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
# Models
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
    def __init__(self, input_size, hidden_size, output_size, num_compartments=4, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = DIRUCell(input_size, hidden_size, num_compartments)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            h = self.cell(x[:, t], h)
        out = self.fc(h)
        return out


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
# Training
# ============================

def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-3, device="cpu"):
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
            loss.backward()
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
# Multi-seed experiment
# ============================

def run_chaotic_comparison_multiseed(dataset_name, train_dataset_fn, val_dataset_fn,
                                     input_size, hidden_size=64, num_epochs=80,
                                     batch_size=32, device='cpu', seeds=SEEDS):

    all_diru_hist = []
    all_lstm_hist = []
    final_metrics = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_dataset = train_dataset_fn()
        val_dataset = val_dataset_fn()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        diru = DIRU(input_size, hidden_size, output_size=input_size).to(device)
        lstm = LSTMModel(input_size, hidden_size, output_size=input_size).to(device)

        diru_hist = train_model(diru, train_loader, val_loader, num_epochs=num_epochs, device=device)
        lstm_hist = train_model(lstm, train_loader, val_loader, num_epochs=num_epochs, device=device)

        all_diru_hist.append(diru_hist)
        all_lstm_hist.append(lstm_hist)

        d_best = min(diru_hist['val_loss'])
        l_best = min(lstm_hist['val_loss'])
        improvement = ((l_best - d_best) / l_best) * 100

        final_metrics.append({
            "diru_best": d_best,
            "lstm_best": l_best,
            "improvement": improvement
        })

    def stack(hist_list, key):
        return np.stack([h[key] for h in hist_list], axis=0)

    results = {
        "diru_train_mean": stack(all_diru_hist, 'train_loss').mean(axis=0),
        "diru_train_std":  stack(all_diru_hist, 'train_loss').std(axis=0),
        "diru_val_mean":   stack(all_diru_hist, 'val_loss').mean(axis=0),
        "diru_val_std":    stack(all_diru_hist, 'val_loss').std(axis=0),

        "lstm_train_mean": stack(all_lstm_hist, 'train_loss').mean(axis=0),
        "lstm_train_std":  stack(all_lstm_hist, 'train_loss').std(axis=0),
        "lstm_val_mean":   stack(all_lstm_hist, 'val_loss').mean(axis=0),
        "lstm_val_std":    stack(all_lstm_hist, 'val_loss').std(axis=0),

        "final_metrics": final_metrics,
        "improvement_mean": np.mean([m['improvement'] for m in final_metrics]),
        "improvement_std":  np.std([m['improvement'] for m in final_metrics])
    }

    print("\n=====", dataset_name, "=====")
    print("DIRU best (mean±std):", np.mean([m['diru_best'] for m in final_metrics]), 
          "±", np.std([m['diru_best'] for m in final_metrics]))
    print("LSTM best (mean±std):", np.mean([m['lstm_best'] for m in final_metrics]), 
          "±", np.std([m['lstm_best'] for m in final_metrics]))
    print("Improvement (% mean±std):", results['improvement_mean'], "±", results['improvement_std'])

    return results

# ============================
# Plotting
# ============================

def visualize(results, title, save_path=None):
    epochs = np.arange(1, len(results["diru_train_mean"]) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Train
    ax[0].plot(epochs, results["diru_train_mean"], label="DIRU")
    ax[0].fill_between(epochs,
                       results["diru_train_mean"]-results["diru_train_std"],
                       results["diru_train_mean"]+results["diru_train_std"], alpha=0.2)

    ax[0].plot(epochs, results["lstm_train_mean"], label="LSTM")
    ax[0].fill_between(epochs,
                       results["lstm_train_mean"]-results["lstm_train_std"],
                       results["lstm_train_mean"]+results["lstm_train_std"], alpha=0.2)

    ax[0].set_title("Training Loss (mean±std)")
    ax[0].set_yscale("log")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    # Val
    ax[1].plot(epochs, results["diru_val_mean"], label="DIRU")
    ax[1].fill_between(epochs,
                       results["diru_val_mean"]-results["diru_val_std"],
                       results["diru_val_mean"]+results["diru_val_std"], alpha=0.2)

    ax[1].plot(epochs, results["lstm_val_mean"], label="LSTM")
    ax[1].fill_between(epochs,
                       results["lstm_val_mean"]-results["lstm_val_std"],
                       results["lstm_val_mean"]+results["lstm_val_std"], alpha=0.2)

    ax[1].set_title("Validation Loss (mean±std)")
    ax[1].set_yscale("log")
    ax[1].legend(); ax[1].grid(alpha=0.3)

    plt.suptitle(title, fontweight="bold")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

# ============================
# Main
# ============================

if __name__ == "__main__":

    results = run_chaotic_comparison_multiseed(
        "Lorenz Attractor",
        train_dataset_fn=lambda: LorenzDataset(num_samples=800, seq_len=50, prediction_horizon=5),
        val_dataset_fn=lambda:   LorenzDataset(num_samples=200, seq_len=50, prediction_horizon=5),
        input_size=3,
        hidden_size=HIDDEN_SIZE,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        seeds=SEEDS
    )

    visualize(results, "Lorenz System — DIRU vs LSTM (Multi-seed)", "lorenz_multiseed.png")

    print("\nExperiment complete. Plot saved as lorenz_multiseed.png")
