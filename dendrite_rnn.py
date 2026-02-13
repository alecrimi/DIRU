"""
Dendrite-Inspired Recurrent Units (DIRU)
A novel RNN architecture inspired by dendritic computation in biological neurons.

Key idea: Each unit has multiple nonlinear sub-compartments (dendrites) that compute
local functions before combining them, unlike traditional RNNs that use flat summation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time
from typing import Tuple, Optional, List


# ============================================================================
# DENDRITE-INSPIRED RECURRENT UNIT (DIRU)
# ============================================================================

class DendriteCompartment(nn.Module):
    """
    A single dendritic compartment that computes a local nonlinear function.
    Each compartment processes a subset of inputs independently.
    """
    def __init__(self, input_size: int, hidden_size: int, compartment_size: int):
        super().__init__()
        self.compartment_size = compartment_size
        
        # Local computation weights
        self.W_x = nn.Linear(input_size, compartment_size, bias=False)
        self.W_h = nn.Linear(hidden_size, compartment_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(compartment_size))
        
        # Local nonlinearity - can be different per compartment
        self.activation = nn.Tanh()
        
        # Compartment-specific gating (inspired by voltage-gated channels)
        self.gate = nn.Linear(input_size + hidden_size, compartment_size)
        
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Compute local dendritic function with gating.
        
        Args:
            x: Input tensor (batch_size, input_size)
            h: Previous hidden state (batch_size, hidden_size)
        
        Returns:
            Compartment output (batch_size, compartment_size)
        """
        # Local computation
        local_input = self.W_x(x) + self.W_h(h) + self.bias
        local_output = self.activation(local_input)
        
        # Gating mechanism (like voltage-gated ion channels)
        gate_input = torch.cat([x, h], dim=1)
        gate_value = torch.sigmoid(self.gate(gate_input))
        
        # Modulated output
        return local_output * gate_value


class DIRUCell(nn.Module):
    """
    Dendrite-Inspired Recurrent Unit Cell.
    
    Features:
    - Multiple dendritic compartments for parallel local computation
    - Inter-compartment gating for dynamic information routing
    - Integration mechanism to combine compartment outputs
    """
    def __init__(self, input_size: int, hidden_size: int, num_compartments: int = 4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_compartments = num_compartments
        
        # Each compartment processes part of the hidden state
        self.compartment_size = hidden_size // num_compartments
        assert hidden_size % num_compartments == 0, "hidden_size must be divisible by num_compartments"
        
        # Create dendritic compartments
        self.compartments = nn.ModuleList([
            DendriteCompartment(input_size, hidden_size, self.compartment_size)
            for _ in range(num_compartments)
        ])
        
        # Inter-compartment integration (inspired by somatic integration)
        self.integration = nn.Linear(hidden_size, hidden_size)
        
        # Global modulation gate (like neuromodulators)
        self.global_gate = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through DIRU cell.
        
        Args:
            x: Input tensor (batch_size, input_size)
            h: Previous hidden state (batch_size, hidden_size) or None
        
        Returns:
            New hidden state (batch_size, hidden_size)
        """
        batch_size = x.size(0)
        
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Compute outputs from all dendritic compartments in parallel
        compartment_outputs = []
        for compartment in self.compartments:
            comp_out = compartment(x, h)
            compartment_outputs.append(comp_out)
        
        # Concatenate compartment outputs
        combined = torch.cat(compartment_outputs, dim=1)
        
        # Somatic integration with nonlinearity
        integrated = torch.tanh(self.integration(combined))
        
        # Global modulation (like neuromodulatory signals)
        gate_input = torch.cat([x, h], dim=1)
        global_mod = torch.sigmoid(self.global_gate(gate_input))
        
        # Modulated state
        modulated_state = integrated * global_mod
        
        # Output projection with residual connection
        h_new = self.output_proj(modulated_state) + h
        h_new = torch.tanh(h_new)
        
        return h_new


class DIRU(nn.Module):
    """
    Dendrite-Inspired Recurrent Unit Network.
    Can be used for sequence-to-sequence or sequence-to-value tasks.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 num_compartments: int = 4, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        # Stack of DIRU cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size
            self.cells.append(DIRUCell(cell_input_size, hidden_size, num_compartments))
        
        # Output projection
        self.fc_out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor, return_sequences: bool = False) -> torch.Tensor:
        """
        Forward pass through DIRU network.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            return_sequences: If True, return outputs for all time steps
        
        Returns:
            Output tensor (batch_size, output_size) or (batch_size, seq_len, output_size)
        """
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden states for each layer
        h_states = [None] * self.num_layers
        
        # Store outputs if needed
        outputs = [] if return_sequences else None
        
        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Pass through layers
            for i, cell in enumerate(self.cells):
                x_t = cell(x_t, h_states[i])
                h_states[i] = x_t
            
            if return_sequences:
                outputs.append(self.fc_out(x_t))
        
        if return_sequences:
            return torch.stack(outputs, dim=1)
        else:
            # Return output from final time step
            return self.fc_out(h_states[-1])


# ============================================================================
# TRADITIONAL LSTM FOR COMPARISON
# ============================================================================

class LSTMModel(nn.Module):
    """Standard LSTM for comparison."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor, return_sequences: bool = False) -> torch.Tensor:
        """
        Forward pass through LSTM.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_size)
            return_sequences: If True, return outputs for all time steps
        
        Returns:
            Output tensor (batch_size, output_size) or (batch_size, seq_len, output_size)
        """
        lstm_out, _ = self.lstm(x)
        
        if return_sequences:
            return self.fc_out(lstm_out)
        else:
            # Take output from last time step
            return self.fc_out(lstm_out[:, -1, :])


# ============================================================================
# TOY TIME SERIES DATASETS
# ============================================================================

class SineWaveDataset(Dataset):
    """
    Multiple superimposed sine waves with different frequencies.
    Tests ability to capture multi-scale temporal patterns.
    """
    def __init__(self, num_samples: int = 1000, seq_len: int = 50, num_features: int = 1):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.num_features = num_features
        
        self.X, self.y = self._generate_data()
        
    def _generate_data(self):
        X = []
        y = []
        
        for _ in range(self.num_samples):
            t = np.linspace(0, 4*np.pi, self.seq_len + 10)
            
            # Multiple frequency components (hierarchical temporal structure)
            freq1 = np.random.uniform(0.5, 2.0)
            freq2 = np.random.uniform(3.0, 5.0)
            freq3 = np.random.uniform(7.0, 10.0)
            
            signal = (np.sin(freq1 * t) + 
                     0.5 * np.sin(freq2 * t) + 
                     0.25 * np.sin(freq3 * t))
            
            # Add noise
            signal += np.random.normal(0, 0.1, len(t))
            
            # Input: first seq_len points
            # Target: predict next 10 points
            X.append(signal[:self.seq_len].reshape(-1, 1))
            y.append(signal[self.seq_len:].mean())  # Predict average of next 10 points
        
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)).unsqueeze(1)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MemoryTaskDataset(Dataset):
    """
    Copy task with variable-length delays.
    Tests long-term memory capabilities.
    """
    def __init__(self, num_samples: int = 1000, seq_len: int = 50, vocab_size: int = 8):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        self.X, self.y = self._generate_data()
        
    def _generate_data(self):
        X = []
        y = []
        
        for _ in range(self.num_samples):
            # Signal length
            signal_len = np.random.randint(3, 8)
            delay_len = self.seq_len - signal_len - 5
            
            # Random signal
            signal = np.random.randint(1, self.vocab_size, signal_len)
            
            # Create sequence: [signal, delay, recall_marker]
            sequence = np.zeros(self.seq_len)
            sequence[:signal_len] = signal
            sequence[signal_len + delay_len] = self.vocab_size  # Recall marker
            
            # Target: sum of signal elements (simple memory task)
            target = signal.sum()
            
            X.append(sequence.reshape(-1, 1))
            y.append(target)
        
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)).unsqueeze(1)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NonlinearDynamicsDataset(Dataset):
    """
    Chaotic system (Lorenz-like) prediction.
    Tests ability to model complex nonlinear dynamics.
    """
    def __init__(self, num_samples: int = 1000, seq_len: int = 50):
        self.num_samples = num_samples
        self.seq_len = seq_len
        
        self.X, self.y = self._generate_data()
        
    def _generate_data(self):
        def lorenz_step(x, y, z, dt=0.01, sigma=10, rho=28, beta=8/3):
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt
            return x + dx, y + dy, z + dz
        
        X = []
        y_out = []
        
        for _ in range(self.num_samples):
            # Random initial conditions
            x, y, z = np.random.randn(3) * 5
            
            trajectory = []
            for _ in range(self.seq_len + 10):
                trajectory.append([x, y, z])
                x, y, z = lorenz_step(x, y, z)
            
            trajectory = np.array(trajectory)
            
            # Use only x-coordinate for simplicity
            X.append(trajectory[:self.seq_len, [0]])
            y_out.append(trajectory[self.seq_len:, 0].mean())
        
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y_out)).unsqueeze(1)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, device='cpu'):
    """Train a model and return training history."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch_time': []
    }
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        epoch_time = time.time() - start_time
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epoch_time'].append(epoch_time)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, Time: {epoch_time:.2f}s")
    
    return history


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_comparison(dataset_name: str, train_dataset, val_dataset, 
                   input_size: int, hidden_size: int, output_size: int,
                   num_epochs: int = 50, batch_size: int = 32, device: str = 'cpu'):
    """
    Run comparison between DIRU and LSTM on a dataset.
    """
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize models
    diru_model = DIRU(input_size, hidden_size, output_size, num_compartments=4, num_layers=1)
    lstm_model = LSTMModel(input_size, hidden_size, output_size, num_layers=1)
    
    print(f"\nModel Parameters:")
    print(f"DIRU: {count_parameters(diru_model):,} parameters")
    print(f"LSTM: {count_parameters(lstm_model):,} parameters")
    
    # Train DIRU
    print(f"\nTraining DIRU...")
    diru_history = train_model(diru_model, train_loader, val_loader, 
                               num_epochs=num_epochs, device=device)
    
    # Train LSTM
    print(f"\nTraining LSTM...")
    lstm_history = train_model(lstm_model, train_loader, val_loader, 
                               num_epochs=num_epochs, device=device)
    
    # Calculate statistics
    diru_final_val = diru_history['val_loss'][-1]
    lstm_final_val = lstm_history['val_loss'][-1]
    diru_avg_time = np.mean(diru_history['epoch_time'])
    lstm_avg_time = np.mean(lstm_history['epoch_time'])
    
    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"{'='*70}")
    print(f"Final Validation Loss:")
    print(f"  DIRU: {diru_final_val:.6f}")
    print(f"  LSTM: {lstm_final_val:.6f}")
    print(f"  Improvement: {((lstm_final_val - diru_final_val) / lstm_final_val * 100):.2f}%")
    print(f"\nAverage Epoch Time:")
    print(f"  DIRU: {diru_avg_time:.3f}s")
    print(f"  LSTM: {lstm_avg_time:.3f}s")
    
    return {
        'diru_history': diru_history,
        'lstm_history': lstm_history,
        'diru_model': diru_model,
        'lstm_model': lstm_model
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comparison(results_dict: dict, dataset_name: str, save_path: str = None):
    """Plot training curves for DIRU vs LSTM."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    diru_history = results_dict['diru_history']
    lstm_history = results_dict['lstm_history']
    
    epochs = range(1, len(diru_history['train_loss']) + 1)
    
    # Training loss
    axes[0].plot(epochs, diru_history['train_loss'], label='DIRU', linewidth=2, color='#2E86AB')
    axes[0].plot(epochs, lstm_history['train_loss'], label='LSTM', linewidth=2, color='#A23B72')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title(f'{dataset_name} - Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Validation loss
    axes[1].plot(epochs, diru_history['val_loss'], label='DIRU', linewidth=2, color='#2E86AB')
    axes[1].plot(epochs, lstm_history['val_loss'], label='LSTM', linewidth=2, color='#A23B72')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Validation Loss', fontsize=12)
    axes[1].set_title(f'{dataset_name} - Validation Loss', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_predictions(model, dataset, num_samples=5, device='cpu'):
    """Visualize model predictions on sample sequences."""
    model.eval()
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    with torch.no_grad():
        for idx, ax in zip(indices, axes):
            x, y_true = dataset[idx]
            x = x.unsqueeze(0).to(device)
            y_pred = model(x).cpu().item()
            y_true = y_true.item()
            
            # Plot sequence
            ax.plot(x.cpu().squeeze().numpy(), linewidth=1.5)
            ax.axhline(y=y_true, color='g', linestyle='--', label=f'True: {y_true:.2f}', linewidth=2)
            ax.axhline(y=y_pred, color='r', linestyle='--', label=f'Pred: {y_pred:.2f}', linewidth=2)
            ax.legend(fontsize=8)
            ax.set_xlabel('Time Step')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Hyperparameters
    HIDDEN_SIZE = 64
    NUM_EPOCHS = 50
    BATCH_SIZE = 32
    
    # ========================================================================
    # EXPERIMENT 1: Multi-Frequency Sine Waves
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 1: Multi-Frequency Sine Wave Prediction")
    print("="*70)
    
    train_sine = SineWaveDataset(num_samples=800, seq_len=50)
    val_sine = SineWaveDataset(num_samples=200, seq_len=50)
    
    sine_results = run_comparison(
        "Multi-Frequency Sine Waves",
        train_sine, val_sine,
        input_size=1, hidden_size=HIDDEN_SIZE, output_size=1,
        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, device=device
    )
    
    # ========================================================================
    # EXPERIMENT 2: Memory Task
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 2: Variable-Delay Memory Task")
    print("="*70)
    
    train_memory = MemoryTaskDataset(num_samples=800, seq_len=50, vocab_size=8)
    val_memory = MemoryTaskDataset(num_samples=200, seq_len=50, vocab_size=8)
    
    memory_results = run_comparison(
        "Variable-Delay Memory Task",
        train_memory, val_memory,
        input_size=1, hidden_size=HIDDEN_SIZE, output_size=1,
        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, device=device
    )
    
    # ========================================================================
    # EXPERIMENT 3: Nonlinear Dynamics (Lorenz System)
    # ========================================================================
    print("\n" + "="*70)
    print("EXPERIMENT 3: Chaotic Dynamics Prediction (Lorenz System)")
    print("="*70)
    
    train_lorenz = NonlinearDynamicsDataset(num_samples=800, seq_len=50)
    val_lorenz = NonlinearDynamicsDataset(num_samples=200, seq_len=50)
    
    lorenz_results = run_comparison(
        "Chaotic Dynamics (Lorenz)",
        train_lorenz, val_lorenz,
        input_size=1, hidden_size=HIDDEN_SIZE, output_size=1,
        num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, device=device
    )
    
    # ========================================================================
    # GENERATE PLOTS
    # ========================================================================
    print("\n" + "="*70)
    print("Generating comparison plots...")
    print("="*70)
    
    fig1 = plot_comparison(sine_results, "Multi-Frequency Sine Waves", 
                          "/home/claude/sine_comparison.png")
    fig2 = plot_comparison(memory_results, "Variable-Delay Memory Task",
                          "/home/claude/memory_comparison.png")
    fig3 = plot_comparison(lorenz_results, "Chaotic Dynamics (Lorenz)",
                          "/home/claude/lorenz_comparison.png")
    
    # Sample predictions
    print("\nGenerating sample predictions...")
    fig4 = plot_predictions(sine_results['diru_model'], val_sine, num_samples=5, device=device)
    plt.savefig("/home/claude/diru_predictions.png", dpi=300, bbox_inches='tight')
    
    print("\n" + "="*70)
    print("All experiments completed!")
    print("="*70)
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    
    experiments = [
        ("Sine Waves", sine_results),
        ("Memory Task", memory_results),
        ("Lorenz System", lorenz_results)
    ]
    
    for name, results in experiments:
        diru_val = results['diru_history']['val_loss'][-1]
        lstm_val = results['lstm_history']['val_loss'][-1]
        improvement = ((lstm_val - diru_val) / lstm_val * 100)
        
        print(f"\n{name}:")
        print(f"  DIRU Final Val Loss: {diru_val:.6f}")
        print(f"  LSTM Final Val Loss: {lstm_val:.6f}")
        print(f"  DIRU Improvement: {improvement:+.2f}%")
