"""
Quick demo script to test DIRU architecture and visualize results.
Run this for a fast comparison on a small dataset.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from dendrite_rnn import (
    DIRU, LSTMModel, SineWaveDataset, 
    train_model, count_parameters
)
from torch.utils.data import DataLoader

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

print("="*70)
print("DENDRITE-INSPIRED RECURRENT UNITS (DIRU) - Quick Demo")
print("="*70)

# Configuration
HIDDEN_SIZE = 32
NUM_EPOCHS = 30
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"\nDevice: {DEVICE}")
print(f"Hidden Size: {HIDDEN_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Batch Size: {BATCH_SIZE}")

# Create small dataset
print("\nGenerating toy dataset (multi-frequency sine waves)...")
train_dataset = SineWaveDataset(num_samples=400, seq_len=40)
val_dataset = SineWaveDataset(num_samples=100, seq_len=40)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize models
print("\nInitializing models...")
diru = DIRU(input_size=1, hidden_size=HIDDEN_SIZE, output_size=1, 
            num_compartments=4, num_layers=1)
lstm = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE, output_size=1, 
                 num_layers=1)

print(f"DIRU parameters: {count_parameters(diru):,}")
print(f"LSTM parameters: {count_parameters(lstm):,}")

# Train DIRU
print("\n" + "-"*70)
print("Training DIRU...")
print("-"*70)
diru_history = train_model(diru, train_loader, val_loader, 
                          num_epochs=NUM_EPOCHS, lr=0.001, device=DEVICE)

# Train LSTM
print("\n" + "-"*70)
print("Training LSTM...")
print("-"*70)
lstm_history = train_model(lstm, train_loader, val_loader, 
                          num_epochs=NUM_EPOCHS, lr=0.001, device=DEVICE)

# Results
print("\n" + "="*70)
print("RESULTS")
print("="*70)

diru_final = diru_history['val_loss'][-1]
lstm_final = lstm_history['val_loss'][-1]
improvement = ((lstm_final - diru_final) / lstm_final) * 100

print(f"\nFinal Validation Loss:")
print(f"  DIRU: {diru_final:.6f}")
print(f"  LSTM: {lstm_final:.6f}")
print(f"  Improvement: {improvement:+.2f}%")

diru_time = np.mean(diru_history['epoch_time'])
lstm_time = np.mean(lstm_history['epoch_time'])
print(f"\nAverage Epoch Time:")
print(f"  DIRU: {diru_time:.3f}s")
print(f"  LSTM: {lstm_time:.3f}s")

# Visualization
print("\nGenerating plots...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

epochs = range(1, NUM_EPOCHS + 1)

# Training loss
axes[0].plot(epochs, diru_history['train_loss'], label='DIRU', 
            linewidth=2, color='#2E86AB', marker='o', markersize=3)
axes[0].plot(epochs, lstm_history['train_loss'], label='LSTM', 
            linewidth=2, color='#A23B72', marker='s', markersize=3)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Training Loss', fontsize=12)
axes[0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Validation loss
axes[1].plot(epochs, diru_history['val_loss'], label='DIRU', 
            linewidth=2, color='#2E86AB', marker='o', markersize=3)
axes[1].plot(epochs, lstm_history['val_loss'], label='LSTM', 
            linewidth=2, color='#A23B72', marker='s', markersize=3)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Validation Loss', fontsize=12)
axes[1].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('quick_demo_results.png', dpi=300, bbox_inches='tight')
print("Plot saved to: quick_demo_results.png")

# Sample predictions
print("\nGenerating sample predictions...")
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
axes2 = axes2.flatten()

diru.eval()
lstm.eval()

indices = np.random.choice(len(val_dataset), 3, replace=False)

with torch.no_grad():
    for i, idx in enumerate(indices):
        x, y_true = val_dataset[idx]
        x_input = x.unsqueeze(0).to(DEVICE)
        
        diru_pred = diru(x_input).cpu().item()
        lstm_pred = lstm(x_input).cpu().item()
        y_true = y_true.item()
        
        # DIRU prediction
        axes2[i].plot(x.numpy(), linewidth=1.5, color='black', alpha=0.7)
        axes2[i].axhline(y=y_true, color='green', linestyle='--', 
                        label=f'True: {y_true:.3f}', linewidth=2)
        axes2[i].axhline(y=diru_pred, color='#2E86AB', linestyle='--', 
                        label=f'DIRU: {diru_pred:.3f}', linewidth=2)
        axes2[i].set_title(f'Sample {i+1} - DIRU', fontweight='bold')
        axes2[i].legend(fontsize=9)
        axes2[i].grid(True, alpha=0.3)
        
        # LSTM prediction
        axes2[i+3].plot(x.numpy(), linewidth=1.5, color='black', alpha=0.7)
        axes2[i+3].axhline(y=y_true, color='green', linestyle='--', 
                          label=f'True: {y_true:.3f}', linewidth=2)
        axes2[i+3].axhline(y=lstm_pred, color='#A23B72', linestyle='--', 
                          label=f'LSTM: {lstm_pred:.3f}', linewidth=2)
        axes2[i+3].set_title(f'Sample {i+1} - LSTM', fontweight='bold')
        axes2[i+3].legend(fontsize=9)
        axes2[i+3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('predictions_comparison.png', dpi=300, bbox_inches='tight')
print("Predictions plot saved to: predictions_comparison.png")

print("\n" + "="*70)
print("Demo completed successfully!")
print("="*70)
print("\nGenerated files:")
print("  - quick_demo_results.png (training curves)")
print("  - predictions_comparison.png (sample predictions)")
print("\nTo run the full comparison with all datasets, execute:")
print("  python dendrite_rnn.py")
