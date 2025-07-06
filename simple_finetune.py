#!/usr/bin/env python3
"""
Simplified fine-tuning script for Toto that uses the forecaster interface.
This avoids internal model complexity and focuses on the forecasting task.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set up environment
project_root = os.path.dirname(os.path.abspath(__file__))
toto_path = os.path.join(project_root, "toto")
sys.path.insert(0, project_root)
sys.path.insert(0, toto_path)

os.environ["PYTHONPATH"] = f"{project_root}:{toto_path}:{os.environ.get('PYTHONPATH', '')}"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from toto.data.util.dataset import MaskedTimeseries
from toto.model.toto import Toto
from toto.inference.forecaster import TotoForecaster


class VeniceDataset(Dataset):
    """Dataset for Venice water level time series."""
    
    def __init__(self, df, context_length=2048, prediction_length=96, stride=24):
        self.df = df
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.total_length = context_length + prediction_length
        
        # Calculate valid starting positions
        self.valid_starts = []
        for i in range(0, len(df) - self.total_length + 1, stride):
            self.valid_starts.append(i)
        
        print(f"Created dataset with {len(self.valid_starts)} samples")
        
    def __len__(self):
        return len(self.valid_starts)
    
    def __getitem__(self, idx):
        start_idx = self.valid_starts[idx]
        end_idx = start_idx + self.total_length
        
        # Get the window of data
        window = self.df.iloc[start_idx:end_idx]
        
        # Split into context and target
        context = window.iloc[:self.context_length]
        target = window.iloc[self.context_length:]
        
        # Return raw data - we'll process it in the training loop
        return {
            'context_values': context['Livello Punta Salute (cm)'].values.astype(np.float32),
            'context_timestamps': context['timestamp_seconds'].values.astype(np.int64),
            'target_values': target['Livello Punta Salute (cm)'].values.astype(np.float32)
        }


def train_step(model, forecaster, batch, device, prediction_length):
    """Single training step using the forecaster interface."""
    
    # Prepare batch data
    batch_size = len(batch['context_values'])
    context_values = torch.stack([torch.tensor(v) for v in batch['context_values']]).to(device)
    context_timestamps = torch.stack([torch.tensor(t) for t in batch['context_timestamps']]).to(device)
    target_values = torch.stack([torch.tensor(t) for t in batch['target_values']]).to(device)
    
    # Add variate dimension
    context_values = context_values.unsqueeze(1)  # (batch, 1, context_length)
    context_timestamps = context_timestamps.unsqueeze(1)  # (batch, 1, context_length)
    target_values = target_values.unsqueeze(1)  # (batch, 1, prediction_length)
    
    # Create MaskedTimeseries for context
    context = MaskedTimeseries(
        series=context_values,
        padding_mask=torch.ones_like(context_values, dtype=torch.bool),
        id_mask=torch.zeros_like(context_values, dtype=torch.int),
        timestamp_seconds=context_timestamps,
        time_interval_seconds=torch.full((batch_size, 1), 3600, dtype=torch.int).to(device),
    )
    
    # Use forecaster to generate predictions
    # Note: We'll use mean predictions for loss computation
    forecast = forecaster.forecast(
        context,
        prediction_length=prediction_length,
        num_samples=1,  # Single sample for training efficiency
        samples_per_batch=1,
        use_kv_cache=False  # Disable KV cache for training
    )
    
    # Get mean predictions
    predictions = forecast.mean  # Should be (batch, 1, prediction_length)
    
    # Compute MSE loss
    loss = nn.functional.mse_loss(predictions, target_values)
    
    return loss


def main():
    parser = argparse.ArgumentParser(description='Simple fine-tuning for Toto')
    parser.add_argument('--csv-file', default='toto/datasets/venezia.csv', help='Path to CSV file')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--context-length', type=int, default=1024, help='Context length')
    parser.add_argument('--prediction-length', type=int, default=96, help='Prediction length')
    parser.add_argument('--save-path', default='toto_venezia_simple_finetuned.pt', help='Path to save model')
    
    args = parser.parse_args()
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: CUDA not available. Training on CPU will be very slow!")
        return
    
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.csv_file}...")
    df = pd.read_csv(args.csv_file)
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values('Data')
    df['timestamp_seconds'] = (df['Data'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    
    # Use only recent data for faster training
    df = df.tail(50000)  # Last ~5 years of hourly data
    
    # Split data
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    # Create datasets
    train_dataset = VeniceDataset(
        train_df, 
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        stride=args.prediction_length // 2  # 50% overlap
    )
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    # Load model
    print("Loading pre-trained Toto model...")
    model = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
    model.to(device)
    
    # Create forecaster
    forecaster = TotoForecaster(model.model)
    
    # Optimizer - only optimize the last few layers
    trainable_params = []
    for name, param in model.named_parameters():
        if 'layers.10' in name or 'layers.11' in name or 'output' in name:
            param.requires_grad = True
            trainable_params.append(param)
            print(f"Training parameter: {name}")
        else:
            param.requires_grad = False
    
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    
    # Training history
    train_losses = []
    
    # Training loop
    print(f"\nStarting fine-tuning for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            try:
                # Forward pass and compute loss
                loss = train_step(model, forecaster, batch, device, args.prediction_length)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                
                # Track loss
                epoch_losses.append(loss.item())
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
        
        # Calculate average epoch loss
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")
    
    # Save model
    print(f"\nSaving fine-tuned model to {args.save_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': args.epochs,
        'train_losses': train_losses,
        'args': args,
    }, args.save_path)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Fine-tuning Loss History')
    plt.legend()
    plt.grid(True)
    plt.savefig('simple_finetune_loss_history.png')
    print("Loss history saved as 'simple_finetune_loss_history.png'")
    
    print("\nFine-tuning complete!")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    
    print("\nTo test the fine-tuned model, use:")
    print(f"python run_csv_test.py {args.csv_file} --checkpoint {args.save_path}")


if __name__ == "__main__":
    main()