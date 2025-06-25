#!/usr/bin/env python3
"""
Fine-tuning script for Toto on Venice water level data.
Usage: python finetune_venezia.py --csv-file toto/datasets/venezia.csv --epochs 10
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
from datetime import datetime

from toto.data.util.dataset import MaskedTimeseries
from toto.model.toto import Toto

# Set up environment
project_root = os.path.dirname(os.path.abspath(__file__))
toto_path = os.path.join(project_root, "toto")
sys.path.insert(0, project_root)
sys.path.insert(0, toto_path)

os.environ["PYTHONPATH"] = f"{project_root}:{toto_path}:{os.environ.get('PYTHONPATH', '')}"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

# Fix imports
import sys
sys.path.insert(0, os.path.join(project_root, 'toto'))


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
        
        # Prepare context as MaskedTimeseries
        context_values = torch.tensor(
            context['Livello Punta Salute (cm)'].values, 
            dtype=torch.float32
        ).unsqueeze(0)  # Add variate dimension
        
        context_timestamps = torch.tensor(
            context['timestamp_seconds'].values,
            dtype=torch.int64
        ).unsqueeze(0)
        
        context_input = MaskedTimeseries(
            series=context_values,
            padding_mask=torch.ones_like(context_values, dtype=torch.bool),
            id_mask=torch.zeros_like(context_values, dtype=torch.int),
            timestamp_seconds=context_timestamps,
            time_interval_seconds=torch.tensor([3600], dtype=torch.int),  # 1 hour
        )
        
        # Target values
        target_values = torch.tensor(
            target['Livello Punta Salute (cm)'].values,
            dtype=torch.float32
        ).unsqueeze(0)  # Add variate dimension
        
        return context_input, target_values


def compute_loss(model_output, target, prediction_length, context_length):
    """Compute MSE loss for simplicity during fine-tuning."""
    # The model output is a distribution over the entire sequence (context + prediction)
    # We need to extract predicted values for the prediction window
    
    # Get the mean of the distribution as point predictions
    # This should have shape (batch, variates, context_length + prediction_length)
    if hasattr(model_output, 'mean'):
        predictions = model_output.mean
    else:
        # If it's a StudentT distribution, get the loc parameter
        predictions = model_output.loc
    
    # Extract only the prediction part (last prediction_length timesteps)
    predictions = predictions[..., -prediction_length:]
    
    # Compute MSE loss
    # target shape: (batch, variates, prediction_length)
    # predictions shape: (batch, variates, prediction_length)
    mse_loss = torch.nn.functional.mse_loss(predictions, target)
    
    return mse_loss


def evaluate(model, dataloader, device, context_length):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for context, target in dataloader:
            # Move to device
            context = MaskedTimeseries(
                series=context.series.to(device),
                padding_mask=context.padding_mask.to(device),
                id_mask=context.id_mask.to(device),
                timestamp_seconds=context.timestamp_seconds.to(device),
                time_interval_seconds=context.time_interval_seconds.to(device),
            )
            target = target.to(device)
            
            # Forward pass
            # TotoBackbone expects individual components of MaskedTimeseries
            output = model.model(
                context.series,
                context.padding_mask,
                context.id_mask,
                context.timestamp_seconds,
                context.time_interval_seconds
            )
            loss = compute_loss(output, target, target.shape[-1], context_length)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Toto on Venice data')
    parser.add_argument('--csv-file', default='toto/datasets/venezia.csv', help='Path to CSV file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--context-length', type=int, default=2048, help='Context length')
    parser.add_argument('--prediction-length', type=int, default=96, help='Prediction length')
    parser.add_argument('--train-split', type=float, default=0.8, help='Train split ratio')
    parser.add_argument('--save-path', default='toto_venezia_finetuned.pt', help='Path to save model')
    
    args = parser.parse_args()
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: CUDA not available. Training on CPU will be very slow!")
        args.batch_size = min(args.batch_size, 2)  # Reduce batch size for CPU
    
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.csv_file}...")
    df = pd.read_csv(args.csv_file)
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values('Data')
    df['timestamp_seconds'] = (df['Data'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    
    # Split data
    split_idx = int(len(df) * args.train_split)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    # Create datasets
    train_dataset = VeniceDataset(
        train_df, 
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        stride=args.prediction_length  # Non-overlapping windows
    )
    
    val_dataset = VeniceDataset(
        val_df,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        stride=args.prediction_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 for debugging
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Load model
    print("Loading pre-trained Toto model...")
    model = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
    model.to(device)
    
    # Prepare for fine-tuning
    for param in model.parameters():
        param.requires_grad = True
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs * len(train_loader)
    )
    
    # Training history
    train_losses = []
    val_losses = []
    
    # Training loop
    print(f"\nStarting fine-tuning for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for context, target in pbar:
            # Move to device
            context = MaskedTimeseries(
                series=context.series.to(device),
                padding_mask=context.padding_mask.to(device),
                id_mask=context.id_mask.to(device),
                timestamp_seconds=context.timestamp_seconds.to(device),
                time_interval_seconds=context.time_interval_seconds.to(device),
            )
            target = target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            # TotoBackbone expects individual components of MaskedTimeseries
            output = model.model(
                context.series,
                context.padding_mask,
                context.id_mask,
                context.timestamp_seconds,
                context.time_interval_seconds
            )
            loss = compute_loss(output, target, args.prediction_length, args.context_length)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        avg_val_loss = evaluate(model, val_loader, device, args.context_length)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
    
    # Save model
    print(f"\nSaving fine-tuned model to {args.save_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': args.epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'args': args,
    }, args.save_path)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Fine-tuning Loss History')
    plt.legend()
    plt.grid(True)
    plt.savefig('finetune_loss_history.png')
    print("Loss history saved as 'finetune_loss_history.png'")
    
    print("\nFine-tuning complete!")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")
    
    # Test the fine-tuned model
    print("\nTo test the fine-tuned model, use:")
    print(f"python run_csv_test.py {args.csv_file} --checkpoint {args.save_path}")


if __name__ == "__main__":
    main()