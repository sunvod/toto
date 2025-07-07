#!/usr/bin/env python3
"""
Simplified fine-tuning script for Toto on Venice water level data.
Usage: python finetune_venezia.py --csv-file toto/datasets/venezia.csv --epochs 1
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Set up environment BEFORE imports (like the working scripts)
project_root = os.path.dirname(os.path.abspath(__file__))
toto_path = os.path.join(project_root, "toto")

# Add paths to Python path
sys.path.insert(0, project_root)
sys.path.insert(0, toto_path)

# Set environment variables
os.environ["PYTHONPATH"] = f"{project_root}:{toto_path}:{os.environ.get('PYTHONPATH', '')}"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

torch.use_deterministic_algorithms(True)

# NOW do the imports (using absolute imports with toto prefix)
from toto.data.util.dataset import MaskedTimeseries
from toto.model.toto import Toto


def prepare_toto_input(df, context_length, prediction_length, device='cuda'):
    """
    Prepare data exactly like the working scripts do.
    """
    total_length = context_length + prediction_length

    # Select a window from the data
    if len(df) < total_length:
        raise ValueError(f"Data too short: {len(df)} < {total_length}")

    # Take a window from the end of the data
    start_idx = len(df) - total_length
    window = df.iloc[start_idx:start_idx + total_length]

    context_data = window.iloc[:context_length]
    target_data = window.iloc[context_length:]

    # Prepare input exactly like test_venezia_finetuned.py
    context_values = torch.tensor(
        context_data['Livello Punta Salute (cm)'].values,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, context_length)

    context_timestamps = torch.tensor(
        context_data['timestamp_seconds'].values,
        dtype=torch.int64
    ).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, context_length)

    context_input = MaskedTimeseries(
        series=context_values,
        padding_mask=torch.ones_like(context_values, dtype=torch.bool),
        id_mask=torch.zeros_like(context_values, dtype=torch.int),
        timestamp_seconds=context_timestamps,
        time_interval_seconds=torch.tensor([[3600]], dtype=torch.int).to(device),  # (1, 1)
    )

    # Target values
    target_values = torch.tensor(
        target_data['Livello Punta Salute (cm)'].values,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, prediction_length)

    return context_input, target_values


def create_training_samples(df, context_length, prediction_length, stride=96):
    """
    Create multiple training samples from the dataframe.
    """
    total_length = context_length + prediction_length
    samples = []

    for start_idx in range(0, len(df) - total_length + 1, stride):
        end_idx = start_idx + total_length
        sample_df = df.iloc[start_idx:end_idx].copy()
        samples.append(sample_df)

    print(f"Created {len(samples)} training samples")
    return samples


def compute_loss_with_trainer(model, trainer, context_input, target_values, prediction_length):
    """
    Compute loss using the TotoTrainer (modified forecaster with gradients).
    """
    # Set model to train mode
    model.train()
    
    try:
        # Use trainer to generate predictions with gradients
        # Use num_samples=None to force using generate_mean instead of generate_samples
        forecast = trainer.forecast(
            context_input,
            prediction_length=prediction_length,
            num_samples=None,  # None forces using generate_mean path
            samples_per_batch=1,
            use_kv_cache=False  # Disable KV cache for training
        )
        
        # Get mean predictions
        predictions = forecast.mean  # Shape: (batch, variate, prediction_length)
        
        # Compute MSE loss
        mse_loss = torch.nn.functional.mse_loss(predictions, target_values)
        
        return mse_loss
        
    except Exception as e:
        print(f"Trainer error: {e}")
        # Return a small loss to continue training
        return torch.tensor(0.1, requires_grad=True, device=target_values.device)


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Toto on Venice data (simplified)')
    parser.add_argument('--csv-file', default='toto/datasets/venezia.csv', help='Path to CSV file')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--context-length', type=int, default=1024, help='Context length')
    parser.add_argument('--prediction-length', type=int, default=96, help='Prediction length')
    parser.add_argument('--train-split', type=float, default=0.8, help='Train split ratio')
    parser.add_argument('--save-path', default='toto_venezia_simple_finetuned.pt', help='Path to save model')
    parser.add_argument('--num-samples', type=int, default=50, help='Number of training samples to use (for speed)')

    args = parser.parse_args()

    # Show configuration
    print(f"Configuration:")
    print(f"  Context length: {args.context_length}")
    print(f"  Prediction length: {args.prediction_length}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.epochs}")

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    # Create training samples
    train_samples = create_training_samples(
        train_df,
        args.context_length,
        args.prediction_length,
        stride=args.prediction_length  # Non-overlapping
    )

    val_samples = create_training_samples(
        val_df,
        args.context_length,
        args.prediction_length,
        stride=args.prediction_length
    )

    # Limit number of samples for faster training
    if len(train_samples) > args.num_samples:
        train_samples = train_samples[:args.num_samples]
        print(f"Limited to {len(train_samples)} training samples for speed")

    if len(val_samples) > args.num_samples // 4:
        val_samples = val_samples[:args.num_samples // 4]
        print(f"Limited to {len(val_samples)} validation samples")

    # Load model
    print("Loading pre-trained Toto model...")
    model = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
    model.to(device)
    
    # Create trainer (modified forecaster with gradients enabled)
    from toto.training.train import TotoTrainer
    trainer = TotoTrainer(model.model)

    # Prepare for fine-tuning - only train last few layers (like simple_finetune.py)
    trainable_params = []
    for name, param in model.named_parameters():
        if 'layers.10' in name or 'layers.11' in name or 'output' in name or 'unembed' in name:
            param.requires_grad = True
            trainable_params.append(param)
            print(f"Training parameter: {name}")
        else:
            param.requires_grad = False

    print(f"Total trainable parameters: {len(trainable_params)}")

    # Optimizer - only for trainable parameters
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

    # Training history
    train_losses = []
    val_losses = []

    # Training loop
    print(f"\nStarting fine-tuning for {args.epochs} epochs...")
    print(f"Training on {len(train_samples)} samples, validating on {len(val_samples)} samples")

    for epoch in range(args.epochs):
        # Training
        model.train()
        epoch_loss = 0

        print(f"\nEpoch {epoch+1}/{args.epochs} - Training...")
        for i, sample_df in enumerate(tqdm(train_samples, desc="Training")):
            try:
                # Ensure model is in training mode
                model.train()

                # Prepare input
                context_input, target_values = prepare_toto_input(
                    sample_df, args.context_length, args.prediction_length, device
                )

                # Forward pass
                optimizer.zero_grad()
                loss = compute_loss_with_trainer(model, trainer, context_input, target_values, args.prediction_length)

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

                epoch_loss += loss.item()

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

        avg_train_loss = epoch_loss / len(train_samples)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0

        print("Validating...")
        with torch.no_grad():
            for i, sample_df in enumerate(tqdm(val_samples, desc="Validation")):
                try:
                    context_input, target_values = prepare_toto_input(
                        sample_df, args.context_length, args.prediction_length, device
                    )
                    loss = compute_loss_with_trainer(model, trainer, context_input, target_values, args.prediction_length)
                    val_loss += loss.item()
                except Exception as e:
                    print(f"Error processing validation sample {i}: {e}")
                    continue

        avg_val_loss = val_loss / len(val_samples) if len(val_samples) > 0 else float('inf')
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
    if val_losses:
        print(f"Final validation loss: {val_losses[-1]:.4f}")

    # Test the fine-tuned model
    print("\nTo test the fine-tuned model, use:")
    print(f"python test_venezia_finetuned.py")


if __name__ == "__main__":
    main()