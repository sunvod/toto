#!/usr/bin/env python3
"""
Test the fine-tuned Toto model on Venice water level data.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Set up environment
project_root = os.path.dirname(os.path.abspath(__file__))
toto_path = os.path.join(project_root, "toto")
sys.path.insert(0, project_root)
sys.path.insert(0, toto_path)

os.environ["PYTHONPATH"] = f"{project_root}:{toto_path}:{os.environ.get('PYTHONPATH', '')}"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from data.util.dataset import MaskedTimeseries
from model.toto import Toto
from inference.forecaster import TotoForecaster


def test_finetuned_model():
    # Load data
    print("Loading Venice data...")
    df = pd.read_csv('toto/datasets/venezia.csv')
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values('Data')
    df['timestamp_seconds'] = (df['Data'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    
    # Parameters
    context_length = 2048
    prediction_length = 96
    
    # Select a test window from the end of the dataset
    test_start_idx = len(df) - context_length - prediction_length - 100
    test_end_idx = test_start_idx + context_length + prediction_length
    
    test_window = df.iloc[test_start_idx:test_end_idx]
    context_data = test_window.iloc[:context_length]
    actual_future = test_window.iloc[context_length:context_length + prediction_length]
    
    print(f"Context period: {context_data['Data'].iloc[0]} to {context_data['Data'].iloc[-1]}")
    print(f"Prediction period: {actual_future['Data'].iloc[0]} to {actual_future['Data'].iloc[-1]}")
    
    # Load models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Load base model
    print("\nLoading base model for comparison...")
    base_model = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
    base_model.to(device)
    base_model.eval()
    
    # Load fine-tuned model
    print("Loading fine-tuned model...")
    finetuned_model = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
    checkpoint = torch.load('toto_venezia_simple_finetuned.pt', map_location=device, weights_only=False)
    finetuned_model.load_state_dict(checkpoint['model_state_dict'])
    finetuned_model.to(device)
    finetuned_model.eval()
    
    print(f"Fine-tuned model was trained for {checkpoint['epoch']} epochs")
    if 'val_losses' in checkpoint and checkpoint['val_losses']:
        print(f"Final validation loss: {checkpoint['val_losses'][-1]:.4f}")
    
    # Prepare input data
    context_values = torch.tensor(
        context_data['Livello Punta Salute (cm)'].values,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, context_length)
    
    context_timestamps = torch.tensor(
        context_data['timestamp_seconds'].values,
        dtype=torch.int64
    ).unsqueeze(0).unsqueeze(0).to(device)
    
    context_input = MaskedTimeseries(
        series=context_values,
        padding_mask=torch.ones_like(context_values, dtype=torch.bool),
        id_mask=torch.zeros_like(context_values, dtype=torch.int),
        timestamp_seconds=context_timestamps,
        time_interval_seconds=torch.tensor([[3600]], dtype=torch.int).to(device),  # 1 hour
    )
    
    # Generate forecasts
    print("\nGenerating forecasts...")
    
    # Base model forecast
    base_forecaster = TotoForecaster(base_model.model)
    base_forecast = base_forecaster.forecast(
        context_input,
        prediction_length=prediction_length,
        num_samples=100,
    )
    
    # Fine-tuned model forecast
    finetuned_forecaster = TotoForecaster(finetuned_model.model)
    finetuned_forecast = finetuned_forecaster.forecast(
        context_input,
        prediction_length=prediction_length,
        num_samples=100,
    )
    
    # Extract predictions
    base_median = base_forecast.median[0, 0].cpu().numpy()
    base_p10 = base_forecast.quantile(0.1)[0, 0].cpu().numpy()
    base_p90 = base_forecast.quantile(0.9)[0, 0].cpu().numpy()
    
    finetuned_median = finetuned_forecast.median[0, 0].cpu().numpy()
    finetuned_p10 = finetuned_forecast.quantile(0.1)[0, 0].cpu().numpy()
    finetuned_p90 = finetuned_forecast.quantile(0.9)[0, 0].cpu().numpy()
    
    actual_values = actual_future['Livello Punta Salute (cm)'].values
    
    # Calculate metrics
    base_mae = np.mean(np.abs(base_median - actual_values))
    finetuned_mae = np.mean(np.abs(finetuned_median - actual_values))
    
    print(f"\nBase model MAE: {base_mae:.2f} cm")
    print(f"Fine-tuned model MAE: {finetuned_mae:.2f} cm")
    print(f"Improvement: {((base_mae - finetuned_mae) / base_mae * 100):.1f}%")
    
    # Plot results
    plt.figure(figsize=(15, 8))
    
    # Plot context
    context_times = context_data['Data'].values
    plt.plot(context_times[-200:], context_values[0, 0, -200:].cpu().numpy(), 
             'b-', linewidth=1, label='Historical data')
    
    # Plot actual future
    future_times = actual_future['Data'].values
    plt.plot(future_times, actual_values, 'k-', linewidth=2, label='Actual')
    
    # Plot base model predictions
    plt.plot(future_times, base_median, 'g--', linewidth=2, label='Base model (median)')
    plt.fill_between(future_times, base_p10, base_p90, alpha=0.2, color='green')
    
    # Plot fine-tuned model predictions
    plt.plot(future_times, finetuned_median, 'r--', linewidth=2, label='Fine-tuned model (median)')
    plt.fill_between(future_times, finetuned_p10, finetuned_p90, alpha=0.2, color='red')
    
    plt.xlabel('Time')
    plt.ylabel('Water Level (cm)')
    plt.title('Venice Water Level Forecasting: Base vs Fine-tuned Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('venice_forecast_comparison.png', dpi=150)
    print("\nPlot saved as 'venice_forecast_comparison.png'")
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    
    base_errors = base_median - actual_values
    finetuned_errors = finetuned_median - actual_values
    
    plt.subplot(1, 2, 1)
    plt.hist(base_errors, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Prediction Error (cm)')
    plt.ylabel('Frequency')
    plt.title(f'Base Model\nMAE: {base_mae:.2f} cm')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(finetuned_errors, bins=30, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel('Prediction Error (cm)')
    plt.ylabel('Frequency')
    plt.title(f'Fine-tuned Model\nMAE: {finetuned_mae:.2f} cm')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('venice_error_distribution.png', dpi=150)
    print("Error distribution plot saved as 'venice_error_distribution.png'")
    
    plt.show()


if __name__ == "__main__":
    test_finetuned_model()