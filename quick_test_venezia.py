#!/usr/bin/env python3
"""
Quick test of the fine-tuned model on Venice data.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

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


def quick_test():
    # Load data
    print("Loading Venice data...")
    df = pd.read_csv('toto/datasets/venezia.csv')
    df['Data'] = pd.to_datetime(df['Data'])
    df = df.sort_values('Data')
    df['timestamp_seconds'] = (df['Data'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    
    # Parameters
    context_length = 512  # Reduced for faster testing
    prediction_length = 24  # 24 hours
    
    # Select a test window
    test_idx = len(df) - context_length - prediction_length - 100
    context_data = df.iloc[test_idx:test_idx + context_length]
    actual_future = df.iloc[test_idx + context_length:test_idx + context_length + prediction_length]
    
    print(f"Testing on period: {actual_future['Data'].iloc[0]} to {actual_future['Data'].iloc[-1]}")
    
    # Load fine-tuned model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Loading fine-tuned model...")
    model = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
    checkpoint = torch.load('toto_venezia_simple_finetuned.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Prepare input
    context_values = torch.tensor(
        context_data['Livello Punta Salute (cm)'].values,
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(device)
    
    context_timestamps = torch.tensor(
        context_data['timestamp_seconds'].values,
        dtype=torch.int64
    ).unsqueeze(0).unsqueeze(0).to(device)
    
    context_input = MaskedTimeseries(
        series=context_values,
        padding_mask=torch.ones_like(context_values, dtype=torch.bool),
        id_mask=torch.zeros_like(context_values, dtype=torch.int),
        timestamp_seconds=context_timestamps,
        time_interval_seconds=torch.tensor([[3600]], dtype=torch.int).to(device),
    )
    
    # Generate forecast
    print("Generating forecast...")
    forecaster = TotoForecaster(model.model)
    forecast = forecaster.forecast(
        context_input,
        prediction_length=prediction_length,
        num_samples=50,  # Reduced for speed
    )
    
    # Extract results
    forecast_median = forecast.median[0, 0].cpu().numpy()
    forecast_p10 = forecast.quantile(0.1)[0, 0].cpu().numpy()
    forecast_p90 = forecast.quantile(0.9)[0, 0].cpu().numpy()
    actual_values = actual_future['Livello Punta Salute (cm)'].values
    
    # Calculate MAE
    mae = np.mean(np.abs(forecast_median - actual_values))
    print(f"\nMAE: {mae:.2f} cm")
    
    # Simple plot
    plt.figure(figsize=(12, 6))
    
    # Plot last part of context
    context_plot = context_data.tail(100)
    plt.plot(context_plot['Data'], context_plot['Livello Punta Salute (cm)'], 
             'b-', linewidth=1, label='Historical')
    
    # Plot forecast and actual
    plt.plot(actual_future['Data'], actual_values, 'k-', linewidth=2, label='Actual')
    plt.plot(actual_future['Data'], forecast_median, 'r--', linewidth=2, label='Forecast')
    plt.fill_between(actual_future['Data'], forecast_p10, forecast_p90, 
                     alpha=0.3, color='red', label='80% confidence')
    
    plt.xlabel('Time')
    plt.ylabel('Water Level (cm)')
    plt.title(f'Venice Water Level Forecast (MAE: {mae:.2f} cm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig('venice_quick_test.png', dpi=150)
    print("Plot saved as 'venice_quick_test.png'")
    
    # Print some statistics
    print(f"\nForecast statistics:")
    print(f"Mean water level (actual): {np.mean(actual_values):.2f} cm")
    print(f"Mean water level (forecast): {np.mean(forecast_median):.2f} cm")
    print(f"Std deviation (actual): {np.std(actual_values):.2f} cm")
    print(f"Std deviation (forecast): {np.std(forecast_median):.2f} cm")


if __name__ == "__main__":
    quick_test()