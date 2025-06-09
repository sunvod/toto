#!/usr/bin/env python3
"""
Script to test Toto with CSV data.
Usage: python test_csv_data.py <path_to_csv_file>
"""

import os
import sys
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto

# These lines make gpu execution in CUDA deterministic
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

def load_csv_data(file_path):
    """
    Load CSV data and prepare it for Toto.
    
    Expected CSV format:
    - First column: datetime/timestamp
    - Other columns: numeric time series values
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Try to identify and parse datetime column
    datetime_col = None
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
            datetime_col = col
            break
        except:
            continue
    
    if datetime_col:
        print(f"Found datetime column: {datetime_col}")
        df = df.sort_values(by=datetime_col)
        df['timestamp_seconds'] = (df[datetime_col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        
        # Calculate time interval
        if len(df) > 1:
            time_diffs = df['timestamp_seconds'].diff().dropna()
            interval = int(time_diffs.mode()[0])
        else:
            interval = 1
    else:
        print("No datetime column found. Creating index-based timestamps.")
        df['timestamp_seconds'] = np.arange(len(df))
        interval = 1
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'timestamp_seconds' in numeric_cols:
        numeric_cols.remove('timestamp_seconds')
    
    print(f"Found {len(numeric_cols)} numeric columns: {numeric_cols}")
    
    return df, numeric_cols, interval, datetime_col

def prepare_toto_input(df, feature_columns, context_length, interval, device='cuda'):
    """
    Prepare data in Toto's expected format.
    """
    n_variates = len(feature_columns)
    
    # Convert to tensor
    input_series = torch.from_numpy(df[feature_columns].values.T).to(torch.float).to(device)
    
    # Prepare timestamps
    timestamp_seconds = torch.from_numpy(df.timestamp_seconds.values.T).expand((n_variates, context_length)).to(device)
    time_interval_seconds = torch.full((n_variates,), interval).to(device)
    
    # Create MaskedTimeseries object
    inputs = MaskedTimeseries(
        series=input_series,
        padding_mask=torch.full_like(input_series, True, dtype=torch.bool),
        id_mask=torch.zeros_like(input_series, dtype=torch.int),
        timestamp_seconds=timestamp_seconds,
        time_interval_seconds=time_interval_seconds,
    )
    
    return inputs

def plot_results(input_df, target_df, forecast, feature_columns, datetime_col=None):
    """
    Plot forecasts against ground truth.
    """
    fig = plt.figure(figsize=(14, 8), layout="tight", dpi=100)
    plt.suptitle("Toto Forecasts from CSV Data")
    
    n_variates = len(feature_columns)
    n_cols = 2 if n_variates > 3 else 1
    n_rows = (n_variates + n_cols - 1) // n_cols
    
    x_axis = input_df[datetime_col] if datetime_col else input_df.index
    x_axis_target = target_df[datetime_col] if datetime_col else target_df.index
    
    for i, feature in enumerate(feature_columns):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Plot historical data
        plt.plot(x_axis, input_df[feature], color='blue', label='Historical')
        
        # Plot ground truth if available
        if feature in target_df.columns:
            plt.plot(x_axis_target, target_df[feature], color='green', label='Ground Truth')
        
        # Plot forecast
        forecast_median = np.median(forecast.samples.squeeze()[i].cpu(), axis=-1)
        plt.plot(x_axis_target, forecast_median, color='red', linestyle='--', label='Forecast')
        
        # Plot confidence interval
        alpha = 0.05
        qs = forecast.samples.quantile(
            q=torch.tensor([alpha, 1 - alpha], device=forecast.samples.device), 
            dim=-1
        )
        plt.fill_between(
            x_axis_target if isinstance(x_axis_target, pd.Series) else np.arange(len(x_axis_target)),
            qs[0].squeeze()[i].cpu(),
            qs[1].squeeze()[i].cpu(),
            color='red',
            alpha=0.3
        )
        
        plt.title(feature)
        plt.legend()
        
        # Rotate x labels if using datetime
        if datetime_col:
            plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('toto_csv_forecast.png')
    print("Plot saved as 'toto_csv_forecast.png'")
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python test_csv_data.py <path_to_csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: CUDA not available. Running on CPU will be slower.")
    
    # Load data
    df, feature_columns, interval, datetime_col = load_csv_data(csv_file)
    
    # Set parameters
    context_length = min(2048, len(df) // 2)  # Use half the data as context, max 2048
    prediction_length = min(96, len(df) - context_length)  # Predict up to 96 steps
    
    print(f"\nUsing context length: {context_length}")
    print(f"Prediction length: {prediction_length}")
    
    # Split data
    if len(df) > context_length + prediction_length:
        # If we have enough data, use the last part for validation
        input_df = df.iloc[-(context_length+prediction_length):-prediction_length]
        target_df = df.iloc[-prediction_length:]
    else:
        # Otherwise, just use what we have
        input_df = df.iloc[:context_length]
        # Create synthetic future timestamps for prediction
        if datetime_col:
            last_timestamp = input_df[datetime_col].iloc[-1]
            future_timestamps = pd.date_range(
                start=last_timestamp + pd.Timedelta(seconds=interval),
                periods=prediction_length,
                freq=f'{interval}s'
            )
            target_df = pd.DataFrame({datetime_col: future_timestamps})
        else:
            target_df = pd.DataFrame(index=range(len(input_df), len(input_df) + prediction_length))
        
        target_df['timestamp_seconds'] = input_df['timestamp_seconds'].iloc[-1] + (np.arange(1, prediction_length + 1) * interval)
    
    # Prepare input
    inputs = prepare_toto_input(input_df, feature_columns, context_length, interval, device)
    
    # Load model
    print("\nLoading Toto model...")
    toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
    toto.to(device)
    toto.compile()
    
    # Generate forecast
    print("\nGenerating forecasts...")
    forecaster = TotoForecaster(toto.model)
    forecast = forecaster.forecast(
        inputs,
        prediction_length=prediction_length,
        num_samples=64,  # Reduced for faster testing
        samples_per_batch=64,
        use_kv_cache=True,
    )
    
    print("\nForecast complete!")
    
    # Display results
    print("\nForecast summary (median values):")
    for i, feature in enumerate(feature_columns):
        median_forecast = np.median(forecast.samples.squeeze()[i].cpu(), axis=-1)
        print(f"{feature}: mean={median_forecast.mean():.2f}, std={median_forecast.std():.2f}")
    
    # Plot results
    plot_results(input_df, target_df, forecast, feature_columns, datetime_col)

if __name__ == "__main__":
    main()