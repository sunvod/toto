#!/usr/bin/env python3
"""
Enhanced CSV testing script with date range support.
Usage: python run_csv_test.py <csv_file> [--train-start YYYY-MM-DD] [--train-end YYYY-MM-DD] [--test-start YYYY-MM-DD] [--test-end YYYY-MM-DD] [--context-length N] [--prediction-length N]
"""

import os
import sys
import argparse
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto

# Set up environment
project_root = os.path.dirname(os.path.abspath(__file__))
toto_path = os.path.join(project_root, "toto")

# Add paths to Python path
sys.path.insert(0, project_root)
sys.path.insert(0, toto_path)

# Set environment variables
os.environ["PYTHONPATH"] = f"{project_root}:{toto_path}:{os.environ.get('PYTHONPATH', '')}"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)



def load_csv_with_date_range(file_path, train_start=None, train_end=None, test_start=None, test_end=None):
    """
    Load CSV data with specific date ranges for training and testing.
    """
    print(f"Loading data from {file_path}...")
    
    # Load full dataset first
    df = pd.read_csv(file_path)
    
    # Find datetime column
    datetime_col = None
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
            datetime_col = col
            break
        except:
            continue
    
    if not datetime_col:
        raise ValueError("No datetime column found in CSV")
    
    print(f"Found datetime column: {datetime_col}")
    print(f"Date range: {df[datetime_col].min()} to {df[datetime_col].max()}")
    
    # Sort by datetime
    df = df.sort_values(by=datetime_col)
    df['timestamp_seconds'] = (df[datetime_col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    
    # Calculate time interval
    if len(df) > 1:
        time_diffs = df['timestamp_seconds'].diff().dropna()
        interval = int(time_diffs.mode()[0])
    else:
        interval = 1
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'timestamp_seconds' in numeric_cols:
        numeric_cols.remove('timestamp_seconds')
    
    print(f"Found {len(numeric_cols)} numeric columns: {numeric_cols}")
    
    # Filter by date ranges
    train_df = None
    test_df = None
    
    if train_start or train_end:
        train_mask = pd.Series(True, index=df.index)
        if train_start:
            train_start = pd.to_datetime(train_start)
            train_mask &= (df[datetime_col] >= train_start)
            print(f"Training start: {train_start}")
        if train_end:
            train_end = pd.to_datetime(train_end)
            train_mask &= (df[datetime_col] <= train_end)
            print(f"Training end: {train_end}")
        train_df = df[train_mask].copy()
        print(f"Training data: {len(train_df)} rows")
    
    if test_start or test_end:
        test_mask = pd.Series(True, index=df.index)
        if test_start:
            test_start = pd.to_datetime(test_start)
            test_mask &= (df[datetime_col] >= test_start)
            print(f"Test start: {test_start}")
        if test_end:
            test_end = pd.to_datetime(test_end)
            test_mask &= (df[datetime_col] <= test_end)
            print(f"Test end: {test_end}")
        test_df = df[test_mask].copy()
        print(f"Test data: {len(test_df)} rows")
    
    return df, train_df, test_df, numeric_cols, interval, datetime_col

def prepare_toto_input(df, feature_columns, interval, device='cuda'):
    """
    Prepare data in Toto's expected format.
    """
    n_variates = len(feature_columns)
    context_length = len(df)
    
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

def plot_results(train_df, test_df, forecast, feature_columns, datetime_col, prediction_length):
    """
    Plot ONLY the forecast period with MAE - no historical context.
    """
    n_variates = len(feature_columns)
    
    # Create figure with 2:1 ratio for forecast vs MAE
    fig = plt.figure(figsize=(15, 10), dpi=100)
    
    # Use GridSpec for custom layout
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 0.1], hspace=0.3, figure=fig)
    
    # Store MAE values for each feature
    mae_values = {}
    
    # Get forecast dates - these will be used for both plots
    forecast_dates = test_df[datetime_col].iloc[:prediction_length]
    
    # Main forecast plot
    ax1 = plt.subplot(gs[0])
    
    for i, feature in enumerate(feature_columns):
        # Get forecast values
        if n_variates == 1:
            forecast_median = np.median(forecast.samples.squeeze().cpu(), axis=-1)
            qs = forecast.samples.quantile(
                q=torch.tensor([0.05, 0.95], device=forecast.samples.device), 
                dim=-1
            )
            q_low = qs[0].squeeze().cpu()
            q_high = qs[1].squeeze().cpu()
        else:
            forecast_median = np.median(forecast.samples.squeeze()[i].cpu(), axis=-1)
            qs = forecast.samples.quantile(
                q=torch.tensor([0.05, 0.95], device=forecast.samples.device), 
                dim=-1
            )
            q_low = qs[0].squeeze()[i].cpu()
            q_high = qs[1].squeeze()[i].cpu()
        
        # Get actual values
        actual_values = test_df[feature].iloc[:prediction_length].values
        
        # Plot ONLY the forecast period
        ax1.plot(forecast_dates, actual_values, color='green', label=f'{feature} (Actual)', linewidth=2.5)
        ax1.plot(forecast_dates, forecast_median, color='red', linestyle='--', label=f'{feature} (Forecast)', linewidth=2)
        ax1.fill_between(forecast_dates, q_low, q_high, color='red', alpha=0.2)
        
        # Calculate MAE
        mae = np.mean(np.abs(forecast_median - actual_values))
        mae_values[feature] = mae
    
    # Format main plot
    ax1.set_title(f'Forecast vs Actual - {forecast_dates.iloc[0].strftime("%Y-%m-%d %H:%M")} to {forecast_dates.iloc[-1].strftime("%Y-%m-%d %H:%M")}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(forecast_dates.iloc[0], forecast_dates.iloc[-1])
    
    # MAE plot - exactly aligned with main plot
    ax2 = plt.subplot(gs[1], sharex=ax1)  # Share x-axis with main plot
    
    for i, feature in enumerate(feature_columns):
        # Get forecast values again
        if n_variates == 1:
            forecast_median = np.median(forecast.samples.squeeze().cpu(), axis=-1)
        else:
            forecast_median = np.median(forecast.samples.squeeze()[i].cpu(), axis=-1)
        
        actual_values = test_df[feature].iloc[:prediction_length].values
        
        # Calculate point-wise absolute errors
        point_errors = np.abs(forecast_median - actual_values)
        
        # Calculate cumulative MAE over time
        cumulative_mae = np.array([np.mean(point_errors[:j+1]) for j in range(len(point_errors))])
        
        # Plot the MAE evolution
        ax2.plot(forecast_dates, cumulative_mae, label=f'{feature} (Final MAE: {mae_values[feature]:.2f})', linewidth=2.5)
        
        # Add horizontal line for final MAE
        ax2.axhline(y=mae_values[feature], color='gray', linestyle=':', alpha=0.5)
    
    # Format MAE plot
    ax2.set_title('Mean Absolute Error (MAE) Evolution', fontsize=12, fontweight='bold')
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(forecast_dates.iloc[0], forecast_dates.iloc[-1])  # Exactly same limits
    ax2.set_ylim(bottom=0)  # MAE starts from 0
    
    # Format x-axis dates
    import matplotlib.dates as mdates
    
    # Set date formatter for better readability
    if prediction_length <= 24:  # Less than a day - show hours
        date_formatter = mdates.DateFormatter('%m-%d %H:%M')
        ax2.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    elif prediction_length <= 168:  # Less than a week - show days
        date_formatter = mdates.DateFormatter('%m-%d')
        ax2.xaxis.set_major_locator(mdates.DayLocator())
    else:  # More than a week - show dates
        date_formatter = mdates.DateFormatter('%Y-%m-%d')
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    
    ax2.xaxis.set_major_formatter(date_formatter)
    
    # Rotate x-axis labels for readability
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Hide x-axis labels on top plot since they're shared with bottom plot
    ax1.tick_params(axis='x', which='both', labelbottom=False)
    
    # Overall title
    overall_mae = np.mean(list(mae_values.values()))
    fig.suptitle(f'Toto Forecast: {prediction_length} steps, Overall MAE: {overall_mae:.2f}', fontsize=16, fontweight='bold')
    
    # Use constrained_layout instead of tight_layout to avoid warnings
    plt.subplots_adjust(top=0.93)
    plt.savefig('toto_forecast_mae_aligned.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'toto_forecast_mae_aligned.png'")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Test Toto with CSV data and custom date ranges')
    parser.add_argument('csv_file', help='Path to CSV file')
    parser.add_argument('--train-start', help='Training start date (YYYY-MM-DD)')
    parser.add_argument('--train-end', help='Training end date (YYYY-MM-DD)')
    parser.add_argument('--test-start', help='Test start date (YYYY-MM-DD)')
    parser.add_argument('--test-end', help='Test end date (YYYY-MM-DD)')
    parser.add_argument('--context-length', type=int, default=2048, help='Context length for training (default: 2048)')
    parser.add_argument('--prediction-length', type=int, default=96, help='Prediction length (default: 96)')
    parser.add_argument('--num-samples', type=int, default=32, help='Number of forecast samples (default: 32)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to fine-tuned model checkpoint')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: CUDA not available. Running on CPU will be slower.")
    
    # Load data with date filtering
    full_df, train_df, test_df, feature_columns, interval, datetime_col = load_csv_with_date_range(
        args.csv_file, args.train_start, args.train_end, args.test_start, args.test_end
    )
    
    # Use training data or fall back to automatic splitting
    if train_df is not None:
        # Use specified training data
        context_data = train_df.tail(args.context_length)  # Use last N rows of training data
        print(f"Using {len(context_data)} rows from training data as context")
    else:
        # Automatic splitting - use before test period as training
        if test_df is not None:
            split_date = test_df[datetime_col].iloc[0]
            context_data = full_df[full_df[datetime_col] < split_date].tail(args.context_length)
            print(f"Using {len(context_data)} rows before test period as context")
        else:
            # No date ranges specified, use traditional split
            context_data = full_df.tail(args.context_length + args.prediction_length).head(args.context_length)
            test_df = full_df.tail(args.prediction_length)
            print(f"Using automatic split: {len(context_data)} context, {len(test_df)} test")
    
    if len(context_data) == 0:
        raise ValueError("No training/context data available with specified date ranges")
    
    # Ensure we have test data
    if test_df is None or len(test_df) == 0:
        raise ValueError("No test data available with specified date ranges")
    
    prediction_length = min(args.prediction_length, len(test_df))
    print(f"Using context length: {len(context_data)}, prediction length: {prediction_length}")
    
    # Prepare input
    inputs = prepare_toto_input(context_data, feature_columns, interval, device)
    
    # Load model
    if args.checkpoint:
        print(f"\nLoading fine-tuned model from {args.checkpoint}...")
        toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
        checkpoint = torch.load(args.checkpoint, map_location=device)
        toto.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model fine-tuned for {checkpoint['epoch']} epochs")
    else:
        print("\nLoading pre-trained Toto model...")
        toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
    
    toto.to(device)
    if device != 'cpu':
        toto.compile()
    
    # Generate forecast
    print(f"\nGenerating forecasts for {prediction_length} steps...")
    forecaster = TotoForecaster(toto.model)
    forecast = forecaster.forecast(
        inputs,
        prediction_length=prediction_length,
        num_samples=args.num_samples,
        samples_per_batch=min(args.num_samples, 32),
        use_kv_cache=True,
    )
    
    print("\nForecast complete!")
    
    # Display results
    print("\nForecast summary (median values):")
    print("-" * 80)
    
    all_metrics = []
    
    for i, feature in enumerate(feature_columns):
        if len(feature_columns) == 1:
            forecast_values = forecast.samples.squeeze().cpu()
        else:
            forecast_values = forecast.samples.squeeze()[i].cpu()
        
        median_forecast = np.median(forecast_values, axis=-1)
        actual_values = test_df[feature].iloc[:prediction_length].values
        
        # Calculate various metrics
        mae = np.mean(np.abs(median_forecast - actual_values))
        mse = np.mean((median_forecast - actual_values) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual_values - median_forecast) / actual_values)) * 100 if np.all(actual_values != 0) else np.nan
        
        # Calculate R² score
        ss_res = np.sum((actual_values - median_forecast) ** 2)
        ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        
        print(f"\n{feature}:")
        print(f"  Forecast - mean: {median_forecast.mean():.2f}, std: {median_forecast.std():.2f}")
        print(f"  Actual   - mean: {actual_values.mean():.2f}, std: {actual_values.std():.2f}")
        print(f"  Metrics:")
        print(f"    MAE:  {mae:.2f}")
        print(f"    RMSE: {rmse:.2f}")
        print(f"    MAPE: {mape:.2f}%" if not np.isnan(mape) else "    MAPE: N/A (contains zeros)")
        print(f"    R²:   {r2:.4f}" if not np.isnan(r2) else "    R²:   N/A")
        
        all_metrics.append({
            'feature': feature,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        })
    
    print("-" * 80)
    print(f"\nOverall MAE: {np.mean([m['mae'] for m in all_metrics]):.2f}")
    print(f"Overall RMSE: {np.mean([m['rmse'] for m in all_metrics]):.2f}")
    
    # Plot results
    plot_results(train_df if train_df is not None else context_data, test_df, forecast, feature_columns, datetime_col, prediction_length)

if __name__ == "__main__":
    main()
    