# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

"""
Configurable training script for Toto that manages checkpoints and model organization.

This script:
- Takes data file path and time series parameters as input
- Creates organized model directories based on configuration
- Automatically resumes from checkpoint if exists
- Saves checkpoints periodically during training
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple, List, Union
import logging
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm

from toto.data.util.dataset import MaskedTimeseries
from toto.model.toto import Toto
from toto.training.train import TotoTrainer
from toto.inference.forecaster import TotoForecaster


logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """Dataset for time series with configurable window parameters."""
    
    def __init__(
        self, 
        df: pd.DataFrame,
        target_column: str,
        context_length: int,
        prediction_length: int,
        stride: int,
        time_interval_seconds: int = 3600,
        mode: str = "train",
        feature_columns: Optional[List[str]] = None
    ):
        self.df = df
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.stride = stride
        self.time_interval_seconds = time_interval_seconds
        self.mode = mode
        self.total_length = context_length + prediction_length
        
        # Determine all columns to use (target + features)
        if self.feature_columns is None:
            self.all_columns = [self.target_column]
        else:
            # Ensure target is first, then features
            self.all_columns = [self.target_column] + [col for col in self.feature_columns if col != self.target_column]
        
        self.n_variates = len(self.all_columns)
        
        # Calculate valid starting positions
        self.valid_starts = []
        for i in range(0, len(df) - self.total_length + 1, stride):
            self.valid_starts.append(i)
        
        logger.info(f"Created {self.mode} dataset with {len(self.valid_starts)} samples, {self.n_variates} variates")
    
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
        
        if self.n_variates == 1:
            # Univariate case - maintain backward compatibility
            return {
                'context_values': context[self.target_column].values.astype(np.float32),
                'context_timestamps': context['timestamp_seconds'].values.astype(np.int64),
                'target_values': target[self.target_column].values.astype(np.float32),
                'time_interval': self.time_interval_seconds,
                'n_variates': 1
            }
        else:
            # Multivariate case - shape (variates, time_steps)
            context_values = context[self.all_columns].values.T.astype(np.float32)
            target_values = target[self.target_column].values.astype(np.float32)  # Only target for loss
            
            return {
                'context_values': context_values,
                'context_timestamps': context['timestamp_seconds'].values.astype(np.int64),
                'target_values': target_values,
                'time_interval': self.time_interval_seconds,
                'n_variates': self.n_variates
            }


class TotoTrainingPipeline:
    """Main training pipeline with checkpoint management."""
    
    def __init__(
        self,
        data_path: str,
        target_column: str,
        context_length: int,
        prediction_length: int,
        model_name: str = "toto_base",
        base_model: str = "Datadog/Toto-Open-Base-1.0",
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        min_lr: float = 1e-6,
        lr_factor: float = 0.1,
        lr_patience: int = 10,
        epochs: int = 100,
        loss_fn: str = 'mse',
        eval_every: int = 5,
        save_every: int = 10,
        patience: int = 20,
        trainable_layers: list = None,
        time_interval_seconds: int = 3600,
        start_date: str = None,
        train_samples: int = None,
        val_samples: str = '10%',
        test_samples: str = '10%',
        feature_columns: Optional[Union[List[str], int, str]] = None
    ):
        self.data_path = data_path
        self.target_column = target_column
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.model_name = model_name
        self.base_model = base_model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.eval_every = eval_every
        self.save_every = save_every
        self.patience = patience
        self.trainable_layers = trainable_layers or [10, 11]
        self.time_interval_seconds = time_interval_seconds
        self.start_date = start_date
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.feature_columns = feature_columns
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model directory
        self.model_dir = self._create_model_directory()
        
        # Setup logging
        self._setup_logging()
        
        # Log device info after logging is configured
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(self.device)
            gpu_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            logger.info("Using CPU - training will be slow!")
        
        # Initialize tracking variables
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
    
    def _parse_column_spec(self, spec: Union[str, int], df: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> Union[str, List[str]]:
        """Parse column specification (name, index, range, or pattern).
        
        Args:
            spec: Column specification (string name, index, range like '1:5', or pattern like '*sales*')
            df: DataFrame to get columns from
            exclude_columns: Columns to exclude from selection
        
        Returns:
            Column name(s) matching the specification
        """
        available_columns = [col for col in df.columns if col not in (exclude_columns or [])]
        
        if isinstance(spec, int):
            # Numeric index
            if spec < 0:
                # Negative indexing
                return available_columns[spec]
            else:
                return available_columns[spec]
        
        elif isinstance(spec, str):
            # Check if it's a range like "1:5"
            if ':' in spec:
                start, end = map(int, spec.split(':'))
                return available_columns[start:end]
            
            # Check if it's a pattern with wildcards
            elif '*' in spec:
                pattern = spec.replace('*', '.*')
                regex = re.compile(pattern, re.IGNORECASE)
                matches = [col for col in available_columns if regex.search(col)]
                if matches:
                    return matches[0]  # Return first match
                else:
                    raise ValueError(f"No column matches pattern '{spec}'")
            
            # Check if it's a number as string
            elif spec.isdigit() or (spec.startswith('-') and spec[1:].isdigit()):
                return self._parse_column_spec(int(spec), df, exclude_columns)
            
            # Otherwise treat as column name
            else:
                if spec in df.columns:
                    return spec
                else:
                    raise ValueError(f"Column '{spec}' not found in dataset")
        
        else:
            raise ValueError(f"Invalid column specification: {spec}")
    
    def _parse_feature_columns(self, spec: Union[List[str], int, str], df: pd.DataFrame, target_column: str) -> Optional[List[str]]:
        """Parse feature columns specification.
        
        Args:
            spec: Feature columns specification
            df: DataFrame to get columns from
            target_column: Target column to exclude from features
        
        Returns:
            List of feature column names or None
        """
        if spec is None:
            return None
        
        exclude = ['timestamp_seconds', target_column]
        available_columns = [col for col in df.columns if col not in exclude]
        
        if isinstance(spec, int):
            # Take first N columns (excluding target and timestamp)
            if spec == -1:
                # All available columns
                return available_columns
            else:
                return available_columns[:spec]
        
        elif isinstance(spec, str):
            # Parse as single column spec
            result = self._parse_column_spec(spec, df, exclude)
            if isinstance(result, list):
                return result
            else:
                return [result]
        
        elif isinstance(spec, list):
            # List of column specs
            feature_columns = []
            for item in spec:
                result = self._parse_column_spec(item, df, exclude)
                if isinstance(result, list):
                    feature_columns.extend(result)
                else:
                    feature_columns.append(result)
            # Remove duplicates while preserving order
            seen = set()
            return [x for x in feature_columns if not (x in seen or seen.add(x))]
        
        else:
            raise ValueError(f"Invalid feature columns specification: {spec}")
    
    def _parse_sample_count(self, sample_spec: str, train_samples: int) -> int:
        """Parse sample count from string (absolute or percentage)."""
        if sample_spec.endswith('%'):
            # Percentage of train samples
            percentage = float(sample_spec[:-1]) / 100
            return int(train_samples * percentage)
        else:
            # Absolute number
            return int(sample_spec)
    
    def _create_model_directory(self) -> Path:
        """Create organized model directory structure."""
        model_dir = Path("toto/model") / f"{self.model_name}_{self.context_length}_{self.prediction_length}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (model_dir / "checkpoints").mkdir(exist_ok=True)
        (model_dir / "logs").mkdir(exist_ok=True)
        (model_dir / "plots").mkdir(exist_ok=True)
        
        return model_dir
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.model_dir / "logs" / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        # Save configuration
        config = {
            'data_path': self.data_path,
            'target_column': self.target_column,
            'context_length': self.context_length,
            'prediction_length': self.prediction_length,
            'model_name': self.model_name,
            'base_model': self.base_model,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'min_lr': self.min_lr,
            'lr_factor': self.lr_factor,
            'lr_patience': self.lr_patience,
            'epochs': self.epochs,
            'loss_fn': self.loss_fn,
            'trainable_layers': self.trainable_layers,
            'time_interval_seconds': self.time_interval_seconds,
            'start_date': self.start_date,
            'train_samples': self.train_samples,
            'val_samples': self.val_samples,
            'test_samples': self.test_samples,
            'feature_columns': self.feature_columns,
            'n_variates': self.n_variates if hasattr(self, 'n_variates') else 1
        }
        
        with open(self.model_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def _load_checkpoint(self) -> bool:
        """Load checkpoint if exists."""
        checkpoint_path = self.model_dir / "checkpoints" / "latest_checkpoint.pt"
        
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint['best_val_loss']
            self.patience_counter = checkpoint['patience_counter']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            
            logger.info(f"Resumed from epoch {self.start_epoch} with best val loss {self.best_val_loss:.4f}")
            return True
        
        return False
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': {
                'context_length': self.context_length,
                'prediction_length': self.prediction_length,
                'model_name': self.model_name
            }
        }
        
        # Save latest checkpoint
        latest_path = self.model_dir / "checkpoints" / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
        
        # Save periodic checkpoint
        if epoch % self.save_every == 0:
            periodic_path = self.model_dir / "checkpoints" / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, periodic_path)
        
        # Save best model
        if is_best:
            best_path = self.model_dir / "checkpoints" / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with val loss: {self.best_val_loss:.4f}")
    
    def _load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and prepare data."""
        logger.info(f"Loading data from {self.data_path}")
        
        # Load CSV
        df = pd.read_csv(self.data_path)
        
        # Parse target column
        if isinstance(self.target_column, (int, str)):
            try:
                # First check if it's a direct column name
                if isinstance(self.target_column, str) and self.target_column in df.columns:
                    parsed_target = self.target_column
                else:
                    # Try parsing as index/pattern
                    parsed_target = self._parse_column_spec(self.target_column, df)
                    if isinstance(parsed_target, list):
                        parsed_target = parsed_target[0]  # Take first if multiple matches
                self.target_column = parsed_target
                logger.info(f"Target column: {self.target_column}")
            except Exception as e:
                logger.error(f"Error parsing target column: {e}")
                raise
        
        # Store the number of variates for configuration
        self.n_variates = 1  # Default, will be updated after parsing features
        
        # Parse feature columns
        if self.feature_columns is not None:
            try:
                parsed_features = self._parse_feature_columns(self.feature_columns, df, self.target_column)
                self.feature_columns = parsed_features
                if parsed_features:
                    self.n_variates = len(parsed_features) + 1  # +1 for target
                    logger.info(f"Feature columns ({len(parsed_features)}): {', '.join(parsed_features[:5])}{'...' if len(parsed_features) > 5 else ''}")
                    logger.info(f"Total variates: {self.n_variates} (target + {len(parsed_features)} features)")
            except Exception as e:
                logger.error(f"Error parsing feature columns: {e}")
                raise
        
        # Handle timestamps
        if 'timestamp_seconds' not in df.columns:
            if 'Data' in df.columns:
                df['Data'] = pd.to_datetime(df['Data'])
                df['timestamp_seconds'] = (df['Data'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['timestamp_seconds'] = (df['timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
            else:
                # Assume regular intervals
                df['timestamp_seconds'] = np.arange(len(df)) * self.time_interval_seconds
        
        # Sort by time
        df = df.sort_values('timestamp_seconds').reset_index(drop=True)
        
        # Filter by start date if specified
        if self.start_date:
            start_timestamp = pd.to_datetime(self.start_date)
            start_seconds = (start_timestamp - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
            df = df[df['timestamp_seconds'] >= start_seconds].reset_index(drop=True)
            logger.info(f"Filtered data from {self.start_date}, remaining points: {len(df)}")
        
        # Log data info
        if 'Data' in df.columns:
            logger.info(f"Data range: {df['Data'].min()} to {df['Data'].max()}")
        logger.info(f"Total data points available: {len(df)}")
        
        # CORRECTED LOGIC: samples = data points, not windows
        n = len(df)
        
        # Step 1: Determine exact number of data points needed for each split
        if self.train_samples:
            train_data_points = self.train_samples
        else:
            # Use 80% of available data for train if not specified
            train_data_points = int(n * 0.8)
        
        # Step 2: Calculate val/test data points
        val_data_points = self._parse_sample_count(self.val_samples, train_data_points)
        test_data_points = self._parse_sample_count(self.test_samples, train_data_points)
        
        # Step 3: Total data points needed
        total_data_points_needed = train_data_points + val_data_points + test_data_points
        
        logger.info(f"Data points needed - Train: {train_data_points}, Val: {val_data_points}, Test: {test_data_points}")
        logger.info(f"Total data points needed: {total_data_points_needed}, Available: {n}")
        
        # Step 4: Check if we have enough data
        if total_data_points_needed > n:
            logger.error(f"Not enough data! Need {total_data_points_needed} but have {n}. Please reduce sample counts.")
            raise ValueError(f"Insufficient data: need {total_data_points_needed} points but only have {n}")
        
        # Step 5: Create sequential splits
        train_end = train_data_points
        val_end = train_end + val_data_points
        test_end = val_end + test_data_points
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:test_end]
        
        logger.info(f"Actual data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Create datasets
        stride = self.prediction_length // 2  # 50% overlap
        
        train_dataset = TimeSeriesDataset(
            train_df, self.target_column, self.context_length, 
            self.prediction_length, stride, self.time_interval_seconds, 
            mode="train", feature_columns=self.feature_columns
        )
        val_dataset = TimeSeriesDataset(
            val_df, self.target_column, self.context_length,
            self.prediction_length, self.prediction_length, self.time_interval_seconds, 
            mode="validation", feature_columns=self.feature_columns
        )
        test_dataset = TimeSeriesDataset(
            test_df, self.target_column, self.context_length,
            self.prediction_length, self.prediction_length, self.time_interval_seconds, 
            mode="test", feature_columns=self.feature_columns
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 2, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size * 2, shuffle=False, num_workers=4)
        
        return train_loader, val_loader, test_loader
    
    def _setup_model(self):
        """Setup model, optimizer, and scheduler."""
        logger.info(f"Loading base model: {self.base_model}")
        
        # Load model
        self.model = Toto.from_pretrained(self.base_model)
        self.model.to(self.device)
        
        # Setup trainer and forecaster
        self.trainer = TotoTrainer(self.model.model)
        self.forecaster = TotoForecaster(self.model.model)
        
        # Configure trainable parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        trainable_params = []
        trainable_count = 0
        for name, param in self.model.named_parameters():
            for layer_idx in self.trainable_layers:
                if f'layers.{layer_idx}' in name or 'output' in name:
                    param.requires_grad = True
                    trainable_params.append(param)
                    trainable_count += 1
                    break
        
        logger.info(f"Training layers: {self.trainable_layers} + output distribution layers ({trainable_count} parameters)")
        
        logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(trainable_params, lr=self.learning_rate, weight_decay=0.01)
        # ReduceLROnPlateau scheduler that monitors validation loss
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min',  # Minimize validation loss
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.min_lr
        )
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        epoch_maes = []
        
        for batch in tqdm(train_loader, desc="Training"):
            # Prepare batch
            n_variates = batch['n_variates'][0]  # All items in batch have same n_variates
            
            if n_variates == 1:
                # Univariate case - maintain backward compatibility
                context_values = torch.from_numpy(np.stack(batch['context_values'])).to(self.device)
                context_timestamps = torch.from_numpy(np.stack(batch['context_timestamps'])).to(self.device)
                target_values = torch.from_numpy(np.stack(batch['target_values'])).to(self.device)
                time_intervals = torch.as_tensor(batch['time_interval']).to(self.device)
                
                # Add variate dimension
                batch_size = context_values.shape[0]
                context_values = context_values.unsqueeze(1)
                context_timestamps = context_timestamps.unsqueeze(1)
                target_values = target_values.unsqueeze(1)
            else:
                # Multivariate case
                context_values = torch.from_numpy(np.stack(batch['context_values'])).to(self.device)
                target_values = torch.from_numpy(np.stack(batch['target_values'])).to(self.device)
                time_intervals = torch.as_tensor(batch['time_interval']).to(self.device)
                
                batch_size = context_values.shape[0]
                # context_values already has shape (batch, variates, time)
                # target_values has shape (batch, time) - add variate dimension
                target_values = target_values.unsqueeze(1)
                
                # Expand timestamps for all variates
                context_timestamps = torch.from_numpy(np.stack(batch['context_timestamps'])).to(self.device)
                context_timestamps = context_timestamps.unsqueeze(1).expand(-1, n_variates, -1)
            
            # Create MaskedTimeseries
            masked_ts = MaskedTimeseries(
                series=context_values,
                padding_mask=torch.ones_like(context_values, dtype=torch.bool),
                id_mask=torch.zeros_like(context_values, dtype=torch.int),
                timestamp_seconds=context_timestamps,
                time_interval_seconds=time_intervals.unsqueeze(1).expand(-1, n_variates) if n_variates > 1 else time_intervals.unsqueeze(1)
            )
            
            # Forward pass
            forecast = self.trainer.forecast(
                masked_ts,
                prediction_length=self.prediction_length,
                num_samples=None,
                use_kv_cache=False
            )
            
            # Compute loss
            if n_variates == 1:
                predictions = forecast.mean
            else:
                # For multivariate, extract only the target variable predictions (first variate)
                predictions = forecast.mean[:, 0:1, :]  # Keep dimension for consistency
            
            # Choose loss function
            if self.loss_fn == 'mse':
                loss = nn.functional.mse_loss(predictions, target_values)
            elif self.loss_fn == 'mae':
                loss = nn.functional.l1_loss(predictions, target_values)
            elif self.loss_fn == 'huber':
                loss = nn.functional.huber_loss(predictions, target_values)
            elif self.loss_fn == 'smooth_l1':
                loss = nn.functional.smooth_l1_loss(predictions, target_values)
            
            # Always compute both metrics for monitoring
            mse = nn.functional.mse_loss(predictions, target_values)
            mae = nn.functional.l1_loss(predictions, target_values)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            epoch_losses.append(mse.item())
            epoch_maes.append(mae.item())
        
        return {
            'mse': np.mean(epoch_losses),
            'mae': np.mean(epoch_maes)
        }
    
    @torch.no_grad()
    def _evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()
        losses = []
        maes = []
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Prepare batch
            n_variates = batch['n_variates'][0]
            
            if n_variates == 1:
                # Univariate case
                context_values = torch.from_numpy(np.stack(batch['context_values'])).to(self.device)
                context_timestamps = torch.from_numpy(np.stack(batch['context_timestamps'])).to(self.device)
                target_values = torch.from_numpy(np.stack(batch['target_values'])).to(self.device)
                time_intervals = torch.as_tensor(batch['time_interval']).to(self.device)
                
                # Add variate dimension
                context_values = context_values.unsqueeze(1)
                context_timestamps = context_timestamps.unsqueeze(1)
                target_values = target_values.unsqueeze(1)
            else:
                # Multivariate case
                context_values = torch.from_numpy(np.stack(batch['context_values'])).to(self.device)
                target_values = torch.from_numpy(np.stack(batch['target_values'])).to(self.device)
                time_intervals = torch.as_tensor(batch['time_interval']).to(self.device)
                
                # Handle shapes
                target_values = target_values.unsqueeze(1)
                context_timestamps = torch.from_numpy(np.stack(batch['context_timestamps'])).to(self.device)
                context_timestamps = context_timestamps.unsqueeze(1).expand(-1, n_variates, -1)
            
            # Create MaskedTimeseries
            masked_ts = MaskedTimeseries(
                series=context_values,
                padding_mask=torch.ones_like(context_values, dtype=torch.bool),
                id_mask=torch.zeros_like(context_values, dtype=torch.int),
                timestamp_seconds=context_timestamps,
                time_interval_seconds=time_intervals.unsqueeze(1).expand(-1, n_variates) if n_variates > 1 else time_intervals.unsqueeze(1)
            )
            
            # Forward pass
            forecast = self.forecaster.forecast(
                masked_ts,
                prediction_length=self.prediction_length,
                num_samples=None,
                use_kv_cache=True
            )
            
            # Compute metrics
            if n_variates == 1:
                predictions = forecast.mean
            else:
                # For multivariate, use only target variable predictions
                predictions = forecast.mean[:, 0:1, :]
            
            loss = nn.functional.mse_loss(predictions, target_values)
            mae = nn.functional.l1_loss(predictions, target_values)
            
            losses.append(loss.item())
            maes.append(mae.item())
        
        return {
            'loss': np.mean(losses),
            'mae': np.mean(maes),
            'rmse': np.sqrt(np.mean(losses))
        }
    
    def _plot_training_history(self):
        """Plot and save training history."""
        plt.figure(figsize=(12, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', alpha=0.8)
        if self.val_losses:
            epochs = [i * self.eval_every for i in range(len(self.val_losses))]
            plt.plot(epochs, self.val_losses, label='Val Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot
        plt.subplot(1, 2, 2)
        lrs = [self.optimizer.param_groups[0]['lr']] * len(self.train_losses)
        plt.plot(lrs)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / "plots" / "training_history.png", dpi=150)
        plt.close()
    
    def train(self):
        """Main training loop."""
        # Load data
        train_loader, val_loader, test_loader = self._load_data()
        
        # Setup model
        self._setup_model()
        
        # Try to load checkpoint
        checkpoint_loaded = self._load_checkpoint()
        
        if checkpoint_loaded:
            logger.info(f"Resuming training from epoch {self.start_epoch}, will train for {self.epochs} more epochs")
        else:
            logger.info("Starting training from scratch")
        
        # Training loop
        final_epoch = self.start_epoch
        target_epochs = self.start_epoch + self.epochs
        for epoch in range(self.start_epoch, target_epochs):
            final_epoch = epoch
            logger.info(f"\nEpoch {epoch + 1}/{target_epochs}")
            
            # Train
            train_metrics = self._train_epoch(train_loader)
            train_mse = train_metrics['mse']
            self.train_losses.append(train_mse)
            logger.info(f"Train MSE: {train_mse:.4f}, MAE: {train_metrics['mae']:.4f}, RMSE: {np.sqrt(train_mse):.4f} (optimizing {self.loss_fn})")
            
            # Evaluate
            if (epoch + 1) % self.eval_every == 0:
                val_metrics = self._evaluate(val_loader)
                val_loss = val_metrics['loss']
                self.val_losses.append(val_loss)
                
                logger.info(f"Val MSE: {val_loss:.4f}, MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
                
                # Update scheduler with validation loss
                old_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_loss)
                new_lr = self.optimizer.param_groups[0]['lr']
                if new_lr < old_lr:
                    logger.info(f"Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}")
                
                # Check for improvement
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Early stopping
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            else:
                is_best = False
            
            # Save checkpoint
            self._save_checkpoint(epoch, is_best)
            
            # Plot progress
            if (epoch + 1) % self.save_every == 0:
                self._plot_training_history()
        
        # Final evaluation
        logger.info("\nFinal evaluation on test set...")
        test_metrics = self._evaluate(test_loader)
        logger.info(f"Test MSE: {test_metrics['loss']:.4f}, MAE: {test_metrics['mae']:.4f}, RMSE: {test_metrics['rmse']:.4f}")
        
        # Save final results
        results = {
            'final_epoch': final_epoch + 1,
            'total_epochs_trained': final_epoch + 1 - (self.start_epoch - self.epochs),
            'best_val_loss': self.best_val_loss,
            'test_metrics': test_metrics,
            'model_dir': str(self.model_dir)
        }
        
        with open(self.model_dir / "final_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\nTraining complete! Model saved in {self.model_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Toto model with configuration')
    parser.add_argument('data_path', type=str, help='Path to CSV data file')
    parser.add_argument('--target-column', type=str, required=True, help='Target column name or index (e.g., "sales", 0, -1 for last column)')
    parser.add_argument('--context-length', type=int, default=2048, help='Context window length')
    parser.add_argument('--prediction-length', type=int, default=96, help='Prediction horizon length')
    parser.add_argument('--model-name', type=str, default='toto_base', help='Model name for directory')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--lr-factor', type=float, default=0.1, help='Factor to reduce LR by when plateau detected')
    parser.add_argument('--lr-patience', type=int, default=10, help='Number of epochs with no improvement before reducing LR')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--loss-fn', type=str, default='mse', choices=['mse', 'mae', 'huber', 'smooth_l1'], help='Loss function to use')
    parser.add_argument('--trainable-layers', type=int, nargs='+', default=[10, 11], help='Layers to train')
    parser.add_argument('--time-interval', type=int, default=3600, help='Time interval in seconds')
    parser.add_argument('--start-date', type=str, help='Start date for training data (YYYY-MM-DD HH:MM:SS)')
    parser.add_argument('--train-samples', type=int, help='Number of data points to use for training')
    parser.add_argument('--val-samples', type=str, default='10%', help='Validation samples (absolute number or percentage, default: 10%)')
    parser.add_argument('--test-samples', type=str, default='10%', help='Test samples (absolute number or percentage, default: 10%)')
    parser.add_argument('--feature-columns', nargs='*', help='Feature columns to use (names, indices, ranges, or count). Examples: "col1" "col2", 3, "0:5", -1 for all')
    
    args = parser.parse_args()
    
    # Parse feature columns if provided
    feature_columns = None
    if args.feature_columns:
        if len(args.feature_columns) == 1:
            # Single argument - could be a number, range, or single column name
            arg = args.feature_columns[0]
            try:
                # Try to parse as integer
                feature_columns = int(arg)
            except ValueError:
                # Not an integer, keep as string
                feature_columns = arg
        else:
            # Multiple arguments - list of column names
            feature_columns = args.feature_columns
    
    # Create pipeline
    pipeline = TotoTrainingPipeline(
        data_path=args.data_path,
        target_column=args.target_column,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        model_name=args.model_name,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        epochs=args.epochs,
        loss_fn=args.loss_fn,
        trainable_layers=args.trainable_layers,
        time_interval_seconds=args.time_interval,
        start_date=args.start_date,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        feature_columns=feature_columns
    )
    
    # Run training
    pipeline.train()


if __name__ == "__main__":
    main()