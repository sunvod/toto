# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2025 Datadog, Inc.

"""
Joint training module for Toto model on multiple datasets/tasks simultaneously.

This module implements:
- Multi-dataset loading with different frequencies and characteristics
- Joint training with shared model parameters
- Task-specific evaluation metrics
- Adaptive sampling strategies for balanced training
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import yaml
from tqdm import tqdm
import wandb

from toto.data.util.dataset import MaskedTimeseries, pad_array, pad_id_mask
from toto.model.toto import Toto
from toto.training.train import TotoTrainer
from toto.inference.forecaster import TotoForecaster, Forecast


logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for a single dataset in joint training."""
    name: str
    path: str
    target_column: str
    feature_columns: List[str] = field(default_factory=list)
    context_length: int = 2048
    prediction_length: int = 96
    stride: int = 24
    time_interval_seconds: int = 3600  # Default 1 hour
    dataset_id: int = 0  # Unique ID for this dataset
    weight: float = 1.0  # Sampling weight for this dataset
    normalize: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class JointTrainingConfig:
    """Configuration for joint training."""
    datasets: List[DatasetConfig]
    model_name: str = "Datadog/Toto-Open-Base-1.0"
    batch_size: int = 32
    learning_rate: float = 1e-4
    min_lr: float = 1e-6
    epochs: int = 100
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    weight_decay: float = 0.01
    use_mixed_precision: bool = True
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_every: int = 5
    eval_every: int = 1
    patience: int = 10
    use_wandb: bool = False
    wandb_project: str = "toto-joint-training"
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    optimizer: str = "adamw"  # "adamw", "sgd", "adam"
    scheduler: str = "cosine"  # "cosine", "onecycle", "reduce_on_plateau"
    loss_fn: str = "mse"  # "mse", "mae", "huber", "quantile"
    quantile_levels: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    freeze_backbone: bool = False
    freeze_layers: List[int] = field(default_factory=list)
    trainable_layers: List[int] = field(default_factory=lambda: [10, 11])  # Last layers by default


class MultiDatasetTimeSeries(Dataset):
    """Dataset that handles multiple time series datasets for joint training."""
    
    def __init__(
        self, 
        config: DatasetConfig,
        df: pd.DataFrame,
        mode: str = "train"
    ):
        self.config = config
        self.df = df
        self.mode = mode
        self.total_length = config.context_length + config.prediction_length
        
        # Calculate valid starting positions
        self.valid_starts = []
        for i in range(0, len(df) - self.total_length + 1, config.stride):
            self.valid_starts.append(i)
        
        # Compute normalization statistics if needed
        if config.normalize and mode == "train":
            self.compute_stats()
        elif config.normalize:
            # Load pre-computed stats for val/test
            self.load_stats()
        
        logger.info(f"Created {mode} dataset '{config.name}' with {len(self.valid_starts)} samples")
    
    def compute_stats(self):
        """Compute normalization statistics from training data."""
        all_columns = [self.config.target_column] + self.config.feature_columns
        self.stats = {}
        
        for col in all_columns:
            values = self.df[col].values
            self.stats[col] = {
                'mean': np.nanmean(values),
                'std': np.nanstd(values) + 1e-6  # Avoid division by zero
            }
        
        # Save stats for val/test sets
        stats_path = Path(self.config.path).parent / f"{self.config.name}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f)
    
    def load_stats(self):
        """Load pre-computed normalization statistics."""
        stats_path = Path(self.config.path).parent / f"{self.config.name}_stats.json"
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)
    
    def normalize_values(self, values: np.ndarray, column: str) -> np.ndarray:
        """Normalize values using pre-computed statistics."""
        if not self.config.normalize:
            return values
        
        stats = self.stats[column]
        return (values - stats['mean']) / stats['std']
    
    def denormalize_values(self, values: np.ndarray, column: str) -> np.ndarray:
        """Denormalize values back to original scale."""
        if not self.config.normalize:
            return values
        
        stats = self.stats[column]
        return values * stats['std'] + stats['mean']
    
    def __len__(self):
        return len(self.valid_starts)
    
    def __getitem__(self, idx):
        start_idx = self.valid_starts[idx]
        end_idx = start_idx + self.total_length
        
        # Get the window of data
        window = self.df.iloc[start_idx:end_idx]
        
        # Split into context and target
        context = window.iloc[:self.config.context_length]
        target = window.iloc[self.config.context_length:]
        
        # Prepare all columns (target + features)
        all_columns = [self.config.target_column] + self.config.feature_columns
        num_variates = len(all_columns)
        
        # Extract and normalize values
        context_values = []
        target_values = []
        
        for col in all_columns:
            ctx_vals = self.normalize_values(context[col].values.astype(np.float32), col)
            tgt_vals = self.normalize_values(target[col].values.astype(np.float32), col)
            context_values.append(ctx_vals)
            target_values.append(tgt_vals)
        
        context_values = np.stack(context_values)  # (variates, context_length)
        target_values = np.array(target_values[0])  # Only target column for loss
        
        # Extract timestamps
        context_timestamps = context['timestamp_seconds'].values.astype(np.int64)
        
        # Create ID mask for this dataset
        id_mask = np.full((num_variates, 1), self.config.dataset_id, dtype=np.int32)
        
        return {
            'context_values': context_values,
            'context_timestamps': context_timestamps,
            'target_values': target_values,
            'id_mask': id_mask,
            'time_interval': self.config.time_interval_seconds,
            'dataset_name': self.config.name,
            'dataset_id': self.config.dataset_id,
            'num_variates': num_variates
        }


class JointTrainingModule:
    """Main module for joint training of Toto on multiple datasets."""
    
    def __init__(self, config: JointTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds
        self.set_seed(config.seed)
        
        # Setup directories
        self.setup_directories()
        
        # Initialize logging
        self.setup_logging()
        
        # Load model
        self.load_model()
        
        # Setup optimizer and scheduler
        self.setup_optimization()
        
        # Initialize best metrics
        self.best_metrics = defaultdict(lambda: float('inf'))
        self.patience_counter = 0
        
        # Setup mixed precision if enabled
        if config.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Initialize wandb if enabled
        if config.use_wandb:
            self.init_wandb()
    
    def set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def setup_directories(self):
        """Create necessary directories."""
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = Path(self.config.log_dir) / "joint_training.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def load_model(self):
        """Load pre-trained Toto model and setup for training."""
        logger.info(f"Loading model: {self.config.model_name}")
        self.model = Toto.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        
        # Setup trainer and forecaster
        self.trainer = TotoTrainer(self.model.model)
        self.forecaster = TotoForecaster(self.model.model)
        
        # Configure trainable parameters
        self.configure_trainable_params()
    
    def configure_trainable_params(self):
        """Configure which parameters to train based on config."""
        # First, freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze specific layers
        trainable_params = []
        for name, param in self.model.named_parameters():
            should_train = False
            
            # Check if in trainable layers
            for layer_idx in self.config.trainable_layers:
                if f'layers.{layer_idx}' in name:
                    should_train = True
                    break
            
            # Always train output layers
            if 'output' in name or 'head' in name:
                should_train = True
            
            # Check if should be frozen
            for layer_idx in self.config.freeze_layers:
                if f'layers.{layer_idx}' in name:
                    should_train = False
                    break
            
            if should_train and not self.config.freeze_backbone:
                param.requires_grad = True
                trainable_params.append(param)
                logger.info(f"Training parameter: {name}")
        
        self.trainable_params = trainable_params
        logger.info(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    def setup_optimization(self):
        """Setup optimizer and learning rate scheduler."""
        # Optimizer
        if self.config.optimizer == "adamw":
            self.optimizer = AdamW(
                self.trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adam":
            self.optimizer = torch.optim.Adam(
                self.trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.trainable_params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        
        # Scheduler
        if self.config.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler == "onecycle":
            steps_per_epoch = 100  # Will be updated when dataloaders are created
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=self.config.warmup_epochs / self.config.epochs
            )
    
    def init_wandb(self):
        """Initialize Weights & Biases logging."""
        wandb.init(
            project=self.config.wandb_project,
            config=self.config.__dict__,
            name=f"joint_training_{len(self.config.datasets)}_datasets"
        )
        wandb.watch(self.model, log="all")
    
    def load_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load all datasets and create joint dataloaders."""
        train_datasets = []
        val_datasets = []
        test_datasets = []
        dataset_weights = []
        
        for dataset_config in self.config.datasets:
            # Load CSV
            logger.info(f"Loading dataset: {dataset_config.name}")
            df = pd.read_csv(dataset_config.path)
            
            # Ensure timestamp column exists
            if 'timestamp_seconds' not in df.columns:
                if 'Data' in df.columns:
                    df['Data'] = pd.to_datetime(df['Data'])
                    df['timestamp_seconds'] = (df['Data'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
                elif 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df['timestamp_seconds'] = (df['timestamp'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
                else:
                    # Assume hourly data if no timestamp
                    df['timestamp_seconds'] = np.arange(len(df)) * dataset_config.time_interval_seconds
            
            # Sort by time
            df = df.sort_values('timestamp_seconds')
            
            # Split data
            n = len(df)
            train_end = int(n * dataset_config.train_split)
            val_end = train_end + int(n * dataset_config.val_split)
            
            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]
            
            # Create datasets
            train_ds = MultiDatasetTimeSeries(dataset_config, train_df, mode="train")
            val_ds = MultiDatasetTimeSeries(dataset_config, val_df, mode="val")
            test_ds = MultiDatasetTimeSeries(dataset_config, test_df, mode="test")
            
            train_datasets.append(train_ds)
            val_datasets.append(val_ds)
            test_datasets.append(test_ds)
            
            # Add weights for sampling
            dataset_weights.extend([dataset_config.weight] * len(train_ds))
        
        # Create concatenated datasets
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
        test_dataset = ConcatDataset(test_datasets)
        
        # Create weighted sampler for balanced training
        sampler = WeightedRandomSampler(
            weights=dataset_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=self.config.persistent_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size * 2,  # Can use larger batch for validation
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        # Update scheduler steps if using OneCycle
        if self.config.scheduler == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.epochs,
                steps_per_epoch=len(train_loader),
                pct_start=self.config.warmup_epochs / self.config.epochs
            )
        
        logger.info(f"Created dataloaders - Train: {len(train_loader)} batches, "
                   f"Val: {len(val_loader)} batches, Test: {len(test_loader)} batches")
        
        return train_loader, val_loader, test_loader
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss based on configured loss function."""
        if self.config.loss_fn == "mse":
            return nn.functional.mse_loss(predictions, targets)
        elif self.config.loss_fn == "mae":
            return nn.functional.l1_loss(predictions, targets)
        elif self.config.loss_fn == "huber":
            return nn.functional.huber_loss(predictions, targets)
        elif self.config.loss_fn == "quantile":
            # Quantile loss for multiple quantiles
            losses = []
            for q in self.config.quantile_levels:
                errors = targets - predictions
                losses.append(torch.max(q * errors, (q - 1) * errors).mean())
            return torch.stack(losses).mean()
        else:
            raise ValueError(f"Unknown loss function: {self.config.loss_fn}")
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step."""
        # Move batch to device
        context_values = batch['context_values'].to(self.device)
        context_timestamps = batch['context_timestamps'].to(self.device)
        target_values = batch['target_values'].to(self.device)
        id_masks = batch['id_mask'].to(self.device)
        time_intervals = batch['time_interval'].to(self.device)
        
        # Expand id_mask to match time dimension
        batch_size, num_variates, _ = id_masks.shape
        context_length = context_values.shape[-1]
        id_masks_expanded = id_masks.expand(-1, -1, context_length)
        
        # Create MaskedTimeseries
        masked_ts = MaskedTimeseries(
            series=context_values,
            padding_mask=torch.ones_like(context_values, dtype=torch.bool),
            id_mask=id_masks_expanded,
            timestamp_seconds=context_timestamps.unsqueeze(1).expand(-1, num_variates, -1),
            time_interval_seconds=time_intervals.unsqueeze(1).expand(-1, num_variates)
        )
        
        # Forward pass with mixed precision
        if self.config.use_mixed_precision:
            with torch.cuda.amp.autocast():
                forecast = self.trainer.forecast(
                    masked_ts,
                    prediction_length=target_values.shape[-1],
                    num_samples=None,  # Use mean for training
                    use_kv_cache=False
                )
                predictions = forecast.mean[:, 0, :]  # Extract target variable
                loss = self.compute_loss(predictions, target_values)
        else:
            forecast = self.trainer.forecast(
                masked_ts,
                prediction_length=target_values.shape[-1],
                num_samples=None,
                use_kv_cache=False
            )
            predictions = forecast.mean[:, 0, :]
            loss = self.compute_loss(predictions, target_values)
        
        # Compute additional metrics
        with torch.no_grad():
            mae = nn.functional.l1_loss(predictions, target_values).item()
            mse = nn.functional.mse_loss(predictions, target_values).item()
            rmse = np.sqrt(mse)
        
        return {
            'loss': loss,
            'mae': mae,
            'mse': mse,
            'rmse': rmse
        }
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, phase: str = "val") -> Dict[str, float]:
        """Evaluate model on a dataset."""
        self.model.eval()
        
        # Metrics per dataset
        dataset_metrics = defaultdict(lambda: defaultdict(list))
        all_losses = []
        
        for batch in tqdm(dataloader, desc=f"Evaluating {phase}"):
            # Get dataset names from batch
            dataset_names = batch['dataset_name']
            
            # Forward pass
            metrics = self.train_step(batch)
            all_losses.append(metrics['loss'].item())
            
            # Accumulate metrics per dataset
            for i, dataset_name in enumerate(dataset_names):
                for metric_name, metric_value in metrics.items():
                    if metric_name != 'loss':  # Loss is a tensor
                        dataset_metrics[dataset_name][metric_name].append(metric_value)
        
        # Aggregate metrics
        results = {
            f'{phase}/loss': np.mean(all_losses)
        }
        
        # Per-dataset metrics
        for dataset_name, metrics in dataset_metrics.items():
            for metric_name, values in metrics.items():
                avg_value = np.mean(values)
                results[f'{phase}/{dataset_name}/{metric_name}'] = avg_value
        
        # Overall metrics
        for metric_name in ['mae', 'mse', 'rmse']:
            all_values = []
            for dataset_metrics_dict in dataset_metrics.values():
                all_values.extend(dataset_metrics_dict[metric_name])
            if all_values:
                results[f'{phase}/{metric_name}'] = np.mean(all_values)
        
        return results
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint with val_loss: {metrics['val/loss']:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['metrics']
    
    def train(self):
        """Main training loop."""
        # Load datasets
        train_loader, val_loader, test_loader = self.load_datasets()
        
        # Training loop
        for epoch in range(self.config.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            
            # Training phase
            self.model.train()
            train_metrics = defaultdict(list)
            
            pbar = tqdm(train_loader, desc="Training")
            for batch in pbar:
                # Training step
                step_metrics = self.train_step(batch)
                loss = step_metrics['loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                
                if self.config.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, self.config.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, self.config.gradient_clip)
                    self.optimizer.step()
                
                # Update scheduler if OneCycle
                if self.config.scheduler == "onecycle":
                    self.scheduler.step()
                
                # Accumulate metrics
                for k, v in step_metrics.items():
                    if k == 'loss':
                        train_metrics[k].append(v.item())
                    else:
                        train_metrics[k].append(v)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
            
            # Average training metrics
            train_results = {f'train/{k}': np.mean(v) for k, v in train_metrics.items()}
            
            # Validation phase
            if (epoch + 1) % self.config.eval_every == 0:
                val_results = self.evaluate(val_loader, phase="val")
                
                # Combine results
                epoch_metrics = {**train_results, **val_results}
                
                # Log metrics
                self.log_metrics(epoch_metrics, epoch)
                
                # Check for improvement
                val_loss = val_results['val/loss']
                is_best = val_loss < self.best_metrics['val_loss']
                if is_best:
                    self.best_metrics['val_loss'] = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Save checkpoint
                if (epoch + 1) % self.config.save_every == 0 or is_best:
                    self.save_checkpoint(epoch + 1, epoch_metrics, is_best)
                
                # Early stopping
                if self.patience_counter >= self.config.patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Update scheduler if Cosine
            if self.config.scheduler == "cosine":
                self.scheduler.step()
        
        # Final evaluation on test set
        logger.info("\nFinal evaluation on test set...")
        test_results = self.evaluate(test_loader, phase="test")
        self.log_metrics(test_results, self.config.epochs)
        
        # Save final model
        final_path = Path(self.config.checkpoint_dir) / "final_model.pt"
        torch.save(self.model.state_dict(), final_path)
        logger.info(f"Training complete! Final model saved to {final_path}")
    
    def log_metrics(self, metrics: Dict[str, float], epoch: int):
        """Log metrics to console and wandb."""
        # Console logging
        logger.info(f"Epoch {epoch} metrics:")
        for k, v in sorted(metrics.items()):
            logger.info(f"  {k}: {v:.4f}")
        
        # Wandb logging
        if self.config.use_wandb:
            wandb.log(metrics, step=epoch)


def main():
    """Example usage of joint training module."""
    # Example configuration
    config = JointTrainingConfig(
        datasets=[
            DatasetConfig(
                name="electricity",
                path="data/electricity.csv",
                target_column="OT",
                feature_columns=["MT_001", "MT_002", "MT_003"],
                context_length=2048,
                prediction_length=96,
                dataset_id=0,
                weight=1.0
            ),
            DatasetConfig(
                name="temperature",
                path="data/temperature.csv",
                target_column="OT",
                feature_columns=[],
                context_length=2048,
                prediction_length=96,
                dataset_id=1,
                weight=0.5
            ),
            DatasetConfig(
                name="venezia",
                path="data/venezia.csv",
                target_column="Livello Punta Salute (cm)",
                feature_columns=[],
                context_length=1024,
                prediction_length=48,
                dataset_id=2,
                weight=0.8
            )
        ],
        batch_size=16,
        learning_rate=5e-5,
        epochs=50,
        use_wandb=True
    )
    
    # Create trainer and start training
    trainer = JointTrainingModule(config)
    trainer.train()


if __name__ == "__main__":
    main()