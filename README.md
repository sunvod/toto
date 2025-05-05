# Toto - Time Series Optimized Transformer for Observability
[Paper](#) | [Toto Model Card](#) | [BOOM Dataset](#)

Toto is a foundation model for multivariate time series forecasting with a focus on observability metrics. This model leverages innovative architectural designs to efficiently handle the high-dimensional, complex time series that are characteristic of observability data.

## Features

- **Zero-Shot Forecasting**: Perform forecasting without fine-tuning on your specific time series
- **State-of-the-Art Performance**: Achieves top scores in benchmarks covering diverse time series forecasting tasks. This includes the established multi-domain benchmark [GiftEval](https://huggingface.co/spaces/Salesforce/GIFT-Eval), as well as our own observability-focused benchmark
[BOOM](https://huggingface.co/datasets/Datadog/BOOM).
- **Multi-Variate Support**: Efficiently process multiple variables using Proportional Factorized Space-Time Attention
- **Probabilistic Predictions**: Generate both point forecasts and uncertainty estimates using a Student-T mixture model
- **High-Dimensional Support**: Handle time series with a large number of variables efficiently
- **Decoder-Only Architecture**: Support for variable prediction horizons and context lengths
- **Pre-trained on Massive Data**: Trained on over 2 trillion time series data points, the largest pretraiing dataset for any open-weights time series foundation model to date. Note: this open-source, open-weights model was training **without** any customer data.


## Model Weights

Toto-Open, the open-weights release of Toto, is available on Hugging Face. Currently available checkpoints:

| Checkpoint | Parameters | Notes |
|------------|------------|-------|
| [Toto-Open-Base-1.0](https://huggingface.co/Datadog/Toto-Open-Base-1.0) | 151M | The initial open relase of Toto. Achieves state-of-the-art performance on both general-purpose and observability-focused benchmarking tasks, as described in our paper. |



## Installation

```bash
# Clone the repository
git clone https://github.com/DataDog/toto.git
cd toto

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Here's a simple example to get you started with forecasting:

```python
import torch
from data.util.dataset import MaskedTimeseries
from inference.forecaster import TotoForecaster
from model.toto import Toto

# Load the pre-trained model
toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
toto.to('cuda')  # Move to GPU

# Optionally compile the model for faster inference
toto.compile()  # Uses Torch's JIT compilation for better performance

forecaster = TotoForecaster(toto.model)

# Prepare your input time series (channels, time_steps)
input_series = torch.randn(7, 4096).to('cuda')  # Example with 7 variables and 4096 timesteps

# Prepare timestamp information (optional, but expected by API; not used by the current model release)
timestamp_seconds = torch.zeros(7, 4096).to('cuda')
time_interval_seconds = torch.full((7,), 60*15).to('cuda')  # 15-minute intervals

# Create a MaskedTimeseries object
inputs = MaskedTimeseries(
    series=input_series,
    padding_mask=torch.full_like(input_series, True, dtype=torch.bool),
    id_mask=torch.zeros_like(input_series),
    timestamp_seconds=timestamp_seconds,
    time_interval_seconds=time_interval_seconds,
)

# Generate forecasts for the next 336 timesteps
forecast = forecaster.forecast(
    inputs,
    prediction_length=336,
    num_samples=256,  # Number of samples for probabilistic forecasting
    samples_per_batch=256,  # Control memory usage during inference
)

# Access results
mean_prediction = forecast.mean  # Point forecasts
prediction_samples = forecast.samples  # Probabilistic samples
lower_quantile = forecast.quantile(0.1)  # 10th percentile for lower confidence bound
upper_quantile = forecast.quantile(0.9)  # 90th percentile for upper confidence bound
```

## Tutorials

For a comprehensive guide on using Toto for time series forecasting, check out our tutorial notebooks:

- [Basic Inference Tutorial](toto/notebooks/inference_tutorial.ipynb): Learn how to load the model and make forecasts

## Data

Toto was trained on a diverse mixture of time series datasets:
- [GiftEval Pretrain](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain)
- [Chronos](https://huggingface.co/datasets/autogluon/chronos_datasets)
- Synthetic data for robustness

## Requirements

- Python 3.10+
- PyTorch 2.5+
- CUDA-capable device (Ampere generation or newer recommended for optimal performance)

## Citation

If you use Toto in your research, please cite our work:

TODO: add arxiv BibTeX

## License
Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License - see [LICENSE](LICENSE) file for details.

This product includes software developed at Datadog (https://www.datadoghq.com/) Copyright 2025 Datadog, Inc.

## Contributing

We welcome contributions! Please check out our [contributing guidelines](CONTRIBUTING.md) to get started.
