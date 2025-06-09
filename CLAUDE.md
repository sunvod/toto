# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Toto is a foundation model for multivariate time series forecasting with a focus on observability metrics. The repository also hosts code for evaluating time series models on BOOM (Benchmark of Observability Metrics), a large-scale forecasting dataset of real-world observability data.

## Development Environment Setup

```bash
# Clone the repository
git clone https://github.com/DataDog/toto.git
cd toto

# Optional: create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

For optimal inference speed, installing these additional dependencies is recommended:
- xformers: https://github.com/facebookresearch/xformers
- flash-attention: https://github.com/Dao-AILab/flash-attention

## Common Commands

### Running Tests

```bash
# Set environment variables for reproducible results
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$(pwd):$(pwd)/toto:$PYTHONPATH"

# Run all tests
pytest

# Run a specific test file
pytest toto/test/path/to/test_file.py

# Run a specific test
pytest toto/test/path/to/test_file.py::test_function_name

# Run tests with CUDA markers
pytest -m cuda
```

### Code Formatting and Linting

```bash
# Format code with black
black .

# Sort imports with isort
isort .

# Run type checking with mypy
mypy .
```

## Evaluation Commands

### LSF Evaluation

```bash
# Set environment variables
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$(pwd):$(pwd)/toto:$PYTHONPATH"

# Run LSF evaluation for a specific dataset
python toto/evaluation/run_lsf_eval.py \
    --datasets ETTh1 \
    --context-length 2048 \
    --eval-stride 1 \
    --checkpoint-path [CHECKPOINT-NAME-OR-DIR]

# Get help for evaluation options
python toto/evaluation/run_lsf_eval.py --help
```

### BOOM Evaluation

For BOOM evaluation, refer to the Jupyter notebooks in the `boom/notebooks/` directory, particularly `boom/notebooks/toto.ipynb`.

### GIFT-Eval Evaluation

For GIFT-Eval evaluation, refer to the notebook at `toto/evaluation/gift_eval/toto.ipynb`.

## Architecture Overview

Toto uses a decoder-only transformer architecture with several key components:

1. **Model Structure** (`toto/model/`):
   - `toto.py`: Main model implementation
   - `transformer.py`: Transformer implementation
   - `attention.py`: Attention mechanisms
   - `backbone.py`: Core model structure
   - `distribution.py`: Probabilistic output distribution
   - `embedding.py`, `feed_forward.py`, `rope.py`: Supporting components
   - `scaler.py`: Time series scaling

2. **Inference** (`toto/inference/`):
   - `forecaster.py`: Main forecasting interface
   - `gluonts_predictor.py`: Integration with GluonTS

3. **Evaluation** (`toto/evaluation/`):
   - LSF evaluation: `run_lsf_eval.py`
   - GIFT-Eval: `gift_eval/`
   - BOOM benchmark: `boom/`

4. **Data Handling** (`toto/data/`):
   - `util/dataset.py`: Dataset utilities and MaskedTimeseries implementation

## Contributing

1. For bug fixes, claim an existing issue labeled as `type/bug`.
2. For new features, open an issue with the title "[RFC] Title of your proposal" and wait for approval (tag `rfc/approved`) before starting work.
3. Follow the code style of the project (black formatting, isort for imports).
4. Include tests for new functionality.
5. See CONTRIBUTING.md for more details.

## System Requirements

- Python 3.10+
- PyTorch 2.5+
- CUDA-capable device (Ampere generation or newer recommended)