#!/usr/bin/env python3
"""
Example of how to load and use a fine-tuned Toto model.
"""

import os
import sys
import torch

# Set up environment
project_root = os.path.dirname(os.path.abspath(__file__))
toto_path = os.path.join(project_root, "toto")
sys.path.insert(0, project_root)
sys.path.insert(0, toto_path)

from model.toto import Toto
from inference.forecaster import TotoForecaster

def load_finetuned_model(checkpoint_path, device='cuda'):
    """Load a fine-tuned Toto model from checkpoint."""
    
    # First, load the base model architecture
    print("Loading base Toto architecture...")
    model = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
    
    # Load the checkpoint
    print(f"Loading fine-tuned weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load the fine-tuned weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device
    model.to(device)
    model.eval()
    
    # Print training info
    print(f"Model was fine-tuned for {checkpoint['epoch']} epochs")
    print(f"Final validation loss: {checkpoint['val_losses'][-1]:.4f}")
    
    return model, checkpoint

# Example usage
if __name__ == "__main__":
    # Load the fine-tuned model
    model, checkpoint = load_finetuned_model('toto_venezia_finetuned.pt')
    
    # Now you can use it for forecasting
    forecaster = TotoForecaster(model.model)
    
    print("\nModel loaded successfully!")
    print("You can now use this model with run_csv_test.py or in your own code.")