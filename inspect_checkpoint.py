import torch
import sys

# Carica il checkpoint
checkpoint_path = "toto_venezia_simple_finetuned.pt" if len(sys.argv) < 2 else sys.argv[1]
print(f"Ispezione di: {checkpoint_path}\n")

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print(f"Tipo: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"\nChiavi nel checkpoint:")
        for key in checkpoint.keys():
            value = checkpoint[key]
            if isinstance(value, torch.Tensor):
                print(f"  {key}: Tensor di shape {value.shape}")
            elif isinstance(value, dict):
                print(f"  {key}: Dizionario con {len(value)} chiavi")
            else:
                print(f"  {key}: {type(value).__name__}")
    
    elif hasattr(checkpoint, '__class__'):
        print(f"\nClasse del modello: {checkpoint.__class__.__name__}")
        if hasattr(checkpoint, '__module__'):
            print(f"Modulo: {checkpoint.__module__}")
            
    # Se è un argparse.Namespace
    if type(checkpoint).__name__ == 'Namespace':
        print("\nÈ un argparse.Namespace con attributi:")
        for attr in dir(checkpoint):
            if not attr.startswith('_'):
                print(f"  {attr}: {getattr(checkpoint, attr)}")
                
except Exception as e:
    print(f"Errore: {e}")