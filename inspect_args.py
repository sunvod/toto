import torch

checkpoint = torch.load("toto_venezia_simple_finetuned.pt", map_location='cpu', weights_only=False)
args = checkpoint['args']

print("Parametri del modello salvati:")
for attr in dir(args):
    if not attr.startswith('_'):
        value = getattr(args, attr)
        print(f"  {attr}: {value}")