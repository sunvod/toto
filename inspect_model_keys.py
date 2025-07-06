import torch

checkpoint = torch.load("toto_venezia_simple_finetuned.pt", map_location='cpu', weights_only=False)
state_dict = checkpoint['model_state_dict']

print("Prime 20 chiavi del modello:")
for i, key in enumerate(list(state_dict.keys())[:20]):
    print(f"  {key}: {state_dict[key].shape}")
    
print(f"\n... e altre {len(state_dict) - 20} chiavi")

# Verifica se sembra un modello Toto
toto_keys = [k for k in state_dict.keys() if 'toto' in k.lower() or 'patch_embed' in k or 'transformer' in k]
if toto_keys:
    print(f"\nChiavi che sembrano appartenere a Toto: {toto_keys[:5]}")
else:
    print("\nNon sembra essere un modello Toto")