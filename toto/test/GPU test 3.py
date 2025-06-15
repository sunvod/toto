import torch
import time

print("💀 APOCALYPSE TEST - Spingendo al limite...")

# Usiamo ~7-8GB (15% dei 49GB)
device = torch.device('cuda:0')
size = 23000  # ~23K x 23K ≈ 8GB

print(f"💀 Creando matrici {size}x{size} (~8GB)...")
try:
  x = torch.randn(size, size, dtype=torch.float32).to(device)
  y = torch.randn(size, size, dtype=torch.float32).to(device)

  start = time.time()
  z = torch.matmul(x, y)
  torch.cuda.synchronize()
  end = time.time()

  print(f"💀 Matrix {size}x{size} in: {end-start:.3f} secondi")
  print(f"💀 TFLOPS: {(2 * size**3) / (end-start) / 1e12:.2f}")
  print(f"💀 Memoria: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
  print(f"💀 Utilizzo: {torch.cuda.memory_allocated(0) / (49*1024**3) * 100:.1f}% dei 49GB")
  print("💀 RTX A6000 = INARRESTABILE! 🔥")

except Exception as e:
  print(f"💀 Limite raggiunto: {e}")