import torch
import time

print("🔥 Test EXTREME - Sfruttando i 49GB...")

# Matrice GIGANTE per utilizzare più memoria
device = torch.device('cuda:0')  # RTX A6000
size = 16384  # 16K x 16K = ~1GB per matrice

print(f"🔥 Creando matrici {size}x{size}...")
x = torch.randn(size, size, dtype=torch.float32).to(device)
y = torch.randn(size, size, dtype=torch.float32).to(device)

start = time.time()
z = torch.matmul(x, y)
torch.cuda.synchronize()
end = time.time()

print(f"🔥 Matrix {size}K x {size}K in: {end-start:.3f} secondi")
print(f"🔥 Memoria GPU: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"🔥 TFLOPS: {(2 * size**3) / (end-start) / 1e12:.2f}")
print(f"🔥 Utilizzo memoria: {torch.cuda.memory_allocated(0) / (49*1024**3) * 100:.1f}% dei 49GB")