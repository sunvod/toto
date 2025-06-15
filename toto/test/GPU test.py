import torch
import time

# Test più intensivo per le bestie che hai
print("🚀 Stress test RTX A6000/A4000...")

# Matrice grande per sfruttare i 49GB della A6000
device = torch.device('cuda:0')  # RTX A6000
x = torch.randn(8192, 8192, dtype=torch.float32).to(device)
y = torch.randn(8192, 8192, dtype=torch.float32).to(device)

start = time.time()
z = torch.matmul(x, y)
torch.cuda.synchronize()
end = time.time()

print(f"🔥 Matrix 8K x 8K completata in: {end-start:.3f} secondi")
print(f"🔥 Memoria GPU utilizzata: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print(f"🔥 TFLOPS teorici: {(2 * 8192**3) / (end-start) / 1e12:.2f}")