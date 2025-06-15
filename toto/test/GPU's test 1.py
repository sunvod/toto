import torch
import time

print("🔧 DIAGNOSTIC TEST - Risolviamo il mistero!")

def robust_gpu_test(device_id, gpu_name):
  device = torch.device(f'cuda:{device_id}')
  print(f"\n🔍 Diagnostica {gpu_name} (GPU {device_id})")

  # Clear cache prima del test
  torch.cuda.empty_cache()

  # Test più semplice ma robusto
  size = 4096  # Matrice più piccola per essere sicuri

  # Alloca memoria
  print(f"📱 Allocando matrici {size}x{size}...")
  x = torch.randn(size, size, dtype=torch.float32, device=device)
  y = torch.randn(size, size, dtype=torch.float32, device=device)

  # Warm-up per evitare effetti di caching
  print("🔥 Warm-up...")
  for i in range(3):
    _ = torch.matmul(x, y)
    torch.cuda.synchronize(device)

  # Test multipli per verificare consistenza
  times = []
  print("⚡ Test multipli...")

  for i in range(5):
    torch.cuda.synchronize(device)
    start = time.time()

    result = torch.matmul(x, y)

    torch.cuda.synchronize(device)
    end = time.time()

    times.append(end - start)
    print(f"   Run {i+1}: {times[-1]:.6f}s")

  # Statistiche
  avg_time = sum(times) / len(times)
  min_time = min(times)
  max_time = max(times)

  tflops = (2 * size**3) / avg_time / 1e12
  memory_used = torch.cuda.memory_allocated(device_id) / 1024**3

  print(f"📊 RISULTATI {gpu_name}:")
  print(f"   ⚡ Tempo medio: {avg_time:.6f}s")
  print(f"   ⚡ Tempo min: {min_time:.6f}s")
  print(f"   ⚡ Tempo max: {max_time:.6f}s")
  print(f"   🔥 TFLOPS: {tflops:.2f}")
  print(f"   💾 Memoria: {memory_used:.2f} GB")
  print(f"   📈 Consistenza: {(max_time-min_time)/avg_time*100:.1f}% variazione")

  # Verifica che il calcolo sia realmente fatto
  checksum = torch.sum(result).item()
  print(f"   ✅ Checksum: {checksum:.2e} (verifica calcolo reale)")

  torch.cuda.empty_cache()
  return avg_time, tflops

# Test diagnostico
print("🚀 Iniziando diagnostic test...")
a6000_time, a6000_tflops = robust_gpu_test(0, "RTX A6000")
a4000_time, a4000_tflops = robust_gpu_test(1, "RTX A4000")

print(f"\n🏆 CONFRONTO REALE:")
print(f"RTX A6000: {a6000_tflops:.2f} TFLOPS")
print(f"RTX A4000: {a4000_tflops:.2f} TFLOPS")

if a4000_tflops > 100:  # Ancora impossibile
  print("🚨 A4000 mostra ancora risultati impossibili!")
  print("🔧 Possibili cause:")
  print("   - Driver NVIDIA differenti")
  print("   - Problemi CUDA context")
  print("   - Hardware issue")
  print("   - Caching aggressivo")
else:
  print("✅ Risultati ora realistici!")

# Test aggiuntivo: verifica driver e CUDA info
print(f"\n🔍 INFO TECNICHE:")
print(f"PyTorch CUDA version: {torch.version.cuda}")
for i in range(2):
  props = torch.cuda.get_device_properties(i)
  print(f"GPU {i} - {props.name}:")
  print(f"   Compute capability: {props.major}.{props.minor}")
  print(f"   Total memory: {props.total_memory / 1024**3:.1f} GB")
  print(f"   Multiprocessors: {props.multi_processor_count}")