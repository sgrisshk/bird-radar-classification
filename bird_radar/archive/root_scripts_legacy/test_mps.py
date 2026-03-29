#!/usr/bin/env python3
import os
import sys
import platform
import torch

def die(msg: str, code: int = 1):
    print(f"[FAIL] {msg}", file=sys.stderr)
    raise SystemExit(code)

print("python:", sys.version.replace("\n", " "))
print("executable:", sys.executable)
print("platform.machine:", platform.machine())
print("platform.platform:", platform.platform())
print("mac_ver:", platform.mac_ver()[0])
print("torch:", torch.__version__)
print("mps_built:", torch.backends.mps.is_built())
print("mps_available:", torch.backends.mps.is_available())
print("SYSTEM_VERSION_COMPAT:", os.environ.get("SYSTEM_VERSION_COMPAT"))

if not torch.backends.mps.is_built():
    die("PyTorch built without MPS support")
if not torch.backends.mps.is_available():
    die("MPS unavailable in this environment")

torch.manual_seed(42)
a_cpu = torch.randn(256, 256, dtype=torch.float32, device="cpu")
b_cpu = torch.randn(256, 256, dtype=torch.float32, device="cpu")
c_cpu = a_cpu @ b_cpu

a_mps = a_cpu.to("mps")
b_mps = b_cpu.to("mps")
c_mps = (a_mps @ b_mps).to("cpu")

max_abs = (c_cpu - c_mps).abs().max().item()
ok = torch.allclose(c_cpu, c_mps, rtol=1e-3, atol=1e-3)

print("mps_device_test:", a_mps.device)
print("allclose_cpu_vs_mps:", ok)
print("max_abs_diff:", max_abs)

if not ok:
    die(f"MPS matmul mismatch (max_abs_diff={max_abs:.6f})")

print("[OK] MPS works")
