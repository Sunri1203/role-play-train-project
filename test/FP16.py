import sys, torch, torch.nn.functional as F
print("py:", sys.executable)
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "gpu:", torch.cuda.get_device_name(0))
# 最小 GPU 測試
x = torch.randn(1, device="cuda")
print("torch ok:", x.device)
# 測試 embedding（先 FP16，再 BF16）
w = torch.randn(1024, 128, device="cuda", dtype=torch.float16)
idx = torch.randint(0, 1024, (4, 16), device="cuda")
print("embedding fp16:", F.embedding(w, idx).dtype)
try:
    wb = w.to(torch.bfloat16)
    print("embedding bf16:", F.embedding(wb, idx).dtype)
except Exception as e:
    print("embedding bf16 ERROR:", type(e).__name__, e)