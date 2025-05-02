# 학습 entrypoint
import torch

# GPU 사용 가능 여부 확인
print("✅ GPU is available." if torch.cuda.is_available() else "❌ GPU is not available")
