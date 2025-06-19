# 설정 파일
import os
from dotenv import load_dotenv
import torch
from dataclasses import dataclass
from datetime import datetime

load_dotenv()

ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

@dataclass
class Config:
    #데이터 수집 시작할 날짜
    dt: datetime = datetime(2019, 10, 1, 0, 0, 0)

    #가중치(변동없음, 상승,하강)
    weight_tensor: torch.Tensor = torch.tensor([1, 1.3, 1.3], dtype=torch.float32) 

    future_look_ahead_steps: int = 4#몇개의 캔들뒤를 예측할 것인가
    threshold: float = 0.0026 #변동없음 기준점

    hidden_dim: int =64 # GRU의 hidden state 크기
    num_layer: int = 4 # GRU 층의 깊이
    dropout: float = 0.25

    window: int = 24
    epochs: int = 500

    patience: int = 20