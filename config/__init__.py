# 설정 파일
import os
from dotenv import load_dotenv
import torch
from dataclasses import dataclass

load_dotenv()

ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

@dataclass
class Config:
    #가중치(변동없음, 상승,하강)
    weight_tensor: torch.Tensor = torch.tensor([1, 1.5, 1.5], dtype=torch.float32) 
    threshold: float = 0.002 #변동없음 기준점

    hidden_dim: int = 128 # GRU의 hidden state 크기
    num_layer: int = 5 # GRU 층의 깊이
    dropout: float = 0.3 

    window: int = 120
    epochs: int = 500

    patience: int = 30