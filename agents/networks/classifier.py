import torch
import torch.nn as nn
import torch.nn.functional as F

class PriceMovementClassifier(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=64, ):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,          # 입력 feature 차원 (OHLCV 쓰고 있으니 5)
            hidden_dim,         # GRU의 hidden state 크기
            num_layers=2,       # GRU 층의 깊이
            batch_first=True,
            dropout=0.3, 
            bidirectional=True  # 양방향 GRU 사용
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 3)

    def forward(self, x):
        out, _ = self.gru(x)    # out: (B, T, H*2)
        out = out.mean(dim=1)   # 평균 풀링
        return self.fc(self.dropout(out))

class AttentionGRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,          # 입력 feature 차원 (OHLCV 쓰고 있으니 5)
            hidden_dim,         # GRU의 hidden state 크기
            num_layers=2,       # GRU 층의 깊이
            batch_first=True,
            dropout=0.3, 
            bidirectional=True  # 양방향 GRU 사용
        )
        self.dropout = nn.Dropout(0.3)
        self.attn = nn.Linear(hidden_dim*2, 1)  # Attention score 계산용
        self.classifier = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x):  # x: (B, T, F)
        gru_out, _ = self.gru(x)  # (B, T, H)
        # 1. Attention score 계산
        attn_scores = self.attn(gru_out).squeeze(-1)  # (B, T)
        # 2. Softmax로 정규화
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, T)
        # 3. 가중합
        context = torch.sum(gru_out * attn_weights.unsqueeze(-1), dim=1)  # (B, H)
        # 4. 분류
        logits = self.classifier(context)  # (B, C)
        return logits