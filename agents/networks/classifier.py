import torch.nn as nn

class PriceMovementClassifier(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
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
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        out, _ = self.gru(x)    # out: (B, T, H*2)
        out = out.mean(dim=1)   # 평균 풀링
        return self.fc(self.dropout(out))
