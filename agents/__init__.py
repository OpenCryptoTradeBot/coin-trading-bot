from typing import Tuple
import numpy as np
import torch

from agents.networks.classifier import PriceMovementClassifier

class TradingAgent:
    """
    트레이딩 에이전트
    """
    # TODO: 현재 단순 추론 모델로, 나중에 가치/정책 네트워크 추가해 강화학습 에이전트로 변경할 예정
    def __init__(self):
        """
        모델 로드 및 디바이스 설정
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PriceMovementClassifier(input_dim=5).to(self.device)
        self.model.load_state_dict(torch.load("models/price_classifier.pt", map_location=self.device))
        self.model.eval()

        self.profit_cost = []
        self.loss_cost = []
        self.model_correct = 0
        self.model_wrong = 0

    def act(self, state: Tuple[np.ndarray, float, float]) -> int:
        """
        현재 상태에서 행동을 예측함

        Parameters:
            state (Tuple): 
                - price_data (np.ndarray): 시계열 입력 데이터 (window_size, 5)
                - balance (float): 현금 보유량 (사용하지 않음)
                - holdings (float): 코인 보유량 (사용하지 않음)

        Returns:
            int: 선택된 행동 인덱스 (예: 0=SELL, 1=HOLD, 2=BUY)
        """
        price_data, _, _ = state

        # 입력 차원 검사 및 정규화
        if price_data.ndim == 2:
            # (seq_len, features) → (1, seq_len, features)
            x = torch.tensor(price_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        elif price_data.ndim == 3 and price_data.shape[0] == 1:
            # 이미 (1, seq_len, features)
            x = torch.tensor(price_data, dtype=torch.float32).to(self.device)
        else:
            raise ValueError(f"입력 시계열의 shape가 예상과 다릅니다: {price_data.shape}")

        # 추론 (gradient 미사용)
        with torch.no_grad():
            output = self.model(x)
            # 출력 노드 중 값이 가장 높은 노드 선택 (1이면 상승 예정, 0이면 하락 예정)
            pred = torch.argmax(output, dim=1).item()
        return 1 if pred == 1 else 2

    def kelly_with_fee(self, fee = 0.001):
        """ 
        수수료를 고려한 켈리 공식
        - p: 이익확률 (이익 확률 몇?)
        - q: 손해확률 (손해 확률 몇?)
        - b: 순이익률 (이익시 얼만큼 이득?)
        - a: 순손해률 (손해시 얼만큼큼 손해?)
        - fee: 수수료(defualt: 0.001=업비트 수수료)
        """
        try:
            p = self.model_correct / (self.model_correct + self.model_wrong)
            q = self.model_wrong / (self.model_correct + self.model_wrong)
            b = sum(self.profit_cost) / self.model_correct
            a = sum(self.loss_cost) / self.model_wrong
            numerator = p * (b - fee) - q * (a + fee)
            denominator = (b - fee) * (a + fee)
            f = numerator / denominator
            return max(0, min(f, 1.0))  # 0~1 사이로 제한
        except:
            return 1
    
    def append_profit(self, profit_cost: float) -> None:
        self.profit_cost.append(profit_cost)
        self.model_correct += 1

    def append_loss(self, loss_cost: float) -> None:
        self.loss_cost.append(loss_cost)
        self.model_wrong += 1
    