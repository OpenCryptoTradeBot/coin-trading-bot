import numpy as np
from typing import Optional, Tuple, Any
from datetime import datetime

from data.preprocess.ohlcv_data import preprocess_candles, preprocess_sequence_input
from data.upbit_api import UpbitAPI


class TradingEnv:
    """
    강화학습을 위한 트레이딩 환경.
    
    상태:
        - 최근 n개 시세 (window_size)
        - 보유 현금 (balance)
        - 보유 코인 수량 (holdings)
    
    행동:
        - action: int (0=HOLD, 1=BUY, 2=SELL)
        - weight: float (0.0 ~ 1.0)
    """

    def __init__(
        self,
        market: str,
        window_size: int = 20,
        initial_balance: float = 100_000,
        price_data: Optional[np.ndarray] = None
    ) -> None:
        self.market = market
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.api = UpbitAPI()
        self.data = price_data if price_data is not None else np.array([])

        self.reset()

    def reset(self) -> np.ndarray:
        """
        환경 상태 초기화 및 시작 상태 반환
        """
        self.balance = self.initial_balance
        self.holdings = 0.0
        self.current_step = self.window_size
        self.done = False
        return self._get_state()

    def _get_state(self) -> Tuple[np.ndarray, float, float]:
        """
        현재 상태 반환

        Returns:
            Tuple:
                - np.ndarray: 시계열 입력 (window_size, features)
                - float: 보유 현금
                - float: 보유 코인 수량
        """
        if self.data.size == 0:
            raw_data = self.api.get_candles(self.market, count=self.window_size)
            df = preprocess_candles(raw_data)
            window = preprocess_sequence_input(df, self.window_size)
        else:
            window = self.data[self.current_step - self.window_size:self.current_step]

        return window.astype(np.float32), self.balance, self.holdings

    def step(self, action: int, weight: float) -> Tuple[np.ndarray, float, bool, dict]:
        """
        주어진 행동을 수행하고 다음 상태, 보상, 종료 여부 반환

        Args:
            action (int): 행동 타입 (0=HOLD, 1=BUY, 2=SELL)
            weight (float): 투자 비율 (0.0 ~ 1.0)

        Returns:
            Tuple:
                - np.ndarray: 다음 상태
                - float: 보상
                - bool: 종료 여부
                - dict: 기타 정보
        """
        # 현재 시점 가격 (close price)
        price = self.data[self.current_step][3]

        # 행동 실행
        if action == 1 and self.balance > 0 and weight > 0:  # BUY
            buy_amount = self.balance * weight
            coin_bought = buy_amount / price
            self.balance -= buy_amount
            self.holdings += coin_bought
            print(f"✅ BUY: {coin_bought:.4f}개, {buy_amount:.2f}원")

        elif action == 2 and self.holdings > 0 and weight > 0:  # SELL
            sell_amount = self.holdings * weight
            cash_earned = sell_amount * price
            self.holdings -= sell_amount
            self.balance += cash_earned
            print(f"✅ SELL: {sell_amount:.4f}개, {cash_earned:.2f}원")

        # 스텝 이동
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        # 다음 상태
        next_state = self._get_state()
        next_price = self.data[self.current_step][3]
        portfolio = self.balance + self.holdings * next_price

        # 가격 변화 비율
        price_diff_ratio = (next_price - price) / price

        # 보상 계산: 예측 방향 맞으면 양수, 틀리면 음수
        if next_price > price:
            reward = price_diff_ratio if action == 1 else -price_diff_ratio
        elif next_price < price:
            reward = -price_diff_ratio if action == 2 else price_diff_ratio
        else:
            reward = 0.0

        info = {
            "balance": self.balance,
            "holdings": self.holdings,
            "total": portfolio
        }

        return next_state, reward, self.done, info

    def render(self) -> None:
        """
        현재 상태 출력
        """
        print(f"[Step {self.current_step}] Balance: {self.balance:.2f}, Holdings: {self.holdings:.4f}")
