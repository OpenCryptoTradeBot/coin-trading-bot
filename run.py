# 실거래/추론 entrypoint
from datetime import datetime
import torch

from agents import TradingAgent
from data.preprocess.ohlcv_data import preprocess_candles, preprocess_sequence_input
from data.upbit_api import UpbitAPI
from envs.trading_env import TradingEnv


# GPU 사용 가능 여부 확인
print("✅ GPU is available." if torch.cuda.is_available() else "❌ GPU is not available")

# 마켓, 시퀀스 크기 설정
market = "KRW-XRP"
window_size = 20

# 데이터 받아오기
api = UpbitAPI()
raw_data = api.get_candles(market, to=datetime(2025, 5, 12, 0, 0, 0))
df = preprocess_candles(raw_data)
price_data = preprocess_sequence_input(df, window_size)

# 환경 및 에이전트 설정
env = TradingEnv(market, window_size=window_size, price_data=price_data)
agent = TradingAgent()

# 환경이 끝날때까지 반복
state = env.reset()
while not env.done:
    action = agent.act(state)
    next_state, reward, done, info = env.step(action, agent.kelly_with_fee())
    # 보상 확인
    agent.append_profit(reward) if reward > 0 else agent.append_loss(reward)
    # env.render()
    state = next_state
    
print(f"💵 현금: {info['balance']}, 🪙 코인 보유량: {info['holdings']} | 💰 보유 자산: {info['total']}")
