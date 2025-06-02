from data.upbit_api import UpbitAPI
from data.preprocess.ohlcv_data import preprocess_candles
from data.indicators import compute_rsi
from data.visualization import plot_rsi
import matplotlib.pyplot as plt

def main():
    api = UpbitAPI()
    raw_data = api.get_candles("KRW-BTC", unit="minute", interval=5, count=200)

    df = preprocess_candles(raw_data)
    df = compute_rsi(df)

    plot_rsi(df)


if __name__ == "__main__":
    main()
