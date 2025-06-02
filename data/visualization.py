import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")  # 깔끔한 시각화 스타일

def plot_price_ma(df: pd.DataFrame):
    plt.figure(figsize=(12, 6))
    plt.plot(df["datetime"], df["close"], label="Close", color="black")
    if "ma5" in df.columns:
        plt.plot(df["datetime"], df["ma5"], label="MA5", linestyle="--")
    if "ma20" in df.columns:
        plt.plot(df["datetime"], df["ma20"], label="MA20", linestyle="--")
    plt.title(" 종가 + 이동평균선")
    plt.xlabel("Time")
    plt.ylabel("Price (KRW)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_rsi(df: pd.DataFrame):
    if "rsi14" not in df.columns:
        print("RSI 데이터 없음")
        return
    plt.figure(figsize=(12, 4))
    plt.plot(df["datetime"], df["rsi14"], label="RSI(14)", color="purple")
    plt.axhline(70, color="red", linestyle="--", label="Overbought")
    plt.axhline(30, color="green", linestyle="--", label="Oversold")
    plt.title(" RSI (Relative Strength Index)")
    plt.xlabel("Time")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_macd(df: pd.DataFrame):
    if "macd" not in df.columns:
        print("MACD 데이터 없음")
        return
    plt.figure(figsize=(12, 5))
    plt.plot(df["datetime"], df["macd"], label="MACD", color="blue")
    plt.plot(df["datetime"], df["macd_signal"], label="Signal", color="orange")
    plt.bar(df["datetime"], df["macd_hist"], label="Histogram", color="gray", alpha=0.5)
    plt.title("📉 MACD (Moving Average Convergence Divergence)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()