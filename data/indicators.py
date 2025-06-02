import pandas as pd

def compute_ma(df: pd.DataFrame, windows=[5, 20, 60]) -> pd.DataFrame:
    """
    이동평균(Moving Average) 계산
    windows: 이동 평균 기간 리스트 (ex: [5, 20])
    """
    for w in windows:
        df[f"ma{w}"] = df["close"].rolling(window=w).mean()
    return df

def compute_ema(df: pd.DataFrame, windows=[5, 10, 20]) -> pd.DataFrame:
    for w in windows:
        df[f"ema{w}"] = df["close"].ewm(span=w, adjust=False).mean()
    return df

def compute_bollinger_ema(df: pd.DataFrame, window=20, k=2) -> pd.DataFrame:
    ema = df["close"].ewm(span=window, adjust=False).mean()
    std = df["close"].rolling(window=window).std()
    df["bb_ema_mid"] = ema
    df["bb_ema_upper"] = ema + k * std
    df["bb_ema_lower"] = ema - k * std
    return df

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    RSI (Relative Strength Index) 계산
    """
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df[f"rsi{period}"] = 100 - (100 / (1 + rs))

    return df

def compute_volume_ma(df: pd.DataFrame, windows=[5, 20]) -> pd.DataFrame:
    for w in windows:
        df[f"vol_ma{w}"] = df["volume"].rolling(window=w).mean()
    return df

def compute_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence) 계산
    fast: 단기 EMA 기간
    slow: 장기 EMA 기간
    signal: 시그널 EMA 기간
    """
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df

def compute_roc(df: pd.DataFrame, period=12) -> pd.DataFrame:
    df["roc"] = df["close"].pct_change(periods=period) * 100
    return df

def apply_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    모든 지표 일괄 적용 함수
    """
    df = compute_ma(df)
    df = compute_ema(df)
    df = compute_bollinger_ema(df)
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_roc(df)
    df = compute_volume_ma(df)
    return df