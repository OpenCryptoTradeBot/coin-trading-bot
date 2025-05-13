import pandas as pd
import numpy as np


def preprocess_candles(candles: list) -> pd.DataFrame:
    """
    업비트 API에서 받은 캔들 데이터를 pandas DataFrame으로 전처리
    - 날짜 정렬
    - 컬럼 이름 정리 (open, high, low, close, volume)
    """
    if not candles or not isinstance(candles, list):
        raise ValueError("입력 데이터가 비어 있거나 잘못된 형식입니다.")

    # JSON 리스트 → DataFrame
    df = pd.DataFrame(candles)

    # 날짜 파싱 및 정렬
    df["datetime"] = pd.to_datetime(df["candle_date_time_kst"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # 주요 컬럼만 선택
    df = df[[
        "datetime",              # KST 기준 시간
        "opening_price",         # 시가
        "high_price",            # 고가
        "low_price",             # 저가
        "trade_price",           # 종가
        "candle_acc_trade_volume"  # 거래량
    ]]

    # 컬럼 이름 통일 (표준 OHLCV)
    df.columns = ["datetime", "open", "high", "low", "close", "volume"]

    return df

def preprocess_sequence_input(
    df: pd.DataFrame, 
    window: int = 20, 
    feature_cols: list[str] = ["open", "high", "low", "close", "volume"]
) -> np.ndarray:
    """
    OHLCV DataFrame에서 시계열 입력 시퀀스 생성 함수

    Parameters:
        df (DataFrame): preprocess_candles로 정제된 DataFrame
        window (int): 시퀀스 길이 (ex. 20)
        feature_cols (List[str]): 사용할 피처 컬럼들 (기본은 OHLCV)
    
    Returns:
        np.ndarray: [샘플 개수(batch size), 시퀀스 길이(timesteps), 특성 수(features per timestep)] 형태의 시계열
    """
    if not all(col in df.columns for col in feature_cols):
        missing = list(set(feature_cols) - set(df.columns))
        raise ValueError(f"다음 컬럼이 누락되었습니다: {missing}")

    features = df[feature_cols].values.astype(np.float32)

    sequences = [
        features[i : i + window]
        for i in range(len(features) - window)
    ]

    return np.stack(sequences) if sequences else np.empty((0, window, len(feature_cols)), dtype=np.float32)

def create_labels(
    df: pd.DataFrame, 
    window: int = 20,
    label_col: str = "close"
) -> np.ndarray:
    """
    시계열 라벨 생성 함수
    - 입력 시퀀스 이후의 가격 변동을 기반으로 상승/하락 여부 라벨 생성
    - window 이후 시점의 `label_col` 값이 더 크면 1, 아니면 0

    Parameters:
        df (pd.DataFrame): OHLCV 데이터프레임
        window (int): 입력 시퀀스 길이
        label_col (str): 기준 가격 컬럼 (기본은 "close")

    Returns:
        np.ndarray: (N,) 형태의 라벨 배열 (0 or 1)
    """
    if label_col not in df.columns:
        raise ValueError(f"'{label_col}' 컬럼이 DataFrame에 존재하지 않습니다.")

    prices = df[label_col].values
    labels = []

    for i in range(window, len(prices) - 1):
        now = prices[i]
        future = prices[i + 1]
        label = 1 if future > now else 0
        labels.append(label)

    return np.array(labels, dtype=np.int64)
