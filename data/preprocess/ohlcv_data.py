import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from data.indicators import apply_all_indicators

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
    df = apply_all_indicators(df)
    df = df.dropna().reset_index(drop=True)
    return df

def preprocess_sequence_input(
    df: pd.DataFrame, 
    window: int = 30, 
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
    window: int = 30,
    label_col: str = "close",
    future_look_ahead_steps: int = 10,  # 몇 분(캔들 수) 뒤를 예측할 것인가(defualt: 10개개)
    threshold: float = 0.004,       # 예시: 10분 동안 +-0.4% (조절 필요)
) -> np.ndarray:
    """
    시계열 라벨 생성 함수
    - 입력 시퀀스 이후의 가격 변동을 기반으로 상승/하락 여부 라벨 생성
    - window 이후 시점의 `label_col` 값이 더 크면 1, 아니면 0

    Parameters:
        df (pd.DataFrame): OHLCV 데이터프레임
        window (int): 입력 시퀀스 길이
        label_col (str): 기준 가격 컬럼 (기본은 "close")
        future_look_ahead_steps: 몇 분(캔들 수) 뒤를 예측할 것인가(defualt: 10분)
        threshold: float = 0.0015 # 예시: 10분 동안 0.15% 상승 (조절 필요)

    Returns:
        np.ndarray: (N,) 형태의 라벨 배열 (0 or 1)
    """
    if label_col not in df.columns:
        raise ValueError(f"'{label_col}' 컬럼이 DataFrame에 존재하지 않습니다.")

    prices = df[label_col].values
    labels = []

    for i in range(window, len(prices) - future_look_ahead_steps):
        now = prices[i]
        future = prices[i + future_look_ahead_steps]
        change = (future - now) / now
        if change >= threshold:
            labels.append(1)  # 상승
        elif change <= -threshold:
            labels.append(2)  # 하락
        else:
            labels.append(0)  # 변동 없음

    return np.array(labels, dtype=np.int64)

def normalize_features(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    학습용 데이터 기준으로 정규화 스케일러를 fit하고,
    학습/검증 데이터에 동일 스케일로 transform을 적용한다.

    Returns:
        train_features (np.ndarray): 정규화된 학습 피처
        val_features (np.ndarray): 정규화된 검증 피처
        scaler (StandardScaler): 학습된 스케일러 객체 (추후 테스트에 재사용 가능)
    """
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    train_scaled = scaler.transform(train_df[feature_cols])
    val_scaled = scaler.transform(val_df[feature_cols])

    return train_scaled, val_scaled, scaler