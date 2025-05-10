import pandas as pd

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