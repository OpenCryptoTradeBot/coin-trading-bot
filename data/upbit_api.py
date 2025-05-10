import requests
from typing import Optional

class UpbitAPI:
    BASE_URL = "https://api.upbit.com/v1"

    def __init__(self):
        self.session = requests.Session()

    def get_ticker(self, market: str) -> Optional[dict]:
        """지정된 마켓의 현재 시세 정보를 조회한다."""
        url = f"{self.BASE_URL}/ticker"
        params = {"markets": market}
        resp = self.session.get(url, params=params)
        if resp.status_code == 200:
            return resp.json()[0]
        return None

    def get_candles(self, market: str, unit: str = "minute", interval: int = 1, count: int = 200) -> Optional[list]:
        """
        지정된 마켓의 캔들 데이터를 조회한다.
        - unit: 'minute', 'day', 'week', 'month'
        - interval: 분봉일 경우 1, 3, 5, 10, 15, 30, 60, 240 가능
        - count: 데이터 개수 (최대 200)
        """
        if unit == "minute":
            url = f"{self.BASE_URL}/candles/minutes/{interval}"
        else:
            url = f"{self.BASE_URL}/candles/{unit}s"
        params = {"market": market, "count": count}
        resp = self.session.get(url, params=params)
        if resp.status_code == 200:
            return resp.json()
        return None

    def get_market_list(self) -> Optional[list]:
        """업비트의 전체 거래 가능한 마켓 목록을 조회한다."""
        url = f"{self.BASE_URL}/market/all"
        params = {"isDetails": "false"}
        resp = self.session.get(url, params=params)
        if resp.status_code == 200:
            return resp.json()
        return None