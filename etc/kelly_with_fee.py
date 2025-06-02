import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def kelly_with_fee(p, q, b, a, fee = 0.001):
    """ 
    수수료를 고려한 켈리 공식
    - p: 이익확률 (이익 확률 몇?)
    - q: 손해확률 (손해 확률 몇?)
    - b: 순이익률 (이익시 얼만큼 이득?)
    - a: 순손해률 (손해시 얼만큼큼 손해?)
    - fee: 수수료(defualt: 0.001=업비트 수수료료)
    """
    numerator = p * (b - fee) - q * (a + fee)
    denominator = (b - fee) * (a + fee)
    f = numerator / denominator
    return max(0, min(f, 1.0))  # 0~1 사이로 제한

def get_optimal_bet_size(X_new, classifier, reg_b, reg_a, fee=0.001):
    """
    X_new: 신규 입력 피처 (DataFrame 또는 배열 형태)
    classifier: 분류 모델 (predict_proba() 필요)
    reg_b: 수익률 회귀 모델
    reg_a: 손실률 회귀 모델
    """
    # 1. 수익 확률 예측 (p)
    proba = classifier.predict_proba(X_new)[0]
    p = proba[1]  # class 1 = 수익 날 확률
    q = 1 - p

    # 2. 수익률/손실률 예측 (b, a)
    b = max(0, reg_b.predict(X_new)[0])
    a = max(0, reg_a.predict(X_new)[0])

    # 3. 켈리 공식 적용
    f = kelly_with_fee(p, q, b, a, fee)
    return f


def main():
    # 가상의 수익률 시계열 예시 (실제로는 API 데이터에서 계산됨)
    np.random.seed(50)
    returns = np.random.normal(loc=0.001, scale=0.01, size=1000)  # 평균 0.1%, 표준편차 1%
    df = pd.DataFrame({"return": returns})

    def kelly_f(p, q, b, a, fee=0.001):
        # 수익이 날 확률 p, 손실이 날 확률 q, 나머지는 r
        numerator = p * (b - fee) - q * (a + fee)
        denominator = (b - fee) * (a + fee)
        f = numerator / denominator
        return max(0, min(f, 1.0))  # 0~1 사이 제한

    # 기본 설정: 수익 확률 p, 손실 확률 q는 고정으로 가정
    p, q = 0.4, 0.4
    b, a = 0.04, 0.01
    fee = 0.001

    # 시간 순서대로 계산
    wealth = [1]  # 시작 자산
    for r in df["return"]:
        direction = 1 if r > 0 else -1  # 방향
        f = kelly_f(p, q, b, a, fee)

        # 수익률 적용 (수익/손실 비율 기준)
        profit_ratio = (1 + f * r) if direction > 0 else (1 - f * abs(r))
        wealth.append(wealth[-1] * profit_ratio)

    df["wealth"] = wealth[1:]

    plt.figure(figsize=(10, 5))
    plt.plot(df["wealth"], label="Kelly-adjusted Wealth")
    plt.title("💰 로그 누적 수익률 (with Kelly Betting)")
    plt.xlabel("Time Step")
    plt.ylabel("Wealth")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    main()