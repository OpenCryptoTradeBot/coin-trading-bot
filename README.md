# 🤖 OpenCryptoTrader

> [!NOTE]
> 강화학습 기반의 자동 암호화폐 트레이딩 봇.


## ✅ Requirements

- `python` >= 3.10
- Docker + NVIDIA GPU (optional but recommended)

## 📁 Project Structure

```
coin-trading-bot/
├── agents/         # 강화학습 에이전트 (DQN, PPO 등)
├── config/         # 설정 파일
├── data/           # 비트코인 시세 데이터 처리
├── envs/           # 트레이딩 환경
├── models/         # 정책/가치 네트워크 등
├── scripts/        # 학습, 평가, 실거래 실행 스크립트
├── train.py        # 학습 entrypoint
├── run.py          # 실거래/추론 entrypoint
├── .gitignore
├── .dockerignore
└── README.md
```

## ⚡️ Quick start

### 🌱 Setup environment variables

루트 디렉토리에 `.env` 파일 생성:
```
ACCESS_KEY=업비트 ACCESS KEY 입력
SECRET_KEY=업비트 SECRET KEY 입력
```

### 🐳 Build Image

```bash
docker build -t crypto-trader .
```

### ▶️ Run Container (with GPU)

```bash
docker run --gpus all -it --rm -v ${PWD}:/app crypto-trader
```

만약 직접 터미널에 들어가서 작업하고 싶으면:

```bash
docker run --gpus all -it --rm -v ${PWD}:/app crypto-trader
```

### 🏃 Run Training

```bash
python train.py
```

### 🤖 Run Inference / Live Trading
```bash
python run.py
```

## 📊 Features

- 실시간 시세 기반 트레이딩 봇
