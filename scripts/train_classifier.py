from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime, timedelta
from time import sleep
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from agents.networks.classifier import PriceMovementClassifier, AttentionGRUClassifier
from config.model import PMCModelConfig
from data.preprocess.ohlcv_data import preprocess_candles, preprocess_sequence_input, create_labels, normalize_features
from data.upbit_api import UpbitAPI

# Dataset
class OHLCVDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window: int):
        feature_cols = [
            "open", "high", "low", "close", "volume",
            "ema5", "ema10", "ema20", "rsi14",
            "macd", "macd_signal", "macd_hist",
            "bb_ema_upper", "bb_ema_lower",
            "vol_ma5", "vol_ma20"
        ]
        self.features = preprocess_sequence_input(df, window, feature_cols=feature_cols)[:-1]
        self.labels = create_labels(df, window)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


def train(
    model,
    dataloader, 
    val_loader,
    epochs: int = 10, 
    lr: float = 1e-4, 
    save_path: str = "models/price_classifier.pt",
    log_freq: int = 100,
    show_plot: bool = False,
    plot_path: str = "etc/loss_acc_plot.png"
) -> None:
    """
    시계열 가격 추세 분류 모델 학습 함수.

    Parameters:
        model (PriceDirectionClassifier): 상승/하락 추세 분류 모델
        dataloader (torch.utils.data.DataLoader): 학습 데이터 로더 
        epochs (int): 학습 반복 횟수 (default: 10)
        lr (float): 학습률 (default: 1e-4)
        save_path (str): 학습된 모델 저장 경로 (default: models/price_classifier.pt)
        log_freq (int): 학습 중 로그 출력 빈도 (default: 100 step마다)
        show_plot (bool): True이면 학습 정확도/손실 그래프를 출력 (default: False)

    Returns:
        None
    """
    # GPU 사용 (불가능하면 CPU 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 모델을 GPU (혹은 CPU)로 옮김
    model = model.to(device)
    # 에폭 및 학습률 설정
    #epochs = PMCModelConfig.epochs
    lr = PMCModelConfig.lr
    # 손실함수 설정 -> 가중치 적용함
    #loss_fn_class = getattr(torch.nn, PMCModelConfig.loss_fn)
    #loss_fn = loss_fn_class()
    #torch.nn.CrossEntropyLoss
    all_labels = torch.cat([y for _, y in dataloader], dim=0).cpu().numpy()
    weight_tensor = torch.tensor([0.8, 1.5, 1.5], dtype=torch.float32).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    # 옵티마이저 설정
    optim_class = getattr(torch.optim, PMCModelConfig.optim)
    optimizer = optim_class(model.parameters(), lr=lr)
    #최적모델을 저장하기위한 변수 설정정
    best_val_acc = 0.0
    patience = 100
    no_improve_epochs = 0
    # ReduceLROnPlateau 설정
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # 지표 기록용 리스트 변수
    if show_plot:
        loss_list = {"train": [], "val": []}    # 추후 검증용 데이터도 넣을 수도 있기 때문에 따로 추가해둠
        accuracy_list = {"train": [], "val": []}

    
    # 데이터셋을 epochs만큼 반복
    for epoch in range(epochs):
        # 모델 학습 시작
        model.train()

        total_loss = 0
        correct = 0
        total = 0
        for _, data in enumerate(dataloader):
            # data에서 input 값(inputs)과 정답(labels) 받아오기
            inputs, labels = data
            # input과 label 데이터를 동일한 디바이스로 옮김
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # gradient 값을 0으로 초기화
            optimizer.zero_grad()

            # 순전파
            preds = model(inputs)
            loss = loss_fn(preds, labels)
            # 역전파
            loss.backward()
            # 옵티마이저를 통해 최적화
            optimizer.step()

            # 총 손실값 저장
            total_loss += loss.item()
            # 맞춘 데이터 개수 저장
            correct += (preds.argmax(dim=1) == labels).sum().item()
            # 전체 데이터 개수 저장
            total += labels.size(0)

        # --- 검증 ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += loss_fn(preds, yb).item()
                val_correct += (preds.argmax(1) == yb).sum().item()
                val_total += yb.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total

        scheduler.step(val_loss)

        # 출력 및 기록
        if (epoch + 1) % log_freq == 0:
            print(f"[Epoch {epoch + 1:3d}] Train Loss: {total_loss:2.4f}, Accuracy: {correct / total:.8f} | Val Loss: {val_loss:.4f}, Acc: {avg_val_acc:.4f}")

        if show_plot:
            avg_loss = total_loss / len(dataloader)
            avg_acc = correct / total
            loss_list["train"].append(avg_loss)
            accuracy_list["train"].append(avg_acc)
            loss_list["val"].append(avg_val_loss)
            accuracy_list["val"].append(avg_val_acc)
    # 모델 저장
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), save_path)
        else:
            no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"✅ 모델 저장 완료: {save_path}")
            break

    # 그래프 출력
    if show_plot:
        plt.figure(figsize=(8,3))
        # 훈련 손실 그래프
        plt.subplot(121)
        plt.plot(loss_list["train"])
        plt.plot(loss_list["val"])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['train', 'val'], loc='upper right')

        # 훈련 정확도 그래프
        plt.subplot(122)
        plt.plot(accuracy_list["train"])
        plt.plot(accuracy_list["val"])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['train', 'val'], loc='lower right')
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"학습 시각화 이미지 저장 완료: {plot_path}")
        plt.show()
        # 혼동 행렬 시각화
        all_preds = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds = model(xb)
                pred_labels = preds.argmax(1).cpu().numpy()
                all_preds.extend(pred_labels)
                all_labels.extend(yb.numpy())

        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Change", "Up", "Down"])
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(cmap="Blues", values_format="d", ax=ax)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("etc/confusion_matrix.png")
        print("혼동 행렬 이미지 저장 완료: etc/confusion_matrix.png")
        plt.show()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        # 🔽 F1-score, precision, recall 출력
        print(classification_report(
            all_labels, all_preds,
            target_names=["No Change", "Up", "Down"]
        ))

if __name__ == "__main__":
    # 데이터셋 불러오기
    api = UpbitAPI()
    dt = datetime(2025, 2, 28, 0, 0, 0)
    dfs = []  # 여러 개의 DataFrame 저장 리스트

    for i in range(100):
        raw_data = api.get_candles("KRW-XRP", unit="minute", interval=1, count=200, to=dt)
        #print(raw_data)
        df = preprocess_candles(raw_data)
        print(f"Data count: {i * 200}, Date: {dt}")
        dfs.append(df)

        # 가장 과거 시점 기준으로 to 값을 업데이트 (Upbit는 과거로 조회됨)
        earliest_time = df.iloc[-1]["datetime"]
        # 다음 요청 시점
        dt = earliest_time + timedelta(minutes=1) - timedelta(hours=5, minutes=40)  
        sleep(0.1)

    # 전체 데이터프레임 연결
    full_df = pd.concat(dfs, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    full_df = full_df.dropna().reset_index(drop=True) #이거 해준거 아닌가?

    # ✅ feature 컬럼 정의
    feature_cols = [
        "open", "high", "low", "close", "volume",
        "ema5", "ema10", "ema20", "rsi14",
        "macd", "macd_signal", "macd_hist",
        "bb_ema_upper", "bb_ema_lower",
        "vol_ma5", "vol_ma20"
    ]

    # ✅ train/val 분할 (DataFrame 기준)
    cut = int(len(full_df) * 0.8)
    train_df, val_df = full_df.iloc[:cut], full_df.iloc[cut:]

    # ✅ 정규화
    train_scaled, val_scaled, scaler = normalize_features(train_df, val_df, feature_cols)

    # ✅ 시퀀스 생성
    window = 120
    train_seq = preprocess_sequence_input(pd.DataFrame(train_scaled, columns=feature_cols), window, feature_cols)
    val_seq = preprocess_sequence_input(pd.DataFrame(val_scaled, columns=feature_cols), window, feature_cols)

    # ✅ 라벨 생성 (원본 df 기준)
    train_labels = create_labels(train_df, window)[:-1]
    val_labels = create_labels(val_df, window)[:-1]
    # ✅ 길이 맞추기 (시퀀스를 라벨 수에 맞춰 자름)
    train_seq = train_seq[:len(train_labels)]
    val_seq = val_seq[:len(val_labels)]
    # ✅ TensorDataset 구성
    train_set = torch.utils.data.TensorDataset(
        torch.tensor(train_seq, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    )
    val_set = torch.utils.data.TensorDataset(
        torch.tensor(val_seq, dtype=torch.float32),
        torch.tensor(val_labels, dtype=torch.long)
    )

    # ✅ Dataloader
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    # ✅ 모델 초기화
    input_dim = train_seq.shape[-1]
    model = AttentionGRUClassifier(input_dim=input_dim)

    # ✅ 학습 실행
    train(model, dataloader=train_loader, val_loader=val_loader, epochs=500, log_freq=10, show_plot=True)

    