from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from time import sleep
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import matplotlib.pyplot as plt

from agents.networks.classifier import PriceMovementClassifier
from config.model import PMCModelConfig
from data.preprocess.ohlcv_data import preprocess_candles, preprocess_sequence_input, create_labels
from data.upbit_api import UpbitAPI

# Dataset
class OHLCVDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window: int):
        self.features = preprocess_sequence_input(df, window)[:-1]
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
    # 손실함수 설정
    loss_fn_class = getattr(torch.nn, PMCModelConfig.loss_fn)
    loss_fn = loss_fn_class()
    torch.nn.CrossEntropyLoss
    # 옵티마이저 설정
    optim_class = getattr(torch.optim, PMCModelConfig.optim)
    optimizer = optim_class(model.parameters(), lr=lr)

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
    torch.save(model.state_dict(), save_path)
    print(f"✅ 모델 저장 완료: {save_path}")
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
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["변동 없음", "상승", "하락"])
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(cmap="Blues", values_format="d", ax=ax)
        plt.title("혼동 행렬")
        plt.tight_layout()
        plt.savefig("etc/confusion_matrix.png")
        print("혼동 행렬 이미지 저장 완료: etc/confusion_matrix.png")
        plt.show()

if __name__ == "__main__":
    # 데이터셋 불러오기
    api = UpbitAPI()
    dt = datetime(2025, 1, 1, 0, 0, 0)
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
    # 입력 데이터와 라벨 데이터로 전처리(데이터셋)
    dataset = OHLCVDataset(full_df, window=20)
    # ✅ train/val 분할
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, shuffle=False)

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    # 데이터 로더로 저장
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

    # 입력 차원 확인 (ex: (20, 5) => input_dim=5)
    sample_x, _ = dataset[0]
    input_dim = sample_x.shape[-1]  # 마지막 인덱스 값: 한 시점 당 피처 개수

    model = PriceMovementClassifier(input_dim=input_dim)
    train(model, dataloader=train_loader, val_loader=val_loader, epochs=10_000, log_freq= 100,show_plot=True)