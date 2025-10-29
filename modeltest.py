import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt
import pywt
import os

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 读取数据
data_path = "./data/processed_data/df_long_clean.pkl"
df = pd.read_pickle(data_path)

# 采样率
sampling_rate = 500

# 截断信号长度
max_len = 10000
df["ecg"] = df["ecg"].apply(lambda x: x[1000:1000+max_len])
df["ppg"] = df["ppg"].apply(lambda x: x[1000:1000+max_len])

# 信号去噪
def bandpass_filter(sig, fs, lowcut, highcut, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, sig)

def highpass_filter(sig, fs=500, cutoff=5, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype="high")
    return filtfilt(b, a, sig)

def wt_filter(sig, wavelet="db4", threshold=0.1):
    coeffs = pywt.wavedec(sig, wavelet)
    coeffs_thresholded = [pywt.threshold(c, threshold, mode="soft") for c in coeffs]
    return pywt.waverec(coeffs_thresholded, wavelet)

df["denoised_ppg"] = df["ppg"].apply(lambda x: bandpass_filter(x, fs=sampling_rate, lowcut=0.5, highcut=4))
df["denoised_ecg"] = df["ecg"].apply(lambda x: wt_filter(highpass_filter(x, fs=sampling_rate), wavelet="db4", threshold=0.1))

# 下采样到 125Hz
def downsample(sig):
    sig = np.asarray(sig)
    return sig[::4] if sig is not None and sig.size > 0 else sig

df["denoised_ppg"] = df["denoised_ppg"].apply(downsample)
df["denoised_ecg"] = df["denoised_ecg"].apply(downsample)

# 标准化
scaler_ecg = StandardScaler()
scaler_ppg = StandardScaler()
df["normalized_ecg"] = df["denoised_ecg"].apply(lambda x: scaler_ecg.fit_transform(x.reshape(-1, 1)).flatten())
df["normalized_ppg"] = df["denoised_ppg"].apply(lambda x: scaler_ppg.fit_transform(x.reshape(-1, 1)).flatten())

# 构建特征和标签
df["ecg_ppg_diff"] = df.apply(lambda row: row["normalized_ecg"] - row["normalized_ppg"], axis=1)
X = np.stack(df["ecg_ppg_diff"].values)
y_sbp = df["hbp"].values
y_dbp = df["lbp"].values

# 数据集划分
X_temp, X_test, y_sbp_temp, y_sbp_test, y_dbp_temp, y_dbp_test = train_test_split(
    X, y_sbp, y_dbp, test_size=0.2, random_state=42
)
X_train, X_val, y_sbp_train, y_sbp_val, y_dbp_train, y_dbp_val = train_test_split(
    X_temp, y_sbp_temp, y_dbp_temp, test_size=0.2, random_state=42
)

# 转为 tensor
def to_tensor(x, y1, y2):
    return (
        torch.tensor(x, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y1, dtype=torch.float32).unsqueeze(1),
        torch.tensor(y2, dtype=torch.float32).unsqueeze(1),
    )

X_train_tensor, y_sbp_train_tensor, y_dbp_train_tensor = to_tensor(X_train, y_sbp_train, y_dbp_train)
X_val_tensor, y_sbp_val_tensor, y_dbp_val_tensor = to_tensor(X_val, y_sbp_val, y_dbp_val)
X_test_tensor, y_sbp_test_tensor, y_dbp_test_tensor = to_tensor(X_test, y_sbp_test, y_dbp_test)

train_dataset = TensorDataset(X_train_tensor, y_sbp_train_tensor, y_dbp_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_sbp_val_tensor, y_dbp_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_sbp_test_tensor, y_dbp_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

from models.cnnbigru import CNNBiGRU

# 训练参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNBiGRU(in_channels=1).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
NUM_EPOCHS = 200
MODEL_SAVE_PATH = "models/CNNBiGRU_best.pth"
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
PATIENCE = 25
best_val_loss = float("inf")
epochs_no_improve = 0

train_loss_curve = []
val_loss_curve = []

for epoch in range(1, NUM_EPOCHS + 1):
    # 训练
    model.train()
    train_losses = []
    for xb, yb_sbp, yb_dbp in train_loader:
        xb = xb.to(DEVICE)
        yb = torch.cat([yb_sbp, yb_dbp], dim=1).to(DEVICE)
        optimizer.zero_grad()
        pred_sbp, pred_dbp = model(xb)
        preds = torch.cat([pred_sbp, pred_dbp], dim=1)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    avg_train_loss = np.mean(train_losses)
    train_loss_curve.append(avg_train_loss)

    # 验证
    model.eval()
    val_losses = []
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb_sbp, yb_dbp in val_loader:
            xb = xb.to(DEVICE)
            yb = torch.cat([yb_sbp, yb_dbp], dim=1).to(DEVICE)
            pred_sbp, pred_dbp = model(xb)
            preds = torch.cat([pred_sbp, pred_dbp], dim=1)
            loss = criterion(preds, yb)
            val_losses.append(loss.item())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())
    avg_val_loss = np.mean(val_losses)
    val_loss_curve.append(avg_val_loss)

    preds_np = np.vstack(all_preds) if all_preds else np.empty((0, 2))
    targets_np = np.vstack(all_targets) if all_targets else np.empty((0, 2))

    # 计算 MAE, RMSE, R2
    if preds_np.shape[0] > 0:
        sbp_mae = mean_absolute_error(targets_np[:, 0], preds_np[:, 0])
        dbp_mae = mean_absolute_error(targets_np[:, 1], preds_np[:, 1])
        sbp_rmse = np.sqrt(np.mean((preds_np[:, 0] - targets_np[:, 0]) ** 2))
        dbp_rmse = np.sqrt(np.mean((preds_np[:, 1] - targets_np[:, 1]) ** 2))
        sbp_r2 = r2_score(targets_np[:, 0], preds_np[:, 0])
        dbp_r2 = r2_score(targets_np[:, 1], preds_np[:, 1])
    else:
        sbp_mae = dbp_mae = sbp_rmse = dbp_rmse = sbp_r2 = dbp_r2 = float("nan")

    print(
        f"Epoch {epoch:03d} | TrainLoss {avg_train_loss:.6f} | ValLoss {avg_val_loss:.6f} | "
        f"SBP MAE {sbp_mae:.3f} RMSE {sbp_rmse:.3f} R2 {sbp_r2:.3f} | "
        f"DBP MAE {dbp_mae:.3f} RMSE {dbp_rmse:.3f} R2 {dbp_r2:.3f}"
    )

    # early stopping & save best
    if avg_val_loss < best_val_loss - 1e-6:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": best_val_loss,
            },
            MODEL_SAVE_PATH,
        )
        print("  -> Saved best model")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch}, no improvement for {PATIENCE} epochs."
            )
            break

# 损失曲线
plt.figure(figsize=(8, 5))
plt.plot(train_loss_curve, label="Train Loss")
plt.plot(val_loss_curve, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("训练与验证损失曲线")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150)
plt.close()

# 测试集评估
model.eval()
test_preds, test_targets = [], []
with torch.no_grad():
    for xb, yb_sbp, yb_dbp in test_loader:
        xb = xb.to(DEVICE)
        yb = torch.cat([yb_sbp, yb_dbp], dim=1).to(DEVICE)
        pred_sbp, pred_dbp = model(xb)
        preds = torch.cat([pred_sbp, pred_dbp], dim=1)
        test_preds.append(preds.cpu().numpy())
        test_targets.append(yb.cpu().numpy())
test_preds_np = np.vstack(test_preds)
test_targets_np = np.vstack(test_targets)

sbp_mae = mean_absolute_error(test_targets_np[:, 0], test_preds_np[:, 0])
dbp_mae = mean_absolute_error(test_targets_np[:, 1], test_preds_np[:, 1])
sbp_r2 = r2_score(test_targets_np[:, 0], test_preds_np[:, 0])
dbp_r2 = r2_score(test_targets_np[:, 1], test_preds_np[:, 1])

print(f"测试集 SBP MAE: {sbp_mae:.2f}, R2: {sbp_r2:.3f}")
print(f"测试集 DBP MAE: {dbp_mae:.2f}, R2: {dbp_r2:.3f}")