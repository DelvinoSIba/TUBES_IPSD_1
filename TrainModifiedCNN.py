import os
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from Utils.getData import getImageLabel
from Models.simpleCNN import SimpleCNN
from Models.model_2_modified_cnn import ModifiedCNN


def get_model(name, img_size, num_classes):
    if name == "SimpleCNN":
        return SimpleCNN(input_dim=img_size, input_c=3, output=num_classes)
    if name == "Model2":
        return ModifiedCNN(input_c=3, output=num_classes)
    raise ValueError("MODEL_NAME harus 'SimpleCNN' atau 'Model2'")


def train_one_fold(fold, model, train_loader, valid_loader, n_train, n_valid, device, cfg, out_dir):
    optimizer = AdamW(model.parameters(), lr=cfg["LEARNING_RATE"], weight_decay=cfg["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["EPOCH"])
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)   # loss untuk klasifikasi multi-kelas
    ckpt_path = f"{out_dir}/fold{fold}_best.pt"

    history = []
    best_acc, wait = 0.0, 0
    start = time.time()

    for epoch in range(cfg["EPOCH"]):
        # ---------- Training ----------
        train_loss, train_correct = 0, 0
        model.train()
        for batch, (src, trg) in enumerate(tqdm(train_loader, desc=f"Fold {fold} | Epoch {epoch+1} train", leave=False)):
            src = torch.permute(src, (0, 3, 1, 2)).to(device)   # (B,H,W,C) -> (B,C,H,W)
            trg = trg.to(device)

            pred = model(src)
            loss = loss_function(pred, trg)
            train_loss += loss.item()
            train_correct += (pred.argmax(1) == trg.argmax(1)).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ---------- Validasi ----------
        valid_loss, valid_correct = 0, 0
        model.eval()
        with torch.no_grad():
            for batch, (src, trg) in enumerate(valid_loader):
                src = torch.permute(src, (0, 3, 1, 2)).to(device)
                trg = trg.to(device)

                pred = model(src)
                valid_loss += loss_function(pred, trg).item()
                valid_correct += (pred.argmax(1) == trg.argmax(1)).sum().item()
        scheduler.step()

        row = dict(epoch=epoch + 1,
                   train_loss=train_loss / len(train_loader), train_acc=train_correct / n_train,
                   valid_loss=valid_loss / len(valid_loader), valid_acc=valid_correct / n_valid)
        history.append(row)
        print(f"[Fold {fold}] epoch = {row['epoch']} | train loss = {row['train_loss']:.4f}, acc = {row['train_acc']:.4f} "
              f"| valid loss = {row['valid_loss']:.4f}, acc = {row['valid_acc']:.4f}")

        # simpan checkpoint terbaik (berdasarkan akurasi validasi)
        if row['valid_acc'] > best_acc:
            best_acc, wait = row['valid_acc'], 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': row['train_loss'],
                'valid_acc': best_acc,
            }, ckpt_path)
        else:
            wait += 1
            if wait >= cfg["PATIENCE"]:
                print(f"[Fold {fold}] Early stopping di epoch {epoch+1}")
                break

    train_time = time.time() - start
    print(f"[Fold {fold}] Selesai. Waktu latih: {train_time:.1f} detik | best valid acc: {best_acc:.4f}\n")

    # tambahkan info waktu latih ke checkpoint (dipakai Test.py)
    ckpt = torch.load(ckpt_path, weights_only=False)
    ckpt['train_time'] = train_time
    ckpt['epochs_trained'] = len(history)
    ckpt['img_size'] = cfg["IMG_SIZE"]
    torch.save(ckpt, ckpt_path)

    # ---------- Grafik loss & akurasi per fold ----------
    df = pd.DataFrame(history)
    df.to_csv(f"{out_dir}/history_fold{fold}.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(df.epoch, df.train_loss, color="#931a00", label='Training')
    ax[0].plot(df.epoch, df.valid_loss, color="#3399e6", label='Validation')
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].legend(); ax[0].set_title(f"Loss - Fold {fold}")
    ax[1].plot(df.epoch, df.train_acc, color="#931a00", label='Training')
    ax[1].plot(df.epoch, df.valid_acc, color="#3399e6", label='Validation')
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Accuracy"); ax[1].legend(); ax[1].set_title(f"Accuracy - Fold {fold}")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/training_fold{fold}.png", dpi=150)
    plt.close()
    return df


def plot_all_folds(out_dir, model_name, k_folds):
    """Gabungkan kurva semua fold dalam satu gambar (untuk laporan / README)."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for fold in range(1, k_folds + 1):
        path = f"{out_dir}/history_fold{fold}.csv"
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        ax[0].plot(df.epoch, df.valid_loss, label=f"Fold {fold}")
        ax[1].plot(df.epoch, df.valid_acc, label=f"Fold {fold}")
    ax[0].set_title(f"Validation Loss - {model_name}"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].set_title(f"Validation Accuracy - {model_name}"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/training_all_folds.png", dpi=150)
    plt.close()


def main():
    # ===== Konfigurasi =====
    MODEL_NAME = "Model2"      # "SimpleCNN" (Model 1) atau "Model2" (CNN modifikasi)
    K_FOLDS = 5
    FOLDS_TO_RUN = [1, 2, 3, 4, 5]   # bisa dicicil, misal [1, 2] dulu lalu [3, 4, 5]
    SKIP_DONE = True                 # lewati fold yang checkpoint-nya sudah ada (berguna kalau Colab terputus)
    cfg = dict(
        IMG_SIZE=128,
        BATCH_SIZE=32,
        EPOCH=30,
        LEARNING_RATE=0.001,
        WEIGHT_DECAY=1e-4,
        PATIENCE=7,                  # early stopping
    )
    NUM_WORKERS = 2
    SEED = 42
    OUTPUT_DIR = f"./Output/{MODEL_NAME}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Hanya Dataset/train yang dipakai untuk training (dengan augmentasi).
    # Dataset/valid & Dataset/test TIDAK dipakai untuk training.
    train_full = getImageLabel(folder="./Dataset/train", img_size=cfg["IMG_SIZE"], augment=True)
    valid_data = getImageLabel(folder="./Dataset/valid", img_size=cfg["IMG_SIZE"], augment=False)
    valid_loader = DataLoader(valid_data, batch_size=cfg["BATCH_SIZE"], shuffle=False, num_workers=NUM_WORKERS)
    num_classes = len(train_full.classes)
    print(f"Device: {device} | model: {MODEL_NAME} | kelas: {num_classes} | "
          f"train total: {len(train_full)} | valid: {len(valid_data)}")

    # ===== 5-Fold Stratified Cross Validation =====
    # Data train dibagi 5 bagian (proporsi kelas tetap sama). Di tiap fold, model dilatih
    # dengan 4/5 bagian, sehingga setiap fold menghasilkan model yang berbeda.
    # Validasi memakai Dataset/valid dan pengujian memakai Dataset/test (di Test.py).
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(np.zeros(len(train_full)), train_full.labels))

    fold_info = []
    for fold, (train_idx, holdout_idx) in enumerate(splits, start=1):
        fold_info.append(dict(fold=fold, n_train=len(train_idx), n_not_used=len(holdout_idx)))
        if fold not in FOLDS_TO_RUN:
            continue
        if SKIP_DONE and os.path.exists(f"{OUTPUT_DIR}/fold{fold}_best.pt"):
            print(f"[Fold {fold}] checkpoint sudah ada, dilewati.")
            continue

        torch.manual_seed(SEED + fold)
        train_loader = DataLoader(Subset(train_full, train_idx), batch_size=cfg["BATCH_SIZE"],
                                  shuffle=True, num_workers=NUM_WORKERS)
        model = get_model(MODEL_NAME, cfg["IMG_SIZE"], num_classes).to(device)
        if fold == FOLDS_TO_RUN[0]:
            print(model)
            print("Jumlah parameter:", sum(p.numel() for p in model.parameters()))
        print(f"===== Fold {fold}/{K_FOLDS} | data train: {len(train_idx)} =====")
        train_one_fold(fold, model, train_loader, valid_loader, len(train_idx), len(valid_data),
                       device, cfg, OUTPUT_DIR)

    pd.DataFrame(fold_info).to_csv(f"{OUTPUT_DIR}/fold_split.csv", index=False)
    plot_all_folds(OUTPUT_DIR, MODEL_NAME, K_FOLDS)
    print("Training selesai. Jalankan Test.py untuk evaluasi semua fold.")


if __name__ == "__main__":
    main()
