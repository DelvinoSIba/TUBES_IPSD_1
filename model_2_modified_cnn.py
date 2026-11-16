"""
Model 2 - Modified CNN (klasifikasi 50 kelas ngengat / MOTHS dataset)
=====================================================================

Modifikasi dibanding Simple CNN (baseline / Model 1):
  1. Jaringan lebih dalam      : 4 blok konvolusi, tiap blok berisi 2 layer Conv 3x3
                                 (baseline biasanya 2-3 Conv tunggal).
  2. Batch Normalization       : setelah setiap Conv -> training lebih stabil & cepat konvergen.
  3. Dropout                   : Dropout2d di tiap blok + Dropout 0.5 sebelum classifier
                                 -> mengurangi overfitting.
  4. Global Average Pooling    : menggantikan Flatten besar -> parameter jauh lebih sedikit.
  5. Training tricks           : AdamW + weight decay, label smoothing, CosineAnnealing LR,
                                 early stopping, simpan bobot terbaik (berdasarkan val acc).

Cara pakai (dari folder project):
    python model2_modified_cnn.py                       # latih Model 2
    python model2_modified_cnn.py --epochs 40 --img-size 224
    python model2_modified_cnn.py --train-baseline      # latih juga Simple CNN untuk pembanding

Output disimpan di folder  outputs/model2/ :
    best_model2.pth, history.csv, results.json, classification_report.csv,
    training_curves.png, confusion_matrix.png, top_confusions.csv, comparison.csv (jika ada baseline)

Library: torch, torchvision, numpy, pandas, matplotlib, seaborn, scikit-learn, Pillow, tqdm
"""

import argparse
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")  # simpan gambar tanpa membuka jendela
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm


# ----------------------------------------------------------------------------
# 1. Argumen / konfigurasi
# ----------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="Model 2 - Modified CNN")
    p.add_argument("--data-dir", default="Dataset", help="folder berisi train/valid/test")
    p.add_argument("--out-dir", default=os.path.join("outputs", "model2"))
    p.add_argument("--img-size", type=int, default=128, help="128 lebih cepat, 224 lebih akurat")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=7, help="early stopping")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-baseline", action="store_true",
                   help="latih juga Simple CNN (baseline) dengan setting yang sama untuk perbandingan")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# 2. Data loading & preprocessing
#    (samakan dengan preprocessing Model 1 milik teman kalau berbeda)
# ----------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_dataloaders(data_dir, img_size, batch_size, num_workers):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "valid"), transform=eval_tf)
    test_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=eval_tf)

    pin = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin)
    return train_dl, val_dl, test_dl, train_ds.classes


# ----------------------------------------------------------------------------
# 3. Arsitektur
# ----------------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """Baseline (perkiraan Model 1). Ganti dengan class milik teman agar perbandingan adil."""

    def __init__(self, num_classes=50, img_size=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        feat = img_size // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * feat * feat, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def conv_block(in_ch, out_ch, drop):
    """[Conv-BN-ReLU] x2 -> MaxPool -> Dropout2d"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout2d(drop),
    )


class ModifiedCNN(nn.Module):
    """Model 2: CNN lebih dalam + BatchNorm + Dropout + Global Average Pooling."""

    def __init__(self, num_classes=50):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32, 0.10),     # 128 -> 64
            conv_block(32, 64, 0.15),    # 64  -> 32
            conv_block(64, 128, 0.20),   # 32  -> 16
            conv_block(128, 256, 0.25),  # 16  -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------
# 4. Training & evaluasi
# ----------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, desc=desc, leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in tqdm(loader, desc="Testing", leave=False):
        preds.append(model(x.to(device)).argmax(1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(preds)


def train_model(model, name, train_dl, val_dl, device, args, out_dir):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt = os.path.join(out_dir, f"best_{name}.pth")
    history, best_acc, wait = [], 0.0, 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, device, optimizer, f"[{name}] ep{epoch} train")
        va_loss, va_acc = run_epoch(model, val_dl, criterion, device, None, f"[{name}] ep{epoch} val")
        scheduler.step()
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, lr=optimizer.param_groups[0]["lr"]))
        print(f"[{name}] Epoch {epoch:02d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_acc:
            best_acc, wait = va_acc, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= args.patience:
                print(f"[{name}] Early stopping di epoch {epoch}")
                break

    train_time = time.time() - start
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return pd.DataFrame(history), train_time, best_acc


def evaluate_and_save(model, name, history, train_time, best_val, test_dl, classes, device, out_dir):
    y_true, y_pred = predict(model, test_dl, device)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    history.to_csv(os.path.join(out_dir, f"history_{name}.csv"), index=False)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(os.path.join(out_dir, f"classification_report_{name}.csv"))

    # --- Kurva training ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.epoch, history.train_loss, label="train")
    ax[0].plot(history.epoch, history.val_loss, label="val")
    ax[0].set_title(f"Loss - {name}"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history.epoch, history.train_acc, label="train")
    ax[1].plot(history.epoch, history.val_acc, label="val")
    ax[1].set_title(f"Accuracy - {name}"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"training_curves_{name}.png"), dpi=150)
    plt.close()

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    plt.figure(figsize=(18, 15))
    sns.heatmap(cm, cmap="Blues", xticklabels=classes, yticklabels=classes, cbar=True)
    plt.title(f"Confusion Matrix - {name} (test acc {acc:.3f})")
    plt.xlabel("Prediksi"); plt.ylabel("Label asli")
    plt.xticks(fontsize=6, rotation=90); plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{name}.png"), dpi=150)
    plt.close()

    # --- Kelas yang paling sering tertukar ---
    off = cm.copy(); np.fill_diagonal(off, 0)
    pairs = [(classes[i], classes[j], int(off[i, j]))
             for i, j in zip(*np.nonzero(off))]
    top = pd.DataFrame(pairs, columns=["label_asli", "diprediksi_sebagai", "jumlah"]) \
        .sort_values("jumlah", ascending=False).head(10)
    top.to_csv(os.path.join(out_dir, f"top_confusions_{name}.csv"), index=False)

    results = dict(model=name, params=count_params(model), test_accuracy=acc, test_f1_macro=f1,
                   best_val_accuracy=best_val, train_time_sec=round(train_time, 1),
                   epochs_trained=int(history.epoch.max()))
    with open(os.path.join(out_dir, f"results_{name}.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n===== {name} =====")
    for k, v in results.items():
        print(f"{k:>18}: {v}")
    print("Top kelas tertukar:\n", top.to_string(index=False))
    return results


# ----------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------
def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_dl, val_dl, test_dl, classes = get_dataloaders(
        args.data_dir, args.img_size, args.batch_size, args.num_workers)
    num_classes = len(classes)
    print(f"Kelas: {num_classes} | train {len(train_dl.dataset)} | "
          f"val {len(val_dl.dataset)} | test {len(test_dl.dataset)}")

    all_results = []

    # --- (opsional) Baseline Simple CNN ---
    if args.train_baseline:
        set_seed(args.seed)
        base = SimpleCNN(num_classes, args.img_size).to(device)
        print(f"\nSimpleCNN params: {count_params(base):,}")
        hist, t, bv = train_model(base, "simple_cnn", train_dl, val_dl, device, args, args.out_dir)
        all_results.append(evaluate_and_save(base, "simple_cnn", hist, t, bv, test_dl, classes,
                                             device, args.out_dir))

    # --- Model 2 ---
    set_seed(args.seed)
    model = ModifiedCNN(num_classes).to(device)
    print(model)
    print(f"ModifiedCNN params: {count_params(model):,}")
    hist, t, bv = train_model(model, "model2", train_dl, val_dl, device, args, args.out_dir)
    all_results.append(evaluate_and_save(model, "model2", hist, t, bv, test_dl, classes,
                                         device, args.out_dir))

    # --- Perbandingan dengan baseline ---
    # Kalau teman (Model 1) menyimpan hasil di outputs/model1/results.json, otomatis ikut dibandingkan.
    m1_path = os.path.join("outputs", "model1", "results.json")
    if os.path.exists(m1_path):
        with open(m1_path) as f:
            all_results.insert(0, json.load(f))
    if len(all_results) > 1:
        comp = pd.DataFrame(all_results)
        comp.to_csv(os.path.join(args.out_dir, "comparison.csv"), index=False)
        print("\n===== Perbandingan =====")
        print(comp.to_string(index=False))


if __name__ == "__main__":  # wajib di Windows karena num_workers > 0
    main()
