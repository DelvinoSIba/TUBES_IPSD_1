import os
import glob
import json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sn

from torch.utils.data import DataLoader
from Utils.getData import getImageLabel
from Models.CNN import SimpleCNN
from Models.model2_modified_cnn import ModifiedCNN
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, auc)


def get_model(name, img_size, num_classes):
    if name == "SimpleCNN":
        return SimpleCNN(input_dim=img_size, input_c=3, output=num_classes)
    if name == "Model2":
        return ModifiedCNN(input_c=3, output=num_classes)
    raise ValueError("MODEL_NAME harus 'SimpleCNN' atau 'Model2'")


def predict(model, loader, device):
    """Mengembalikan label asli, label prediksi, dan probabilitas softmax (untuk ROC-AUC)."""
    prediction, ground_truth, probs = [], [], []
    with torch.no_grad():
        model.eval()
        for batch, (src, trg) in enumerate(loader):
            src = torch.permute(src, (0, 3, 1, 2)).to(device)
            prob = torch.softmax(model(src), dim=1).cpu().numpy()
            probs.append(prob)
            prediction.extend(prob.argmax(1))
            ground_truth.extend(torch.argmax(trg, dim=1).numpy())
    return np.array(ground_truth), np.array(prediction), np.concatenate(probs)


def macro_roc(y_true, probs, n_classes):
    """Kurva ROC macro-average (one-vs-rest) untuk multi-kelas."""
    y_bin = label_binarize(y_true, classes=range(n_classes))
    grid = np.linspace(0, 1, 200)
    tprs = []
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
        tprs.append(np.interp(grid, fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0.0
    return grid, mean_tpr, auc(grid, mean_tpr)


def main():
    # ===== Konfigurasi =====
    MODEL_NAME = "Model2"          # "SimpleCNN" atau "Model2"  (samakan dengan Train.py)
    K_FOLDS = 5
    IMG_SIZE = 128                 # samakan dengan Train.py
    BATCH_SIZE = 32
    OUTPUT_DIR = f"./Output/{MODEL_NAME}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data = getImageLabel(folder="./Dataset/test", img_size=IMG_SIZE, augment=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    classes = test_data.classes
    n_classes = len(classes)

    rows, roc_curves = [], []
    cm_total = np.zeros((n_classes, n_classes), dtype=int)

    for fold in range(1, K_FOLDS + 1):
        ckpt_path = f"{OUTPUT_DIR}/fold{fold}_best.pt"
        if not os.path.exists(ckpt_path):
            print(f"[Fold {fold}] checkpoint tidak ditemukan ({ckpt_path}), dilewati.")
            continue

        model = get_model(MODEL_NAME, IMG_SIZE, n_classes).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        y_true, y_pred, probs = predict(model, test_loader, device)

        row = dict(
            fold=fold,
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, average='macro', zero_division=0),
            recall=recall_score(y_true, y_pred, average='macro', zero_division=0),
            f1_score=f1_score(y_true, y_pred, average='macro', zero_division=0),
            roc_auc=roc_auc_score(y_true, probs, multi_class='ovr', average='macro'),
            train_time_sec=checkpoint.get('train_time'),
            epochs_trained=checkpoint.get('epochs_trained'),
        )
        rows.append(row)
        print(f"[Fold {fold}] acc = {row['accuracy']:.4f} | precision = {row['precision']:.4f} | "
              f"recall = {row['recall']:.4f} | f1 = {row['f1_score']:.4f} | roc-auc = {row['roc_auc']:.4f}")

        # confusion matrix per fold
        cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
        cm_total += cm
        plt.figure(figsize=(18, 15))
        sn.heatmap(pd.DataFrame(cm, index=classes, columns=classes), cmap="Blues")
        plt.title(f"Confusion Matrix {MODEL_NAME} - Fold {fold} (accuracy = {row['accuracy']:.3f})")
        plt.xlabel("Prediksi"); plt.ylabel("Label asli")
        plt.xticks(fontsize=6, rotation=90); plt.yticks(fontsize=6)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_fold{fold}.png", dpi=120)
        plt.close()

        roc_curves.append((fold, *macro_roc(y_true, probs, n_classes)))

    if not rows:
        print("Tidak ada checkpoint yang bisa dievaluasi. Jalankan Train.py dulu.")
        return

    # ---------- Rata-rata 5 fold ----------
    df = pd.DataFrame(rows)
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    mean, std = df[metrics].mean(), df[metrics].std()
    cols = metrics + ["train_time_sec"]
    summary = pd.concat([df, pd.DataFrame([dict(fold="mean", **df[cols].mean().to_dict()),
                                           dict(fold="std", **df[cols].std().to_dict())])],
                        ignore_index=True)
    summary.to_csv(f"{OUTPUT_DIR}/metrics_per_fold.csv", index=False)

    print(f"\n===== {MODEL_NAME}: rata-rata {len(df)}-fold pada Dataset/test =====")
    for m in metrics:
        print(f"{m:>10} = {mean[m]:.4f} ± {std[m]:.4f}")

    # ---------- Confusion matrix gabungan (jumlah semua fold) ----------
    plt.figure(figsize=(18, 15))
    sn.heatmap(pd.DataFrame(cm_total, index=classes, columns=classes), cmap="Blues")
    plt.title(f"Confusion Matrix {MODEL_NAME} - gabungan {len(df)} fold")
    plt.xlabel("Prediksi"); plt.ylabel("Label asli")
    plt.xticks(fontsize=6, rotation=90); plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_all_folds.png", dpi=150)
    plt.close()

    # ---------- Kurva ROC (macro-average, one-vs-rest) semua fold ----------
    plt.figure(figsize=(7, 6))
    for fold, fpr, tpr, roc_auc in roc_curves:
        plt.plot(fpr, tpr, label=f"Fold {fold} (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (macro-average OvR) - {MODEL_NAME}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=150)
    plt.close()

    # ---------- Kelas yang paling sering tertukar (gabungan semua fold) ----------
    off = cm_total.copy()
    np.fill_diagonal(off, 0)
    pairs = [(classes[i], classes[j], int(off[i, j])) for i, j in zip(*np.nonzero(off))]
    top = pd.DataFrame(pairs, columns=["label_asli", "diprediksi_sebagai", "jumlah"]) \
        .sort_values("jumlah", ascending=False).head(10)
    top.to_csv(f"{OUTPUT_DIR}/top_confusions.csv", index=False)
    print("\nTop kelas tertukar (gabungan semua fold):\n", top.to_string(index=False))

    # ---------- Ringkasan untuk perbandingan antar model ----------
    model = get_model(MODEL_NAME, IMG_SIZE, n_classes)
    results = dict(model=MODEL_NAME, params=sum(p.numel() for p in model.parameters()),
                   n_folds=len(df),
                   **{f"{m}_mean": float(mean[m]) for m in metrics},
                   **{f"{m}_std": float(std[m]) for m in metrics},
                   train_time_sec_mean=float(df.train_time_sec.mean()))
    with open(f"{OUTPUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # tabel markdown siap tempel ke README.md
    with open(f"{OUTPUT_DIR}/results.md", "w") as f:
        f.write(f"| Fold | Accuracy | Precision | Recall | F1-score | ROC-AUC |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['fold']} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
                    f"{r['f1_score']:.4f} | {r['roc_auc']:.4f} |\n")
        f.write("| **Rata-rata** | " + " | ".join(f"**{mean[m]:.4f} ± {std[m]:.4f}**" for m in metrics) + " |\n")

    all_results = []
    for path in sorted(glob.glob("./Output/*/results.json")):
        with open(path) as f:
            all_results.append(json.load(f))
    if len(all_results) > 1:
        comp = pd.DataFrame(all_results)
        comp.to_csv("./Output/comparison.csv", index=False)
        print("\n===== Perbandingan model =====\n", comp.to_string(index=False))


if __name__ == "__main__":
    main()
