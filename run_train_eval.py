import time
import os
import glob
import random
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset import temporal_dataset, frequency_dataset, multi_dataset
from model import S4classifier, FSclassifier, Multiclassifier
from fs_modules import gaussian_adj

warnings.filterwarnings("ignore")


# =========================
# Reproducibility
# =========================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Model forward dispatch
# =========================
def forward_model(model, batch, domain, device):
    if domain == "temporal":
        return model(batch["segment"].to(device))

    if domain == "frequency":
        return model(batch["de_grid"].to(device))

    if domain == "multi":
        return model(batch["segment"].to(device), batch["de_grid"].to(device))

    raise ValueError(f"Unsupported domain: {domain}")


# =========================
# Training
# =========================
def train_and_save_model(
    train_loader,
    domain,
    train_adj,
    subject_name="",
    fold=0,
    model_dir="",
    adj_dir="",
    model=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    for _ in range(50):
        model.train()

        for batch in train_loader:
            labels = batch["label"].to(device).float().view(-1, 1)
            logits = forward_model(model, batch, domain, device)

            optimizer.zero_grad()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    # save model checkpoint
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{subject_name}_fold{fold}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {subject_name}_fold{fold}.pth")

    # save learned adjacency for graph-based settings
    if domain in ["frequency", "multi"] and train_adj:
        os.makedirs(adj_dir, exist_ok=True)
        learned_adj = model.fs_model.A_raw.detach().cpu().numpy()
        adj_path = os.path.join(adj_dir, f"{subject_name}_fold{fold}_adj.npy")
        np.save(adj_path, learned_adj)
        print("Learned adjacency matrix saved")


# =========================
# Evaluation
# =========================
def evaluate_model(
    val_loader,
    domain,
    subject_name="",
    fold=0,
    log_dir="",
    model_dir="",
    model=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(model_dir, f"{subject_name}_fold{fold}.pth")

    # load trained model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    # collect predictions
    with torch.no_grad():
        for batch in val_loader:
            labels = batch["label"].to(device).float().view(-1, 1)
            logits = forward_model(model, batch, domain, device)

            preds = (torch.sigmoid(logits) > 0.5).int()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu().int())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # confusion matrix statistics
    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    tn = ((all_preds == 0) & (all_labels == 0)).sum().item()
    fp = ((all_preds == 1) & (all_labels == 0)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()

    # evaluation metrics
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    metrics = {
        "Accuracy": round(acc * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall": round(rec * 100, 2),
        "Specificity": round(spec * 100, 2),
        "F1-Score": round(f1 * 100, 2),
    }

    # save fold-wise metrics
    os.makedirs(log_dir, exist_ok=True)
    metric_path = os.path.join(log_dir, f"{subject_name}_fold{fold}_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metric_path, index=False)


# =========================
# Main
# =========================
if __name__ == "__main__":
    set_seed(42)

    parser = argparse.ArgumentParser()

    # experiment setting
    parser.add_argument("--domain", type=str, required=True, choices=["temporal", "frequency", "multi"])
    parser.add_argument("--emotion", type=str, required=True, choices=["valence", "arousal", "dominance"])
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)

    # external paths
    parser.add_argument("--data_root", type=str, required=True, help="dataset root directory")
    parser.add_argument("--output_root", type=str, required=True, help="output root directory")

    args = parser.parse_args()

    domain = args.domain
    train_adj = args.train
    emotion = args.emotion

    # dataset and output paths
    root_dir = os.path.join(args.data_root, emotion)
    base_path = args.output_root

    if domain in ["frequency", "multi"]:
        suffix = "train" if train_adj else "not"
        log_dir = os.path.join(base_path, domain, emotion, suffix, "log")
        model_dir = os.path.join(base_path, domain, emotion, suffix, "model")
        adj_dir = os.path.join(base_path, domain, emotion, suffix, "adj")
    else:
        log_dir = os.path.join(base_path, domain, emotion, "log")
        model_dir = os.path.join(base_path, domain, emotion, "model")
        adj_dir = ""

    # S4 configuration
    s4_config = {
        "d_input": 14,
        "d_output": 256,
        "d_model": 256,
        "n_layers": 4,
        "dropout": 0.3,
        "prenorm": False,
    }

    # frequency-spatial configuration
    fs_config = {
        "in_channels": 5,
        "num_classes": 1,
        "dropout": 0.3,
        "initial_adj": gaussian_adj(),
        "adj_train": train_adj,
    }

    # subject-level split
    subject_dirs = sorted(glob.glob(os.path.join(root_dir, "subject_*")))

    # training time log
    os.makedirs(log_dir, exist_ok=True)
    time_log_df = pd.DataFrame(columns=["Subject", "Fold", "TrainTime(s)", "EvalTime(s)"])

    print("Start training pipeline")

    for subject_path in subject_dirs:
        subject_name = os.path.basename(subject_path)
        print(f"Processing subject: {subject_name}")

        for fold in range(10):
            print(f"Fold {fold} | Domain: {domain}")

            # dataset and model selection
            if domain == "temporal":
                train_set = temporal_dataset(subject_dir=subject_path, fold_id=fold, mode="train")
                val_set = temporal_dataset(subject_dir=subject_path, fold_id=fold, mode="val")
                model = S4classifier(s4_config=s4_config)

            elif domain == "frequency":
                train_set = frequency_dataset(subject_dir=subject_path, fold_id=fold, mode="train")
                val_set = frequency_dataset(subject_dir=subject_path, fold_id=fold, mode="val")
                model = FSclassifier(fs_config=fs_config)

            elif domain == "multi":
                train_set = multi_dataset(subject_dir=subject_path, fold_id=fold, mode="train")
                val_set = multi_dataset(subject_dir=subject_path, fold_id=fold, mode="val")
                model = Multiclassifier(s4_config=s4_config, fs_config=fs_config)

            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

            # train model
            start_train_time = time.time()
            train_and_save_model(
                train_loader=train_loader,
                domain=domain,
                train_adj=train_adj,
                subject_name=subject_name,
                fold=fold,
                model_dir=model_dir,
                adj_dir=adj_dir,
                model=model,
            )
            elapsed_train = time.time() - start_train_time

            # evaluate model
            start_eval_time = time.time()
            evaluate_model(
                val_loader=val_loader,
                domain=domain,
                subject_name=subject_name,
                fold=fold,
                log_dir=log_dir,
                model_dir=model_dir,
                model=model,
            )
            elapsed_eval = time.time() - start_eval_time

            time_log_df.loc[len(time_log_df)] = [
                subject_name,
                f"fold_{fold}",
                elapsed_train,
                elapsed_eval,
            ]

    # save execution time log
    time_log_path = os.path.join(log_dir, "time_log.csv")
    time_log_df.to_csv(time_log_path, index=False)
    print(f"Time log saved: {time_log_path}")