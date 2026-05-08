import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize

from model import EnsembleEDLM
from utils.dataset import FoldSequenceDataset


def train_epoch(model, dataloader, optimizer, criterion, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    preds, labels = [], []

    for batch in dataloader:
        features = batch["features"].to(device)
        targets = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        preds.extend(logits.argmax(dim=1).cpu().numpy())
        labels.extend(targets.cpu().numpy())

    return total_loss / len(dataloader), preds, labels


def evaluate(model, dataloader, criterion, device, num_classes=5):
    model.eval()
    total_loss = 0
    preds, labels, probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            targets = batch["label"].to(device)

            logits = model(features)
            loss = criterion(logits, targets)

            total_loss += loss.item()
            probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            labels.extend(targets.cpu().numpy())

    probs = np.array(probs)
    labels = np.array(labels)
    preds = np.array(preds)

    labels_bin = label_binarize(labels, classes=range(num_classes))
    per_class_auc = []
    for i in range(num_classes):
        try:
            auc = roc_auc_score(labels_bin[:, i], probs[:, i])
            per_class_auc.append(auc)
        except ValueError:
            per_class_auc.append(0.0)

    try:
        overall_auc = roc_auc_score(labels, probs, multi_class="ovr", average="weighted")
    except ValueError:
        overall_auc = 0.0

    cm = confusion_matrix(labels, preds)

    return total_loss / len(dataloader), preds, labels, overall_auc, per_class_auc, cm, probs


def train_and_evaluate(config):
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    with open(config.loso_splits_path, "rb") as f:
        loso_splits = pickle.load(f)

    all_fold_names = sorted(loso_splits.keys())
    if config.num_folds is not None and config.num_folds > 0:
        fold_names = all_fold_names[:min(config.num_folds, len(all_fold_names))]
    else:
        fold_names = all_fold_names
    num_folds = len(fold_names)
    print(f"Using {num_folds} folds for training/evaluation (configured num_folds={config.num_folds}).")

    all_fold_accs = []
    all_fold_f1s = []
    all_fold_aucs = []
    all_per_class_aucs = []
    processed_fold_names = []
    all_preds, all_labels = [], []
    all_cm = np.zeros((config.num_classes, config.num_classes), dtype=int)

    for fold_idx, fold_name in enumerate(fold_names):
        print(f"\n{'='*50}")
        print(f"Training Ensemble Model - {fold_name} ({fold_idx + 1}/{num_folds})")
        print(f"{'='*50}")

        fold_dir = os.path.join(config.features_3d_dir, fold_name)

        if not os.path.exists(fold_dir):
            print(f"Skipping {fold_name}: feature directory not found. Run feature extraction first.")
            continue

        train_dataset = FoldSequenceDataset(fold_dir, prefix="train")
        test_dataset = FoldSequenceDataset(fold_dir, prefix="test")

        if len(train_dataset) == 0 or len(test_dataset) == 0:
            print(f"Skipping {fold_name}: no data found.")
            continue

        train_loader = DataLoader(train_dataset, batch_size=config.ensemble_batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=config.ensemble_batch_size, shuffle=False, num_workers=4)

        model = EnsembleEDLM(num_classes=config.num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.ensemble_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

        best_val_loss = float("inf")
        best_model_state = None
        patience = 3
        patience_counter = 0

        for epoch in range(config.ensemble_epochs):
            train_loss, train_preds, train_labels = train_epoch(
                model, train_loader, optimizer, criterion, device, max_grad_norm=1.0
            )
            scheduler.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in test_loader:
                    features = batch["features"].to(device)
                    targets = batch["label"].to(device)
                    logits = model(features)
                    val_loss += criterion(logits, targets).item()
            val_loss /= len(test_loader)

            if (epoch + 1) % 1 == 0:
                train_acc = accuracy_score(train_labels, train_preds)
                print(f"  Epoch {epoch + 1}/{config.ensemble_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        if best_model_state:
            model.load_state_dict(best_model_state)
        model.to(device)

        test_loss, test_preds, test_labels, test_auc, per_class_auc, cm, _ = evaluate(
            model, test_loader, criterion, device, num_classes=config.num_classes
        )

        fold_acc = accuracy_score(test_labels, test_preds)
        fold_f1 = f1_score(test_labels, test_preds, average="weighted")

        all_fold_accs.append(fold_acc)
        all_fold_f1s.append(fold_f1)
        all_fold_aucs.append(test_auc)
        all_per_class_aucs.append(per_class_auc)
        processed_fold_names.append(fold_name)
        all_preds.extend(test_preds)
        all_labels.extend(test_labels)
        all_cm += cm

        test_subject = loso_splits[fold_name]["test_subject"]
        print(f"  {fold_name} ({test_subject}) | Acc: {fold_acc:.4f} | F1: {fold_f1:.4f} | AUC: {test_auc:.4f}")
        print(f"  Per-class AUC: {[f'{a:.3f}' for a in per_class_auc]}")

    print(f"\n{'='*50}")
    print("Overall Results (LOSO Cross-Validation)")
    print(f"{'='*50}")
    print(f"Mean Accuracy: {np.mean(all_fold_accs):.4f} (+/- {np.std(all_fold_accs):.4f})")
    print(f"Mean F1 Score: {np.mean(all_fold_f1s):.4f} (+/- {np.std(all_fold_f1s):.4f})")
    print(f"Mean AUC:      {np.mean(all_fold_aucs):.4f} (+/- {np.std(all_fold_aucs):.4f})")

    mean_per_class_auc = np.mean(all_per_class_aucs, axis=0)
    print(f"\nPer-class Mean AUC:")
    for i, auc in enumerate(mean_per_class_auc):
        print(f"  Class {i}: {auc:.4f}")

    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(all_cm)
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds))

    results_dir = config.output_dir
    os.makedirs(results_dir, exist_ok=True)
    np.save(os.path.join(results_dir, "predictions.npy"), np.array(all_preds))
    np.save(os.path.join(results_dir, "labels.npy"), np.array(all_labels))
    np.save(os.path.join(results_dir, "fold_accuracies.npy"), np.array(all_fold_accs))
    np.save(os.path.join(results_dir, "fold_f1s.npy"), np.array(all_fold_f1s))
    np.save(os.path.join(results_dir, "fold_aucs.npy"), np.array(all_fold_aucs))
    np.save(os.path.join(results_dir, "confusion_matrix.npy"), all_cm)

    import json
    fold_results = {}
    for fold_name, acc, f1, auc, pcauc in zip(processed_fold_names, all_fold_accs, all_fold_f1s, all_fold_aucs, all_per_class_aucs):
        fold_results[fold_name] = {
            "accuracy": acc, "f1": f1, "auc": auc,
            "per_class_auc": {str(i): float(a) for i, a in enumerate(pcauc)}
        }
    with open(os.path.join(results_dir, "fold_results.json"), "w") as f:
        json.dump(fold_results, f, indent=2)

    print(f"Results saved to {results_dir}")
