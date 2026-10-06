import argparse
import os
import random
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import faulthandler
faulthandler.enable()
from deel.torchlip import BatchCentering
import traceback
import warnings
import logging

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import optuna
except Exception:
    optuna = None

logger = optuna.logging.get_logger("optuna")

from prep_data_get_dimensions_numpy import prep_data_get_dimensions_numpy


def parse_hidden_layers(value):
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def tf_dataset_to_numpy(dataset):
    x_list = []
    y_list = []
    twice = False

    for batch in dataset:
        try:
            if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                features, labels = batch[0], batch[1]
            else:
                continue
        except Exception:
            continue

        if not twice:
            print(f"{features.shape=}")
            print(f"{labels.shape=}")

        features_np = np.asarray(features.numpy() if hasattr(features, "numpy") else features)
        labels_np = np.asarray(labels.numpy() if hasattr(labels, "numpy") else labels)

        if not twice:
            print(f"{features_np.shape=}")
            print(f"{labels_np.shape=}")
            twice=True
        # if features_np.ndim == 0:
        #     features_np = features_np.reshape(1, 1)
        # elif features_np.ndim == 1:
        #     features_np = features_np.reshape(1, -1)

        # if labels_np.ndim == 0:
        #     labels_np = labels_np.reshape(1)

        x_list.append(features_np)
        y_list.append(labels_np)

    if not x_list:
        return None, None

    x = np.stack(x_list, axis=0)
    y = np.stack(y_list, axis=0)
    print(f"{x.shape=}")
    print(f"{y.shape=}")
    print(f"{y[:5]=}")
    print("Contains NaN:", np.isnan(x).any())
    print("Contains Inf:", np.isinf(x).any())
    return x, y


def _get_deel_torch_layers():
    try:
        from deel import torchlip
    except Exception:
        print(" no deel torch layers ")
        return None

    spectral_linear = getattr(torchlip, "SpectralLinear", None)
    frobenius_linear = getattr(torchlip, "FrobeniusLinear", None)
    group_sort = getattr(torchlip, "GroupSort", None)
    return {
        "torchlip": torchlip,
        "spectral_linear": spectral_linear,
        "frobenius_linear": frobenius_linear,
        "group_sort": group_sort,
    }


def get_deel_losses():
    try:
        from deel.torchlip import KRLoss, HKRLoss, HingeMarginLoss, HKRMulticlassLoss
    except Exception:
        print('Couldn\'t import losses')
        return {
            "KRLoss": None,
            "HKRLoss": None,
            "HingeMarginLoss": None,
            "HKRMulticlassLoss": None,
        }

    return {
        "KRLoss": KRLoss,
        "HKRLoss": HKRLoss,
        "HingeMarginLoss": HingeMarginLoss,
        "HKRMulticlassLoss": HKRMulticlassLoss,
    }



def _make_batch_centering_layer(num_features):
    return BatchCentering(num_features=num_features, bias=True)


def build_torch_model(architecture, input_dim, output_dim, hidden_layers, device, batch_center=False, width_multiplier=1):
    hidden_layers = [x*width_multiplier for x in hidden_layers]
    logger.info(f"{width_multiplier=}: {hidden_layers}")
    if architecture != "deel-torchlib":
        layers = []
        in_dim = input_dim
        for units in hidden_layers:
            if batch_center:
                layers.append(_make_batch_centering_layer(in_dim))
            layers.append(nn.Linear(in_dim, units))
            layers.append(nn.ReLU())
            in_dim = units

        if batch_center:
            layers.append(_make_batch_centering_layer(in_dim))
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers).to(device)

    deel = _get_deel_torch_layers()
    if deel is None or deel["spectral_linear"] is None or deel["frobenius_linear"] is None:
        layers = []
        in_dim = input_dim
        for units in hidden_layers:
            if batch_center:
                layers.append(_make_batch_centering_layer(in_dim))
            layers.append(nn.Linear(in_dim, units))
            layers.append(nn.ReLU())
            in_dim = units

        if batch_center:
            layers.append(_make_batch_centering_layer(in_dim))
        layers.append(nn.Linear(in_dim, output_dim))
        print('non-robust architecture')
        return nn.Sequential(*layers).to(device)

    torchlip = deel["torchlip"]
    spectral_linear = deel["spectral_linear"]
    frobenius_linear = deel["frobenius_linear"]
    group_sort = deel["group_sort"]

    seq = []
    in_dim = input_dim
    for units in hidden_layers:
        if batch_center:
            seq.append(_make_batch_centering_layer(in_dim))
        seq.append(spectral_linear(in_dim, units))
        if group_sort is not None:
            seq.append(group_sort())
        in_dim = units

    if batch_center:
        seq.append(_make_batch_centering_layer(in_dim))
    seq.append(frobenius_linear(in_dim, output_dim))
    model = torchlip.Sequential(*seq).to(device)
    return model


def make_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    print(f'{batch_size=}')
    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def evaluate(model, loader, device, binary):
    model.eval()
    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(device)
            logits = model(Xb)
            logits_np = logits.detach().cpu().numpy()
            y_true_list.append(yb.cpu().numpy())

            if binary:
                pred = (logits_np.reshape(-1) >= 0.0).astype(int)
            else:
                pred = np.argmax(logits_np, axis=1)
            y_pred_list.append(pred)

    if not y_true_list:
        return 0.0

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    if y_true.ndim > 1 and y_true.shape[1] == 1:
        y_true = y_true.reshape(-1)
    elif y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)

    if binary:
        return accuracy_score(y_true, y_pred)
    return f1_score(y_true, y_pred, average="weighted")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []

    for Xb, yb in loader:
        Xb = Xb.to(device).float()
        yb = yb.to(device).float()

        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else -1.0


def choose_criterion(architecture, binary, alpha=None, min_margin=None):
    losses = get_deel_losses()

    if architecture == "deel-torchlib":
        if binary:
            if losses["HKRLoss"] is not None and alpha is not None and min_margin is not None:
                return losses["HKRLoss"](alpha=alpha, min_margin=min_margin)
            return nn.BCEWithLogitsLoss()

        if losses["HKRMulticlassLoss"] is not None:
            return losses["HKRMulticlassLoss"](alpha=alpha or 1, min_margin=min_margin or 0.25)
        if losses["HKRLoss"] is not None:
            return losses["HKRLoss"](alpha=alpha or 1, min_margin=min_margin or 0.25)
        return nn.CrossEntropyLoss()

    if binary:
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


def run_trial(X_train, y_train, X_val, y_val, architecture, hidden_layers, learning_rate, batch_size, device, binary, alpha=None, min_margin=None, epochs=5, batch_center=False, width_multiplier=1):
    output_dim = 1 if binary else int(y_train.shape[1]) if y_train.ndim > 1 and y_train.shape[1] > 1 else 1
    model = build_torch_model(architecture, X_train.shape[1], output_dim, hidden_layers, device, batch_center=batch_center, width_multiplier=width_multiplier)
    criterion = choose_criterion(architecture, binary, alpha=alpha, min_margin=min_margin)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, fused=False)

    train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val, batch_size)

    for _ in tqdm(range(epochs)):
        train_one_epoch(model, train_loader, optimizer, criterion, device)

    score = evaluate(model, val_loader, device, binary)
    print(f"Epoch {_}: score {score}")
    return score, model


def main():
    parser = argparse.ArgumentParser(description="Translated hp_tune_script.py using deel-torchlib")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset folder used by the notebooks")
    parser.add_argument("--architecture", choices=["baseline", "deel-torchlib"], default="deel-torchlib")
    parser.add_argument("--hidden_layers", type=str, default="64,32,16", help="Comma-separated hidden layer sizes")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--project_name", type=str, default="hp_tuning_script")
    parser.add_argument("--study_name", type=str, default="hp_tuning_script", help="Name for the Optuna study")
    parser.add_argument("--batch_center", action="store_true", help="Use BatchCentering before each linear block, default off")
    parser.add_argument("--diffex", action="store_true", help="Use only differentially expressed genes")
    parser.add_argument("--only_train", action="store_true", help="Run only the training based on existing tuning file (make sure to use the same project names etc)")
    args = parser.parse_args()

    hidden_layers = parse_hidden_layers(args.hidden_layers)
    if not hidden_layers:
        hidden_layers = [64, 32, 16]

    data = prep_data_get_dimensions_numpy(args.data, args.diffex)
    binary = data['binary']

    X_train, y_train = data["train_features"], data["train_labels"]
    X_val, y_val = data["val_features"], data["val_labels"]

    if X_train is None:
        raise RuntimeError("Could not convert tf.data dataset to numpy arrays")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    if not args.only_train:
        if optuna is not None:
            print("Using Optuna for hyperparameter search")
            if args.batch_center:
                print("BatchCentering enabled in the model architecture.")
            sample_model = build_torch_model(args.architecture, X_train.shape[1], int(data["output_dim"]), hidden_layers, device, batch_center=args.batch_center)

            def objective(trial):
                learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
                batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
                width_multiplier = trial.suggest_int("width_multiplier", 1, 16, log=True)
                alpha = None
                min_margin = None
                if args.architecture == "deel-torchlib":
                    alpha = trial.suggest_float("alpha", 0, 1, log=False)
                    min_margin = trial.suggest_float("min_margin", 0.01, 1.0, log=True)

                score, _ = run_trial(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    args.architecture,
                    hidden_layers,
                    learning_rate,
                    batch_size,
                    device,
                    binary,
                    alpha=alpha,
                    min_margin=min_margin,
                    epochs=min(4, args.max_epochs),
                    batch_center=args.batch_center,
                    width_multiplier = width_multiplier
                )
                return score

            out_dir = os.path.join(args.project_name)
            os.makedirs(out_dir, exist_ok=True)
            study = optuna.create_study(direction="maximize", study_name=args.project_name)
            print(f"Optuna study name: {study.study_name}")
            study.optimize(objective, n_trials=args.trials)

            for i, trial in enumerate(study.trials):
                row = {
                    "trial": i,
                    "learning_rate": trial.params.get("learning_rate"),
                    "batch_size": trial.params.get("batch_size"),
                    "width_multiplier": trial.params.get("width_multiplier"),
                    "score": trial.value,
                }
                if args.architecture == "deel-torchlib":
                    row["alpha"] = trial.params.get("alpha")
                    row["min_margin"] = trial.params.get("min_margin")
                results.append(row)

            try:
                study_df = study.trials_dataframe()
                study_df.to_csv(os.path.join(out_dir, "optuna_study_trials.csv"), index=False)
            except Exception:
                pass
        else:
            print("Optuna not available; using random search")
            for idx in range(args.trials):
                learning_rate = 10 ** np.random.uniform(-4, -2)
                batch_size = random.choice([32, 64, 128, 256])
                alpha = None
                min_margin = None
                if args.architecture == "deel-torchlib":
                    alpha = np.random.uniform(0, 1)
                    min_margin = float(10 ** np.random.uniform(-2, 0))

                score, _ = run_trial(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    args.architecture,
                    hidden_layers,
                    learning_rate,
                    batch_size,
                    device,
                    binary,
                    alpha=alpha,
                    min_margin=min_margin,
                    epochs=min(4, args.max_epochs),
                    batch_center=args.batch_center,
                    width_multiplier=width_multiplier,
                )
                row = {
                    "trial": idx,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "width_multiplier": width_multiplier,
                    "score": score,
                }
                if args.architecture == "deel-torchlib":
                    row["alpha"] = alpha
                    row["min_margin"] = min_margin
                results.append(row)
                print()
                print(f"Trial {idx}: lr={learning_rate:.2e}, batch={batch_size}, width_multi={width_multiplier}, score={score:.4f}")

        df = pd.DataFrame(results).sort_values("score", ascending=False)
        out_dir = os.path.join(args.project_name)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "top_trials.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved trials to {csv_path}")
    else: #args.only_train is true
        out_dir = os.path.join(args.project_name)
        csv_path = os.path.join(out_dir, "top_trials.csv")
        try:
            df = pd.read_csv(csv_path)
            os.makedirs(out_dir, exist_ok=True)
            print(f"Loaded tuning values from {csv_path}")
        except FileNotFoundError as e:
            print(f"Error encountered: {e}")
            error_trace = traceback.format_exc()
            print(error_trace)


    best = df.iloc[0]
    best_lr = float(best["learning_rate"])
    best_batch = int(best["batch_size"])
    best_alpha = float(best["alpha"]) if "alpha" in best.index else None
    best_min_margin = float(best["min_margin"]) if "min_margin" in best.index else None
    best_width_multiplier = int(best["width_multiplier"]) if "width_multiplier" in best.index else None

    output_dim = int(data["output_dim"])
    final_model = build_torch_model(args.architecture, X_train.shape[1], output_dim, hidden_layers, device, batch_center=args.batch_center, width_multiplier=best_width_multiplier)
    criterion = choose_criterion(args.architecture, binary, alpha=best_alpha, min_margin=best_min_margin)
    optimizer = torch.optim.Adam(params=final_model.parameters(), lr=best_lr)
    train_loader, val_loader = make_dataloaders(X_train, y_train, X_val, y_val, best_batch)

    patience = 5
    lowest_loss = float('inf')  
    epochs_no_improve = 0
    best_model_weights = None

    for epoch in tqdm(range(args.max_epochs)):
        train_loss = train_one_epoch(final_model, train_loader, optimizer, criterion, device)
        val_score = evaluate(final_model, val_loader, device, binary)
        print(f"Retrain epoch {epoch + 1}/{args.max_epochs}: loss={train_loss:.4f}, val_score={val_score:.4f}")
        
        if train_loss < lowest_loss:
            lowest_loss = train_loss
            epochs_no_improve = 0
            import copy
            best_model_weights = copy.deepcopy(final_model.state_dict())
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    if best_model_weights is not None:
        final_model.load_state_dict(best_model_weights)

    model_path = os.path.join(out_dir, "best_model.pt")
    torch.save(final_model.state_dict(), model_path)
    print(f"Saved best model to {model_path}")

if __name__ == "__main__":
    main()

