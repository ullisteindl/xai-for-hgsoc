import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.metrics import classification_report, confusion_matrix, f1_score
logger = optuna.logging.get_logger("optuna")



from prep_data_get_dimensions_numpy import prep_data_get_dimensions_numpy

# Reuse the exact model factory used by the training script.
from hp_tune_deel_torchlip import build_torch_model, parse_hidden_layers

try:
    from xplique.wrappers import TorchWrapper
    from xplique.attributions import (
        GradientInput,
        IntegratedGradients,
        Saliency,
        SmoothGrad,
        VarGrad,
        GuidedBackprop,
        SmoothGrad,
        SquareGrad,
        VarGrad,
        Occlusion,
        Lime,
        KernelShap
    )
    from xplique.metrics import Complexity, Deletion, Insertion, MuFidelity, Sparseness
except Exception as exc:  # pragma: no cover - optional dependency on runtime
    TorchWrapper = None
    GradientInput = IntegratedGradients = Saliency = SmoothGrad = VarGrad = GuidedBackprop = SmoothGrad = SquareGrad = VarGrad = Occlusion = Lime = KernelShap = None
    Complexity = Deletion = Insertion = MuFidelity = Sparseness = None
    XPLIQUE_IMPORT_ERROR = exc
else:
    XPLIQUE_IMPORT_ERROR = None


def prepare_labels(labels, output_dim, binary):
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels.reshape(-1)

    if binary:
        return labels.astype(np.float32).reshape(-1, 1)

    if labels.ndim == 1:
        labels = np.asarray(labels, dtype=int)
        labels = np.eye(output_dim, dtype=np.float32)[labels]
    return labels.astype(np.float32)


def compute_test_evaluation(model, X_test, y_test, binary, output_dim, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X_test, dtype=torch.float32, device=device))
        logits = logits.detach().cpu().numpy()

    if binary:
        probabilities = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
        predicted_classes = np.round(probabilities).astype(int)
    else:
        probabilities = logits
        predicted_classes = np.argmax(logits, axis=-1)

    true_labels = np.asarray(y_test)
    if true_labels.ndim > 1 and true_labels.shape[1] == 1:
        true_labels = true_labels.reshape(-1)
    if true_labels.ndim > 1 and true_labels.shape[-1] > 1:
        true_labels = np.argmax(true_labels, axis=-1)

    if binary:
        true_labels = true_labels.astype(int)
    else:
        true_labels = true_labels.astype(int)

    is_correct = predicted_classes == true_labels
    accuracy = float(np.mean(is_correct))
    f1 = float(f1_score(true_labels, predicted_classes, average="weighted", zero_division=0)) if not binary else float(f1_score(true_labels, predicted_classes, average="binary", zero_division=0))

    labels_for_confusion = list(range(output_dim)) if not binary else [0, 1]
    cm = confusion_matrix(true_labels, predicted_classes, labels=labels_for_confusion)
    report = classification_report(
        true_labels,
        predicted_classes,
        labels=labels_for_confusion,
        target_names=[str(i) for i in labels_for_confusion],
        output_dict=True,
        zero_division=0,
    )

    logger.info(f"Evaluation results: Accuracy {accuracy}, f1 {f1}")

    return {
        "true_labels": true_labels,
        "predicted_labels": predicted_classes,
        "probabilities": probabilities,
        "raw_logits": logits,
        "is_correct": is_correct,
        "accuracy": accuracy,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def normalize_explanation(explanation):
    explanation = np.asarray(explanation)
    if explanation.ndim == 0:
        return explanation.reshape(1, 1)
    if explanation.ndim > 2 and explanation.shape[-1] == 1:
        explanation = explanation[..., 0]
    if explanation.ndim > 2:
        explanation = np.mean(explanation, axis=-1)
    return explanation

def save_payload(payload, out_path):
    logger.info(f"Saving payload to {out_path}")
    with open(out_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)

def compute_chunked_explanations(explainer, X, y, chunk_size=32):
    if chunk_size is None or chunk_size <= 0:
        chunk_size = len(X)

    explanations = []
    for start in range(0, len(X), chunk_size):
        X_chunk = X[start:start+chunk_size]
        y_chunk = y[start:start+chunk_size]
        explanation = explainer(X_chunk, y_chunk)
        explanation = np.asarray(explanation)

        if explanation.ndim > 2 and explanation.shape[-1] == 1:
            explanation = explanation[..., 0]
        if explanation.ndim > 2:
            explanation = np.mean(explanation, axis=-1)

        explanations.append(explanation.astype(np.float32))

    if not explanations:
        return np.empty((0, X.shape[1]), dtype=np.float32)
    return np.concatenate(explanations, axis=0)


def make_explainers(wrapped_model, batch_size, lime_samples=256, shap_samples=256, chunk_size=32):
    return [
        Saliency(wrapped_model),
        GradientInput(wrapped_model),
        IntegratedGradients(wrapped_model, steps=20, batch_size=batch_size),
        SmoothGrad(wrapped_model, nb_samples=20, batch_size=batch_size),
        VarGrad(wrapped_model, nb_samples=20, batch_size=batch_size),
        SquareGrad(wrapped_model, nb_samples=20, batch_size=batch_size),
        # Occlusion(wrapped_model, patch_size=10, patch_stride=5, batch_size=batch_size),
        # GuidedBackprop(wrapped_model),
        Lime(wrapped_model, nb_samples=lime_samples, batch_size=batch_size)
        # KernelShap(wrapped_model, nb_samples=shap_samples, batch_size=batch_size),
    ]

def load_model_state(model_path, architecture, input_dim, output_dim, hidden_layers, device, batch_center=False):
    state = torch.load(model_path, map_location=device, weights_only=True)
    if '1.bias' in state.keys():
        width_multiplier = len(state['1.bias'])//64
        logger.info(f"{width_multiplier=}")
    else:
        logger.warning('No bias in first hidden layer, unexpected architecture')
        logger.warning(f"{state.keys()=}")
        width_multiplier = 1
    logger.warning(f"{architecture, input_dim, output_dim, hidden_layers, device, batch_center,width_multiplier}")
    model = build_torch_model(architecture, input_dim, output_dim, hidden_layers, device, batch_center=batch_center, width_multiplier=width_multiplier)

    # if isinstance(state, dict) and "state_dict" in state:
    #     state = state["state_dict"]

    # if isinstance(state, dict):
    #     cleaned = {}
    #     for key, value in state.items():
    #         cleaned[key.replace("module.", "")] = value
    #     state = cleaned

    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        print(e)
        logger.error(f"{state.keys()=}")
        sys.exit(1)

    model.eval()
    return model


def calculate_metric_scores(method_name, explanations, model, inputs, labels, binary, batch_size):
    explanation_array = normalize_explanation(explanations)

    metrics = {}

    complexity_metrics = {
        "Complexity": Complexity(),
        "Sparseness": Sparseness(),
    }
    for metric_name, metric_obj in complexity_metrics.items():
        scores = metric_obj.evaluate(explanation_array)
        metrics[metric_name] = float(np.mean(scores))

    deletion_metric = Deletion(model, inputs, labels, steps=20)
    insertion_metric = Insertion(model, inputs, labels, steps=20)
    mufidelity_metric = MuFidelity(model, inputs, labels, nb_samples=50)

    metrics["Deletion"] = float(np.mean(deletion_metric(explanation_array)))
    metrics["Insertion"] = float(np.mean(insertion_metric(explanation_array)))
    metrics["MuFidelity"] = float(np.mean(mufidelity_metric(explanation_array)))

    return metrics


def compute_attributions_for_model(model_path, data_path, architecture, hidden_layers, batch_size=128, max_samples=500, diffex=False, batch_center=False, output_dir=None, device=None, explanation_chunk_size=32, lime_samples=256, shap_samples=256, resume=False):
    if XPLIQUE_IMPORT_ERROR is not None:
        raise RuntimeError(f"xplique is required to compute attributions. Import failed: {XPLIQUE_IMPORT_ERROR}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = prep_data_get_dimensions_numpy(data_path, diffex=diffex)
    if "test_features" not in data or data["test_features"] is None:
        raise ValueError(f"No test features found in the dataset loaded from {data_path}")

    X_test = np.asarray(data["test_features"], dtype=np.float32)
    y_test = np.asarray(data["test_labels"], dtype=np.float32)
    binary = bool(data.get("binary", False))
    output_dim = int(data.get("output_dim", 1 if binary else y_test.shape[1]))

    if max_samples is not None and max_samples > 0:
        n_samples = min(max_samples, X_test.shape[0])
        X_test = X_test[:n_samples]
        y_test = y_test[:n_samples]

    model = load_model_state(
        model_path=model_path,
        architecture=architecture,
        input_dim=X_test.shape[1],
        output_dim=output_dim,
        hidden_layers=hidden_layers,
        device=device,
        batch_center=batch_center,
    )

    wrapped_model = TorchWrapper(model, device)
    labels = prepare_labels(y_test, output_dim, binary)
    inputs = X_test.copy()


    model_name = Path(model_path).stem
    output_dir = Path(output_dir) if output_dir else Path(model_path).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_name}_attributions_metrics.pkl"

    if resume:
        with open(out_path, "rb") as fh:
            payload = pickle.load(payload, fh)
        if 'attributions' in payload:
            explanations = payload['attributions']
        else:
            explanations = {}
        if 'metrics' in payload:
            metrics_by_method = payload['metrics']
        else:
            metrics_by_method = {}
        if 'evaluation' in payload:
            eval_summary = metrics['evaluation']
        else:
            eval_summary = compute_test_evaluation(model, inputs, y_test, binary, output_dim, device)
    else:
        eval_summary = compute_test_evaluation(model, inputs, y_test, binary, output_dim, device)
        payload = {
            "model_name": model_name,
            "model_path": str(model_path),
            "data_path": str(data_path),
            "binary": binary,
            "output_dim": output_dim,
            "batch_size": batch_size,
            "max_samples": max_samples,
            "evaluation": eval_summary
        }
        metrics_by_method = {}
        explanations = {}

    for explainer in make_explainers(
        wrapped_model,
        batch_size=batch_size,
        lime_samples=lime_samples,
        shap_samples=shap_samples,
        chunk_size=explanation_chunk_size,
    ):
        explainer_name = explainer.__class__.__name__
        if not explainer_name in explanations:
            logger.info(f"Running explainer {explainer_name} for {data['name']}")
            if explainer_name in {"Lime", "KernelShap"}:
                expl = compute_chunked_explanations(explainer, inputs, labels, chunk_size=explanation_chunk_size)
            else:
                expl = explainer(inputs, labels)
                expl = np.asarray(expl)
                if expl.ndim > 2 and expl.shape[-1] == 1:
                    expl = expl[..., 0]
                if expl.ndim > 2:
                    expl = np.mean(expl, axis=-1)
            explanations[explainer_name] = np.asarray(expl).astype(np.float32)
            payload['attributions'] = explanations
            save_payload(payload, out_path)

        for method_name, explanation in explanations.items():
            if method_name not in metrics_by_method:
                logger.info(f'Calculating {method_name}')
                metrics_by_method[method_name] = calculate_metric_scores(
                    method_name,
                    explanation,
                    wrapped_model,
                    inputs,
                    labels,
                    binary=binary,
                    batch_size=batch_size,
                )
                payload['metrics'] = metrics_by_method
                save_payload(payload, out_path)

    return payload


def main():
    parser = argparse.ArgumentParser(description="Load a trained model, compute xplique attributions and fidelity metrics, and save a pkl file.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved PyTorch model weights (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to the train features npy file, e.g. .../dataset_train_features.npy")
    parser.add_argument("--architecture", choices=["baseline", "deel-torchlib"], default="baseline", help="Neural network architecture used in training")
    parser.add_argument("--hidden-layers", type=str, default="64,32,16", help="Comma-separated hidden layer sizes used by the model")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for the attribution and metric calculations")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples used in attribution and metric evaluation")
    parser.add_argument("--explanation-chunk-size", type=int, default=32, help="Chunk size used when calculating Lime and KernelShap explanations")
    parser.add_argument("--lime-samples", type=int, default=256, help="Number of LIME samples per chunk")
    parser.add_argument("--shap-samples", type=int, default=256, help="Number of KernelSHAP samples per chunk")
    parser.add_argument("--diffex", action="store_true", help="Whether the dataset was filtered by differential expression")
    parser.add_argument("--resume", action="store_true", help="Whether to resume an existing calculation")
    parser.add_argument("--batch-center", action="store_true", help="Use BatchCentering blocks in the reconstructed model")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for the output pkl file; defaults to the model directory")
    parser.add_argument("--device", type=str, default=None, help="Optional device, e.g. cuda or cpu")
    args = parser.parse_args()

    hidden_layers = parse_hidden_layers(args.hidden_layers)
    if not hidden_layers:
        hidden_layers = [64, 32, 16]

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_attributions_for_model(
        model_path=args.model_path,
        data_path=args.data,
        architecture=args.architecture,
        hidden_layers=hidden_layers,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        diffex=args.diffex,
        batch_center=args.batch_center,
        output_dir=args.output_dir,
        device=device,
        explanation_chunk_size=args.explanation_chunk_size,
        lime_samples=args.lime_samples,
        shap_samples=args.shap_samples,
        resume = args.resume
    )


if __name__ == "__main__":
    main()
