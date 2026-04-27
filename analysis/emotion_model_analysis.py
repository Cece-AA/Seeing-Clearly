import argparse
import os
from collections import defaultdict
from pathlib import Path
import sys

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torchvision.datasets import ImageFolder

sys.path.append(str(Path(__file__).resolve().parents[1]))

from seeing_clearly.core import CLASS_NAMES, build_dataset_transform, build_model, unnormalize_image


def collect_predictions(model, dataset, device, max_samples=None):
    y_true = []
    y_pred = []
    per_class_examples = defaultdict(list)

    total_samples = len(dataset) if max_samples is None else min(len(dataset), max_samples)

    for idx in range(total_samples):
        image, label = dataset[idx]
        batch = image.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            pred = int(torch.argmax(probs, dim=1).item())
            confidence = float(torch.max(probs).item())

        y_true.append(label)
        y_pred.append(pred)
        per_class_examples[label].append(
            {
                "dataset_index": idx,
                "pred": pred,
                "confidence": confidence,
                "correct": pred == label
            }
        )

    return np.array(y_true), np.array(y_pred), per_class_examples


def per_class_accuracy(y_true, y_pred, class_names):
    metrics = []
    for class_idx, class_name in enumerate(class_names):
        mask = y_true == class_idx
        total = int(mask.sum())
        correct = int((y_pred[mask] == class_idx).sum()) if total else 0
        accuracy = correct / total if total else 0.0
        metrics.append(
            {
                "class_name": class_name,
                "correct": correct,
                "total": total,
                "accuracy": accuracy
            }
        )
    return metrics


def save_confusion_matrix(y_true, y_pred, class_names, output_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, matrix, title, fmt in [
        (axes[0], cm, "Confusion Matrix (Counts)", "d"),
        (axes[1], cm_normalized, "Confusion Matrix (Row-Normalized)", ".2f")
    ]:
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title(title)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                value = format(matrix[i, j], fmt)
                ax.text(j, i, value, ha="center", va="center", color="black", fontsize=9)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_saliency(model, image_tensor, target_class, device):
    image = image_tensor.unsqueeze(0).to(device)
    image.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(image)
    score = logits[0, target_class]
    score.backward()

    saliency = image.grad.abs().max(dim=1)[0].squeeze().detach().cpu().numpy()
    saliency -= saliency.min()
    saliency /= saliency.max() + 1e-8
    return saliency


def save_saliency_examples(model, dataset, class_names, per_class_examples, device, output_dir, samples_per_class):
    output_dir.mkdir(parents=True, exist_ok=True)

    for class_idx, class_name in enumerate(class_names):
        ranked_examples = sorted(
            per_class_examples[class_idx],
            key=lambda item: (not item["correct"], -item["confidence"])
        )

        selected = ranked_examples[:samples_per_class]
        if not selected:
            continue

        fig, axes = plt.subplots(len(selected), 2, figsize=(10, 4 * len(selected)))
        if len(selected) == 1:
            axes = np.array([axes])

        for row_idx, example in enumerate(selected):
            image_tensor, label = dataset[example["dataset_index"]]
            saliency = compute_saliency(model, image_tensor, example["pred"], device)
            image = unnormalize_image(image_tensor)

            axes[row_idx, 0].imshow(image)
            axes[row_idx, 0].set_title(
                f"True: {class_names[label]} | Pred: {class_names[example['pred']]}\n"
                f"Confidence: {example['confidence']:.2f}"
            )
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(image)
            axes[row_idx, 1].imshow(saliency, cmap="inferno", alpha=0.55)
            axes[row_idx, 1].set_title("Saliency Overlay")
            axes[row_idx, 1].axis("off")

        fig.tight_layout()
        fig.savefig(output_dir / f"{class_name}_saliency.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def write_summary(overall_accuracy, class_metrics, class_names, y_true, y_pred, output_path):
    sorted_metrics = sorted(class_metrics, key=lambda item: item["accuracy"])
    weakest = sorted_metrics[:2]
    strongest = sorted_metrics[-2:]

    confusion_notes = []
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    for class_idx, class_name in enumerate(class_names):
        row = cm[class_idx].copy()
        row[class_idx] = 0
        if row.sum() == 0:
            continue
        confused_with = int(np.argmax(row))
        confused_count = int(row[confused_with])
        if confused_count > 0:
            confusion_notes.append(
                f"- `{class_name}` is most often confused with `{class_names[confused_with]}` ({confused_count} cases)."
            )

    summary_lines = [
        "# Model Analysis Summary",
        "",
        f"- Overall test accuracy: `{overall_accuracy:.2%}`",
        "",
        "## Per-Class Accuracy",
    ]

    for metric in class_metrics:
        summary_lines.append(
            f"- `{metric['class_name']}`: `{metric['accuracy']:.2%}` "
            f"({metric['correct']}/{metric['total']})"
        )

    summary_lines.extend([
        "",
        "## Assistive Interpretation Takeaways",
        f"- Strongest classes right now: `{strongest[0]['class_name']}` and `{strongest[1]['class_name']}`.",
        f"- Weakest classes right now: `{weakest[0]['class_name']}` and `{weakest[1]['class_name']}`.",
        "- Higher-performing classes are more reliable for quick assistive cues in the live demo.",
        "- Lower-performing classes should be treated as softer signals, especially when nearby emotions can look similar.",
        "- Review the confusion matrix and saliency overlays together to see not just what the model predicts, but where it is likely to mislead or over-focus.",
        "",
        "## Common Confusions",
    ])

    if confusion_notes:
        summary_lines.extend(confusion_notes)
    else:
        summary_lines.append("- No major off-diagonal confusions were found in this run.")

    output_path.write_text("\n".join(summary_lines) + "\n")


def resolve_dataset_path(dataset_path_arg):
    if dataset_path_arg:
        return Path(dataset_path_arg).expanduser().resolve()
    return Path(kagglehub.dataset_download("msambare/fer2013")).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze the FER model with class-level metrics and saliency maps.")
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Optional path to the FER2013 dataset root. Defaults to downloading via kagglehub."
    )
    parser.add_argument(
        "--model-path",
        default="models/fer_resnet18.pth",
        help="Path to the trained model weights."
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_outputs",
        help="Directory for confusion matrices, saliency maps, and summary files."
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=2,
        help="How many saliency examples to save per class."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit for quick analysis runs."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = resolve_dataset_path(args.dataset_path)
    test_dir = dataset_path / "test"
    output_dir = Path(args.output_dir)
    saliency_dir = output_dir / "saliency_maps"

    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found at {test_dir}")

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model weights not found at {args.model_path}")

    dataset = ImageFolder(test_dir, transform=build_dataset_transform())
    class_names = dataset.classes or CLASS_NAMES
    model = build_model(model_path=args.model_path, device=device)

    y_true, y_pred, per_class_examples = collect_predictions(
        model,
        dataset,
        device,
        max_samples=args.max_samples
    )

    overall_accuracy = float((y_true == y_pred).mean())
    class_metrics = per_class_accuracy(y_true, y_pred, class_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(y_true, y_pred, class_names, output_dir / "confusion_matrix.png")
    save_saliency_examples(
        model,
        dataset,
        class_names,
        per_class_examples,
        device,
        saliency_dir,
        args.samples_per_class
    )
    write_summary(
        overall_accuracy,
        class_metrics,
        class_names,
        y_true,
        y_pred,
        output_dir / "analysis_summary.md"
    )

    print(f"Overall accuracy: {overall_accuracy:.2%}")
    print("Per-class accuracy:")
    for metric in class_metrics:
        print(f"  {metric['class_name']}: {metric['accuracy']:.2%} ({metric['correct']}/{metric['total']})")
    print(f"Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")
    print(f"Saved saliency maps to {saliency_dir}")
    print(f"Saved summary to {output_dir / 'analysis_summary.md'}")


if __name__ == "__main__":
    main()
