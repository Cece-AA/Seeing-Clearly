import argparse
import copy
import json
import math
import random
from pathlib import Path
import sys

import kagglehub
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

sys.path.append(str(Path(__file__).resolve().parents[1]))

from seeing_clearly.core import MEAN, STD, create_model, get_input_size
try:
    from facenet_pytorch import MTCNN
except ImportError:  # pragma: no cover - optional dependency
    MTCNN = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train a stronger FER2013 emotion model.")
    parser.add_argument("--dataset-path", default=None, help="Optional FER2013 dataset root.")
    parser.add_argument("--output", default="models/fer_best_model.pth", help="Checkpoint output path.")
    parser.add_argument("--architecture", default="inceptionresnetv1", choices=["resnet18", "efficientnet_b0", "efficientnet_b2", "inceptionresnetv1"])
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mix-prob", type=float, default=0.7)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--align-faces", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dataset_root(dataset_path_arg):
    if dataset_path_arg:
        return Path(dataset_path_arg).expanduser().resolve()
    return Path(kagglehub.dataset_download("msambare/fer2013")).resolve()


def build_transforms(architecture):
    image_size = get_input_size(architecture)
    resize_size = max(int(image_size * 1.15), image_size)
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomResizedCrop(image_size, scale=(0.78, 1.0), ratio=(0.92, 1.08)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=6),
        transforms.ColorJitter(brightness=0.18, contrast=0.25),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.0), value="random"),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return train_tf, eval_tf


class FaceAligner:
    def __init__(self, image_size, device):
        if MTCNN is None:
            raise ImportError("facenet-pytorch is required for face alignment.")
        mtcnn_device = "cpu"
        if device.type == "cuda":
            mtcnn_device = "cuda"
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=14,
            post_process=False,
            device=mtcnn_device,
            keep_all=False,
            select_largest=False,
        )

    def align(self, image):
        face = self.mtcnn(image)
        if face is None:
            return image
        aligned = face.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        return Image.fromarray(aligned)


class AlignedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, aligner):
        self.subset = subset
        self.aligner = aligner

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        sample_index = self.subset.indices[index]
        path, label = self.subset.dataset.samples[sample_index]
        image = self.subset.dataset.loader(path)
        image = self.aligner.align(image)
        transformed = self.subset.dataset.transform(image)
        return transformed, label


def make_split_datasets(train_dir, val_size, seed, architecture, aligner=None):
    train_tf, eval_tf = build_transforms(architecture)
    base_dataset = ImageFolder(train_dir)
    targets = np.array(base_dataset.targets)
    indices = np.arange(len(base_dataset))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_size,
        random_state=seed,
        stratify=targets,
    )

    train_dataset = Subset(ImageFolder(train_dir, transform=train_tf), train_indices.tolist())
    val_dataset = Subset(ImageFolder(train_dir, transform=eval_tf), val_indices.tolist())
    if aligner is not None:
        train_dataset = AlignedSubset(train_dataset, aligner)
        val_dataset = AlignedSubset(val_dataset, aligner)
    return base_dataset.classes, train_dataset, val_dataset


def make_test_dataset(test_dir, architecture, aligner=None):
    _, eval_tf = build_transforms(architecture)
    dataset = ImageFolder(test_dir, transform=eval_tf)
    if aligner is not None:
        indices = list(range(len(dataset)))
        dataset = AlignedSubset(Subset(dataset, indices), aligner)
    return dataset


def make_weighted_sampler(train_dataset, class_names):
    subset_targets = []
    base_subset = train_dataset.subset if isinstance(train_dataset, AlignedSubset) else train_dataset
    dataset = base_subset.dataset
    for index in base_subset.indices:
        _, label = dataset.samples[index]
        subset_targets.append(label)

    class_counts = np.bincount(subset_targets, minlength=len(class_names))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = [class_weights[label] for label in subset_targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    return sampler, class_weights_tensor


def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).sum().item(), labels.size(0)


def sample_lambda(alpha):
    if alpha <= 0:
        return 1.0
    return float(np.random.beta(alpha, alpha))


def rand_bbox(size, lam):
    _, _, height, width = size
    cut_ratio = math.sqrt(1.0 - lam)
    cut_width = int(width * cut_ratio)
    cut_height = int(height * cut_ratio)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    x1 = np.clip(cx - cut_width // 2, 0, width)
    y1 = np.clip(cy - cut_height // 2, 0, height)
    x2 = np.clip(cx + cut_width // 2, 0, width)
    y2 = np.clip(cy + cut_height // 2, 0, height)
    return x1, y1, x2, y2


def apply_mixup_or_cutmix(images, labels, mixup_alpha, cutmix_alpha, mix_prob):
    if mix_prob <= 0 or torch.rand(1).item() > mix_prob:
        return images, labels, labels, 1.0, "none"

    batch_size = images.size(0)
    indices = torch.randperm(batch_size, device=images.device)
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]

    use_cutmix = cutmix_alpha > 0 and torch.rand(1).item() < 0.5
    if use_cutmix:
        lam = sample_lambda(cutmix_alpha)
        x1, y1, x2, y2 = rand_bbox(images.size(), lam)
        images = images.clone()
        images[:, :, y1:y2, x1:x2] = shuffled_images[:, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
        return images, labels, shuffled_labels, lam, "cutmix"

    lam = sample_lambda(mixup_alpha)
    mixed_images = lam * images + (1.0 - lam) * shuffled_images
    return mixed_images, labels, shuffled_labels, lam, "mixup"


def mixed_loss(criterion, logits, labels_a, labels_b, lam):
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)


def build_scheduler(optimizer, steps_per_epoch, epochs, warmup_epochs, min_lr):
    warmup_steps = max(warmup_epochs * steps_per_epoch, 1)
    total_steps = max(epochs * steps_per_epoch, warmup_steps + 1)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_factor = min_lr / optimizer.defaults["lr"]
        return max(min_factor, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            correct, count = accuracy_from_logits(logits, labels)
            total_correct += correct
            total_seen += count

    return total_loss / total_seen, total_correct / total_seen


def train(args):
    set_seed(args.seed)
    device = get_device()
    dataset_root = resolve_dataset_root(args.dataset_path)
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"
    aligner = None
    if args.align_faces:
        aligner = FaceAligner(get_input_size(args.architecture), device)

    class_names, train_dataset, val_dataset = make_split_datasets(
        train_dir,
        args.val_size,
        args.seed,
        args.architecture,
        aligner=aligner,
    )
    test_dataset = make_test_dataset(test_dir, args.architecture, aligner=aligner)
    sampler, class_weights = make_weighted_sampler(train_dataset, class_names)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    model = create_model(args.architecture, num_classes=len(class_names), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_learning_rate,
    )

    best_val_acc = 0.0
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_seen = 0
        mix_counts = {"none": 0, "mixup": 0, "cutmix": 0}

        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            mixed_images, labels_a, labels_b, lam, mix_mode = apply_mixup_or_cutmix(
                images,
                labels,
                args.mixup_alpha,
                args.cutmix_alpha,
                args.mix_prob,
            )
            mix_counts[mix_mode] += 1

            optimizer.zero_grad()
            logits = model(mixed_images)
            loss = mixed_loss(criterion, logits, labels_a, labels_b, lam)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * labels.size(0)
            correct, count = accuracy_from_logits(logits, labels)
            running_correct += correct
            running_seen += count

            if batch_idx % args.log_every == 0 or batch_idx == 1 or batch_idx == len(train_loader):
                print(
                    f"Epoch {epoch:02d} batch {batch_idx:03d}/{len(train_loader):03d} | "
                    f"loss={loss.item():.4f} | mode={mix_mode}",
                    flush=True,
                )

        train_loss = running_loss / running_seen
        train_acc = running_correct / running_seen
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
                "mixup_batches": mix_counts["mixup"],
                "cutmix_batches": mix_counts["cutmix"],
            }
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | "
            f"mixup={mix_counts['mixup']} cutmix={mix_counts['cutmix']}",
            flush=True,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Stopping early after {epoch} epochs.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint.")

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture": args.architecture,
        "class_names": class_names,
        "state_dict": best_state,
        "best_epoch": best_epoch,
        "val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "dataset": "FER2013",
        "align_faces": args.align_faces,
        "label_smoothing": args.label_smoothing,
        "mixup_alpha": args.mixup_alpha,
        "cutmix_alpha": args.cutmix_alpha,
        "history": history,
    }
    torch.save(checkpoint, output_path)
    metrics_path = output_path.with_suffix(".json")
    metrics_path.write_text(json.dumps(checkpoint | {"state_dict": "<omitted>"}, indent=2))

    print(f"Saved best checkpoint to {output_path}", flush=True)
    print(f"Saved metrics to {metrics_path}", flush=True)
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}", flush=True)
    print(f"Test accuracy: {test_acc:.4f}", flush=True)


if __name__ == "__main__":
    train(parse_args())
