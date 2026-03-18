"""
SimCLR Contrastive Learning - PyTorch Implementation

SimCLR (Simple Contrastive Learning of Representations) learns visual
representations WITHOUT labels. The core idea:

1. For each image, create two augmented "views" via random transformations.
2. Train an encoder so that embeddings of views from the SAME image (positive
   pairs) are pulled together, while embeddings of views from DIFFERENT images
   (negative pairs) are pushed apart.
3. The contrastive objective is NT-Xent (Normalized Temperature-scaled Cross
   Entropy Loss), which treats the positive pair as the correct class among
   all 2(N-1) negatives in a batch of size N.
4. After pretraining, freeze the encoder and train a linear classifier on top
   (linear probe) to evaluate representation quality. Good representations
   should enable a simple linear layer to achieve high accuracy.

Dataset: FashionMNIST (28x28 grayscale, 10 classes).
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'simclr_lvl3_contrastive',
        'task_type': 'self_supervised_classification',
        'description': (
            'SimCLR contrastive learning on FashionMNIST. Learns visual '
            'representations without labels using NT-Xent loss, then evaluates '
            'with a linear probe.'
        ),
        'input_type': 'float32',
        'output_type': 'int64'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Data augmentation & dataset
# ---------------------------------------------------------------------------

contrastive_transform = transforms.Compose([
    transforms.RandomResizedCrop(28, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

standard_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])


class ContrastiveDataset(Dataset):
    """Wraps a dataset to return two augmented views per image."""

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        # img is a PIL Image (FashionMNIST default when transform=None)
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2, label


def make_dataloaders(batch_size=256, contrastive=True):
    """
    Create dataloaders for FashionMNIST.

    Args:
        batch_size: Batch size.
        contrastive: If True, return ContrastiveDataset loaders (two views).
                     If False, return standard labeled loaders for linear probe.

    Returns:
        train_loader, val_loader
    """
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    if contrastive:
        train_set = torchvision.datasets.FashionMNIST(
            root=data_root, train=True, download=True, transform=None
        )
        train_dataset = ContrastiveDataset(train_set, contrastive_transform)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, drop_last=True
        )
        return train_loader, None
    else:
        train_set = torchvision.datasets.FashionMNIST(
            root=data_root, train=True, download=True,
            transform=standard_transform
        )
        val_set = torchvision.datasets.FashionMNIST(
            root=data_root, train=False, download=True,
            transform=standard_transform
        )
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False, num_workers=0
        )
        return train_loader, val_loader


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SimCLREncoder(nn.Module):
    """
    CNN encoder with a projection head for contrastive learning.

    Backbone produces 64-dim features. The projection head maps to 64-dim
    embeddings used only for the NT-Xent loss. For downstream tasks the
    backbone features (before the projection head) are used.
    """

    def __init__(self):
        super(SimCLREncoder, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.feature_dim = 64
        self.projection_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x):
        """Return projection head output (for contrastive loss)."""
        features = self.backbone(x)
        projections = self.projection_head(features)
        return projections

    def extract_features(self, x):
        """Return backbone features (for linear probe)."""
        return self.backbone(x)


class LinearProbe(nn.Module):
    """Single linear layer trained on frozen encoder features."""

    def __init__(self, feature_dim=64, num_classes=10):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def build_model(model_type, device, feature_dim=64):
    """
    Build a model.

    Args:
        model_type: 'encoder' or 'linear_probe'.
        device: torch device.
        feature_dim: Feature dimension for linear probe.

    Returns:
        model on device.
    """
    if model_type == 'encoder':
        model = SimCLREncoder()
    elif model_type == 'linear_probe':
        model = LinearProbe(feature_dim=feature_dim, num_classes=10)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model.to(device)


# ---------------------------------------------------------------------------
# NT-Xent loss
# ---------------------------------------------------------------------------

def nt_xent_loss(z1, z2, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy Loss.

    Args:
        z1, z2: Embeddings from two views, shape (batch_size, embed_dim).
        temperature: Scaling temperature.

    Returns:
        Scalar loss.
    """
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2N, embed_dim)
    z = F.normalize(z, dim=1)

    # Cosine similarity matrix: (2N, 2N)
    sim_matrix = torch.mm(z, z.t()) / temperature

    # Positive pair labels: i <-> i+N
    labels = torch.cat([
        torch.arange(batch_size) + batch_size,
        torch.arange(batch_size)
    ]).to(z.device)

    # Mask out self-similarity (diagonal)
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim_matrix.masked_fill_(mask, -1e9)

    loss = F.cross_entropy(sim_matrix, labels)
    return loss


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader, device, epochs,
          mode='contrastive', encoder=None):
    """
    Train a model.

    Args:
        model: The model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader (can be None).
        device: Torch device.
        epochs: Number of training epochs.
        mode: 'contrastive' for SimCLR pretraining,
              'linear_probe' for frozen-encoder linear evaluation.
        encoder: (linear_probe mode only) frozen encoder for feature extraction.

    Returns:
        history: dict with training metrics per epoch.
    """
    history = {'train_loss': []}

    if mode == 'contrastive':
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        print(f"  {'Epoch':>5}  {'Contr. Loss':>11}")
        print(f"  {'─'*5}  {'─'*11}")

        for epoch in range(epochs):
            running_loss = 0.0
            n_batches = 0
            for view1, view2, _ in train_loader:
                view1, view2 = view1.to(device), view2.to(device)
                z1 = model(view1)
                z2 = model(view2)
                loss = nt_xent_loss(z1, z2, temperature=0.5)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                n_batches += 1

            epoch_loss = running_loss / n_batches
            history['train_loss'].append(epoch_loss)
            print(
                f"  {epoch+1:>2}/{epochs:<2}  "
                f"{epoch_loss:>11.4f}"
            )

    elif mode == 'linear_probe':
        assert encoder is not None, "encoder required for linear_probe mode"
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        encoder.eval()

        history['val_acc'] = []

        print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Acc':>7}")
        print(f"  {'─'*5}  {'─'*10}  {'─'*7}")

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            n_batches = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.no_grad():
                    features = encoder.extract_features(images)
                logits = model(features)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                n_batches += 1

            epoch_loss = running_loss / n_batches
            history['train_loss'].append(epoch_loss)

            # Validation accuracy
            if val_loader is not None:
                metrics = evaluate(model, val_loader, device, encoder=encoder,
                                   verbose=False)
                history['val_acc'].append(metrics['accuracy'])
                print(
                    f"  {epoch+1:>2}/{epochs:<2}  "
                    f"{epoch_loss:>10.4f}  "
                    f"{metrics['accuracy']:>7.4f}"
                )
            else:
                print(
                    f"  {epoch+1:>2}/{epochs:<2}  "
                    f"{epoch_loss:>10.4f}"
                )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return history


def evaluate(model, data_loader, device, encoder=None, verbose=True):
    """
    Evaluate a model.

    Args:
        model: The model (linear probe or encoder).
        data_loader: Data loader.
        device: Torch device.
        encoder: If provided, extract features with this frozen encoder first.
        verbose: Print metrics.

    Returns:
        dict with 'accuracy' and 'loss'.
    """
    model.eval()
    if encoder is not None:
        encoder.eval()

    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            if encoder is not None:
                features = encoder.extract_features(images)
                logits = model(features)
            else:
                logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / n_batches

    if verbose:
        print(f"  Accuracy: {accuracy:.4f}  Loss: {avg_loss:.4f}")

    return {'accuracy': accuracy, 'loss': avg_loss}


def predict(model, data_loader, device, encoder=None):
    """
    Return predictions tensor.

    Args:
        model: The model.
        data_loader: Data loader.
        device: Torch device.
        encoder: If provided, extract features first.

    Returns:
        Tensor of predicted class indices.
    """
    model.eval()
    if encoder is not None:
        encoder.eval()

    all_preds = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            if encoder is not None:
                features = encoder.extract_features(images)
                logits = model(features)
            else:
                logits = model(images)
            all_preds.append(logits.argmax(dim=1).cpu())

    return torch.cat(all_preds)


def save_artifacts(model, metrics, output_dir='output',
                   pretrain_history=None, pretrained_metrics=None, random_metrics=None):
    """
    Save model state dict, metrics JSON, and plots.

    Args:
        model: The encoder model.
        metrics: Dict of metrics.
        output_dir: Directory to save into.
        pretrain_history: dict with 'train_loss' from contrastive pretraining
        pretrained_metrics: dict with 'accuracy' from pretrained probe
        random_metrics: dict with 'accuracy' from random probe
    """
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(output_dir, 'encoder_state.pt'))
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Artifacts saved to {output_dir}")

    # --- Plots ---

    # 1. Contrastive loss curve
    if pretrain_history is not None:
        losses = pretrain_history['train_loss']
        epochs = range(1, len(losses) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, losses, marker='o', color='#2196F3')
        plt.xlabel('Epoch')
        plt.ylabel('Contrastive Loss (NT-Xent)')
        plt.title('SimCLR Contrastive Pretraining Loss')
        plt.grid(True, alpha=0.3)
        path = os.path.join(output_dir, 'contrastive_loss.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {path}")

    # 2. Probe comparison bar chart
    if pretrained_metrics is not None and random_metrics is not None:
        plt.figure(figsize=(7, 5))
        names = ['Pretrained Probe', 'Random Probe']
        accs = [pretrained_metrics['accuracy'], random_metrics['accuracy']]
        colors = ['#4CAF50', '#FF9800']
        bars = plt.bar(names, accs, color=colors)
        for bar, acc in zip(bars, accs):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{acc:.4f}', ha='center', va='bottom', fontsize=11)
        plt.ylabel('Validation Accuracy')
        plt.title('Linear Probe Accuracy Comparison')
        plt.ylim(0, 1.0)
        plt.grid(axis='y', alpha=0.3)
        path = os.path.join(output_dir, 'probe_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {path}")

    # 3. Training summary (combined figure)
    if pretrain_history is not None and pretrained_metrics is not None and random_metrics is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: loss curve
        losses = pretrain_history['train_loss']
        epochs = range(1, len(losses) + 1)
        ax1.plot(epochs, losses, marker='o', color='#2196F3')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Contrastive Loss')
        ax1.set_title('Pretraining Loss')
        ax1.grid(True, alpha=0.3)

        # Right: accuracy comparison
        names = ['Pretrained', 'Random']
        accs = [pretrained_metrics['accuracy'], random_metrics['accuracy']]
        colors = ['#4CAF50', '#FF9800']
        bars = ax2.bar(names, accs, color=colors)
        for bar, acc in zip(bars, accs):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{acc:.4f}', ha='center', va='bottom', fontsize=11)
        ax2.set_ylabel('Validation Accuracy')
        ax2.set_title('Linear Probe Accuracy')
        ax2.set_ylim(0, 1.0)
        ax2.grid(axis='y', alpha=0.3)

        fig.suptitle('SimCLR Training Summary', fontsize=14)
        fig.tight_layout()
        path = os.path.join(output_dir, 'training_summary.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the full SimCLR contrastive learning pipeline."""
    print("=" * 60)
    print("SimCLR Contrastive Learning - FashionMNIST")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Phase 1: Contrastive pretraining (NO labels)
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Phase 1: SimCLR Contrastive Pretraining")
    print("-" * 60)
    contrastive_loader, _ = make_dataloaders(batch_size=256, contrastive=True)
    encoder = build_model('encoder', device)
    pretrain_history = train(encoder, contrastive_loader, None, device,
                            epochs=20, mode='contrastive')
    encoder.eval()

    # ------------------------------------------------------------------
    # Phase 2: Linear probe on pretrained encoder
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Phase 2: Linear Probe on Pretrained Encoder")
    print("-" * 60)
    train_loader, val_loader = make_dataloaders(batch_size=256,
                                                contrastive=False)
    probe_pretrained = build_model('linear_probe', device)
    train(probe_pretrained, train_loader, val_loader, device, epochs=10,
          mode='linear_probe', encoder=encoder)
    pretrained_metrics = evaluate(probe_pretrained, val_loader, device,
                                 encoder=encoder)

    # ------------------------------------------------------------------
    # Phase 3: Linear probe on RANDOM encoder (baseline)
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Phase 3: Linear Probe on Random Encoder (baseline)")
    print("-" * 60)
    set_seed(99)
    random_encoder = build_model('encoder', device)
    random_encoder.eval()
    probe_random = build_model('linear_probe', device)
    train(probe_random, train_loader, val_loader, device, epochs=10,
          mode='linear_probe', encoder=random_encoder)
    random_metrics = evaluate(probe_random, val_loader, device,
                              encoder=random_encoder)

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    save_artifacts(encoder, {
        'pretrained_probe': pretrained_metrics,
        'random_probe': random_metrics,
        'pretrain_loss_history': pretrain_history['train_loss']
    }, pretrain_history=pretrain_history,
        pretrained_metrics=pretrained_metrics,
        random_metrics=random_metrics)

    # ------------------------------------------------------------------
    # Quality checks
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  Pretrained probe accuracy: {pretrained_metrics['accuracy']:.4f}")
    print(f"  Random probe accuracy:     {random_metrics['accuracy']:.4f}")
    print(f"  Improvement:               "
          f"{pretrained_metrics['accuracy'] - random_metrics['accuracy']:.4f}")

    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    checks_passed = True

    # Check 1: Pretrained > random + 0.05
    gap = pretrained_metrics['accuracy'] - random_metrics['accuracy']
    c1 = gap > 0.05
    tag = "PASS" if c1 else "FAIL"
    print(f"  [{tag}] Pretrained acc > Random acc + 0.05  "
          f"(gap={gap:.4f})")
    checks_passed = checks_passed and c1

    # Check 2: Pretrained probe accuracy > 0.65
    c2 = pretrained_metrics['accuracy'] > 0.65
    tag = "PASS" if c2 else "FAIL"
    print(f"  [{tag}] Pretrained acc > 0.65  "
          f"({pretrained_metrics['accuracy']:.4f})")
    checks_passed = checks_passed and c2

    # Check 3: Contrastive loss decreased
    losses = pretrain_history['train_loss']
    c3 = losses[-1] < losses[0]
    tag = "PASS" if c3 else "FAIL"
    print(f"  [{tag}] Contrastive loss decreased  "
          f"({losses[0]:.4f} -> {losses[-1]:.4f})")
    checks_passed = checks_passed and c3

    print("\n" + "=" * 60)
    if checks_passed:
        print("PASS: All quality checks passed!")
    else:
        print("FAIL: Some quality checks failed!")
    print("=" * 60)

    return 0 if checks_passed else 1


if __name__ == '__main__':
    sys.exit(main())
