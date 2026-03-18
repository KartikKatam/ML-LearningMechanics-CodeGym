"""
Vision Transformer (ViT) Classification on FashionMNIST - PyTorch Implementation

Implements a Vision Transformer from scratch for image classification.

Key concepts:
- Patch Embedding: Each 28x28 image is split into a 4x4 grid of 7x7 patches (16 patches).
  Each patch (49 pixels) is linearly projected to an embedding dimension of 64.
- CLS Token: A learnable classification token is prepended to the patch sequence,
  giving a total sequence length of 17 (16 patches + 1 CLS token).
- Positional Embeddings: Learnable embeddings of shape (1, 17, 64) are added to encode
  spatial information lost during patch extraction.
- Transformer Encoder: 2 layers of multi-head self-attention (4 heads) with feedforward
  dimension 128 process the token sequence.
- Classification Head: The CLS token output (position 0) is passed through LayerNorm
  and a linear layer to produce 10 class logits.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'vit_lvl3_fashionmnist',
        'task_type': 'classification',
        'description': 'Vision Transformer classifier on FashionMNIST (28x28, 10 classes)',
        'model_type': 'vision_transformer',
        'dataset': 'FashionMNIST',
        'input_type': 'float32',
        'output_type': 'int64',
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_device():
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def make_dataloaders(batch_size=128):
    """
    Create FashionMNIST train and validation dataloaders.

    Uses the official train split (60k) and test split (10k) as train/val.

    Returns:
        train_loader, val_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform,
    )
    val_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader


class VisionTransformer(nn.Module):
    """
    Vision Transformer for image classification.

    Architecture:
        - Patch embedding (7x7 patches from 28x28 images -> 16 patches -> linear to embed_dim)
        - Prepend learnable CLS token
        - Add learnable positional embeddings
        - Transformer encoder (2 layers, 4 heads)
        - LayerNorm + linear classification head on CLS token output
    """

    def __init__(
        self,
        img_size=28,
        patch_size=7,
        in_channels=1,
        num_classes=10,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
    ):
        super().__init__()

        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2  # 16
        patch_dim = patch_size * patch_size * in_channels  # 49

        # Patch embedding: flatten each patch and project
        self.patch_embed = nn.Linear(patch_dim, embed_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Positional embeddings: (1, num_patches + 1, embed_dim) = (1, 17, 64)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: (batch, 1, 28, 28) images

        Returns:
            logits: (batch, 10)
        """
        B = x.shape[0]

        # Extract patches: (B, 1, 28, 28) -> (B, num_patches, patch_dim)
        p = self.patch_size
        # Unfold height then width to get a grid of patches
        patches = x.unfold(2, p, p).unfold(3, p, p)  # (B, 1, 4, 4, 7, 7)
        patches = patches.contiguous().view(B, -1, p * p)  # (B, 16, 49)

        # Linear projection
        tokens = self.patch_embed(patches)  # (B, 16, 64)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 64)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (B, 17, 64)

        # Add positional embeddings
        tokens = tokens + self.pos_embed  # (B, 17, 64)

        # Transformer encoder
        tokens = self.transformer(tokens)  # (B, 17, 64)

        # Classification: take CLS token output
        cls_output = tokens[:, 0]  # (B, 64)
        cls_output = self.ln(cls_output)
        logits = self.head(cls_output)  # (B, 10)

        return logits


def build_model(device):
    """
    Build the Vision Transformer model and move it to the given device.

    Returns:
        model: VisionTransformer instance on device
    """
    model = VisionTransformer(
        img_size=28,
        patch_size=7,
        in_channels=1,
        num_classes=10,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
    )
    model = model.to(device)
    return model


def train(model, train_loader, val_loader, device, epochs=8):
    """
    Train the ViT model.

    Args:
        model: VisionTransformer instance
        train_loader: training DataLoader
        val_loader: validation DataLoader
        device: torch device
        epochs: number of training epochs

    Returns:
        dict with loss_history, val_loss_history, val_acc_history
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    loss_history = []
    val_loss_history = []
    val_acc_history = []

    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>8}  {'Val Acc':>7}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*8}  {'─'*7}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_train_loss = running_loss / num_batches
        loss_history.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total

        val_loss_history.append(avg_val_loss)
        val_acc_history.append(val_acc)

        print(
            f"  {epoch + 1:>2}/{epochs:<2}  "
            f"{avg_train_loss:>10.4f}  "
            f"{avg_val_loss:>8.4f}  "
            f"{val_acc:>7.4f}"
        )

    return {
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
        'val_acc_history': val_acc_history,
    }


def evaluate(model, data_loader, device):
    """
    Evaluate the model on a dataset.

    Args:
        model: VisionTransformer instance
        data_loader: DataLoader to evaluate on
        device: torch device

    Returns:
        dict with 'loss', 'accuracy', 'per_class_accuracy' (list of 10 floats)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for i in range(10):
                mask = labels == i
                class_total[i] += mask.sum().item()
                class_correct[i] += (preds[mask] == labels[mask]).sum().item()

    accuracy = correct / total
    avg_loss = total_loss / len(data_loader)
    per_class_accuracy = [
        class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        for i in range(10)
    ]

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'per_class_accuracy': per_class_accuracy,
    }


def predict(model, data_loader, device):
    """
    Return all predictions as a tensor.

    Args:
        model: VisionTransformer instance
        data_loader: DataLoader
        device: torch device

    Returns:
        predictions: tensor of predicted class indices
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())

    return torch.cat(all_preds)


def save_artifacts(model, metrics, output_dir='output', history=None, val_metrics=None):
    """
    Save model state_dict, metrics JSON, and plots.

    Args:
        model: VisionTransformer instance
        metrics: dict of metrics
        output_dir: directory to save into
        history: dict with loss_history, val_loss_history, val_acc_history
        val_metrics: dict with per_class_accuracy
    """
    os.makedirs(output_dir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(output_dir, 'model_state.pt'))

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Artifacts saved to {output_dir}")

    # --- Plots ---
    if history is not None:
        epochs = range(1, len(history['loss_history']) + 1)

        # 1. Loss curves
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history['loss_history'], label='Train Loss', marker='o')
        plt.plot(epochs, history['val_loss_history'], label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        path = os.path.join(output_dir, 'loss_curves.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {path}")

        # 2. Accuracy curve
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, history['val_acc_history'], label='Val Accuracy', marker='o', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        path = os.path.join(output_dir, 'accuracy_curve.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {path}")

    if val_metrics is not None and 'per_class_accuracy' in val_metrics:
        # 3. Confusion-style heatmap of per-class accuracy
        class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot',
        ]
        per_class = val_metrics['per_class_accuracy']
        matrix = np.diag(per_class)
        plt.figure(figsize=(9, 8))
        plt.imshow(matrix, cmap='Blues', vmin=0, vmax=1)
        plt.colorbar(label='Accuracy')
        plt.xticks(range(10), class_names, rotation=45, ha='right')
        plt.yticks(range(10), class_names)
        for i in range(10):
            plt.text(i, i, f'{per_class[i]:.2f}', ha='center', va='center',
                     fontsize=9, color='white' if per_class[i] > 0.5 else 'black')
        plt.title('Per-Class Validation Accuracy')
        plt.xlabel('Class')
        plt.ylabel('Class')
        path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {path}")


def main():
    """Main function to run the ViT classification task."""
    print("=" * 60)
    print("Vision Transformer (ViT) - FashionMNIST Classification")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # Dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = make_dataloaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Build model
    print("\nBuilding model...")
    model = build_model(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Train
    print("\nTraining...")
    history = train(model, train_loader, val_loader, device, epochs=8)

    # Evaluate
    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device)
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}  |  Loss: {train_metrics['loss']:.4f}")

    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}  |  Loss: {val_metrics['loss']:.4f}")

    # Per-class accuracy
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot',
    ]
    print("\nPer-class validation accuracy:")
    for i, name in enumerate(class_names):
        print(f"  {name:15s}: {val_metrics['per_class_accuracy'][i]:.4f}")

    # Save artifacts
    print("\nSaving artifacts...")
    save_artifacts(model, {'train': train_metrics, 'val': val_metrics}, output_dir='output',
                   history=history, val_metrics=val_metrics)

    # Quality checks
    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    checks_passed = True

    check1 = val_metrics['accuracy'] > 0.85
    print(f"{'✓' if check1 else '✗'} Val Accuracy > 0.85: {val_metrics['accuracy']:.4f}")
    checks_passed = checks_passed and check1

    check2 = history['loss_history'][-1] < history['loss_history'][0]
    print(f"{'✓' if check2 else '✗'} Train loss decreased: {history['loss_history'][0]:.4f} -> {history['loss_history'][-1]:.4f}")
    checks_passed = checks_passed and check2

    check3 = val_metrics['accuracy'] > 0.10
    print(f"{'✓' if check3 else '✗'} Val Accuracy > 0.10 (above random): {val_metrics['accuracy']:.4f}")
    checks_passed = checks_passed and check3

    # Final summary
    print("\n" + "=" * 60)
    if checks_passed:
        print("PASS: All quality checks passed!")
        print("=" * 60)
        return 0
    else:
        print("FAIL: Some quality checks failed!")
        print("=" * 60)
        return 1


if __name__ == '__main__':
    sys.exit(main())
