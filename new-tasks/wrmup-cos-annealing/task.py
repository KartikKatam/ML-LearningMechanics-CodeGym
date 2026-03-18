"""
Cosine Annealing + Warmup LR Schedule - PyTorch Implementation

Demonstrates that learning rate scheduling improves neural network training.
Three identical SmallCNN models are trained on FashionMNIST with different LR
strategies to show the benefit of cosine annealing and linear warmup.

Cosine annealing formula:
    lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))

This smoothly decays the learning rate from lr_max to lr_min following a cosine
curve. The intuition is that large steps early in training help explore the loss
landscape, while smaller steps later help converge to a sharp minimum.

Linear warmup rationale:
    During the first N steps, linearly ramp LR from 0 to lr_max. This prevents
    early training instability when the model weights are still random and
    gradients can be noisy/large. After warmup, cosine annealing takes over.

Why LR scheduling helps:
    A fixed LR must compromise between being large enough to make progress early
    and small enough to converge precisely later. Scheduling removes this
    trade-off by adapting the LR over time. SGD (unlike Adam) has no per-parameter
    adaptive rates, so it benefits most from an external schedule.

Runs:
    A) Fixed LR       - constant 1e-2 throughout
    B) Cosine anneal  - starts at 1e-2, decays to ~0 via cosine curve
    C) Warmup+Cosine  - linear warmup (5% of steps) then cosine annealing
"""

import os
import sys
import json
import math
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
        'task_name': 'sched_lvl3_cosine_warmup',
        'task_type': 'classification',
        'description': (
            'Compare fixed LR, cosine annealing, and warmup+cosine annealing '
            'schedules when training a CNN on FashionMNIST with SGD'
        ),
        'input_type': 'float32',
        'output_type': 'int64'
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

    Returns:
        train_loader, val_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    val_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader


class SmallCNN(nn.Module):
    """Small CNN for FashionMNIST classification."""

    def __init__(self):
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# LR Schedulers (manual implementations)
# ---------------------------------------------------------------------------

class FixedScheduler:
    """Scheduler that keeps LR constant (no-op)."""

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        return self.optimizer.param_groups[0]['lr']

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class CosineScheduler:
    """Manual cosine annealing LR scheduler (no warmup)."""

    def __init__(self, optimizer, total_steps, lr_max, lr_min=1e-6):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.current_step = 0

    def step(self):
        self.current_step += 1
        progress = self.current_step / self.total_steps
        lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + math.cos(math.pi * progress)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


class WarmupCosineScheduler:
    """Manual warmup + cosine annealing LR scheduler."""

    def __init__(self, optimizer, warmup_steps, total_steps, lr_max, lr_min=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            lr = self.lr_max * (self.current_step / self.warmup_steps)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1 + math.cos(math.pi * progress)
            )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def build_model(device):
    """
    Build the SmallCNN model.

    Args:
        device: torch device

    Returns:
        model on the specified device
    """
    model = SmallCNN().to(device)
    return model


def train(model, train_loader, val_loader, device, epochs=10,
          scheduler_type='fixed'):
    """
    Train the model with the specified LR scheduling strategy.

    Args:
        model: SmallCNN instance
        train_loader: training DataLoader
        val_loader: validation DataLoader
        device: torch device
        epochs: number of training epochs
        scheduler_type: 'fixed', 'cosine', or 'warmup_cosine'

    Returns:
        dict with loss_history, val_acc_history, lr_history
    """
    lr_max = 1e-2
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_max, momentum=0.9)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.05 * total_steps)

    if scheduler_type == 'fixed':
        scheduler = FixedScheduler(optimizer)
    elif scheduler_type == 'cosine':
        scheduler = CosineScheduler(optimizer, total_steps, lr_max)
    elif scheduler_type == 'warmup_cosine':
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_steps, total_steps, lr_max
        )
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    loss_history = []
    val_acc_history = []
    lr_history = []

    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Acc':>7}  {'LR':>8}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*7}  {'─'*8}")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            num_batches += 1
            lr_history.append(scheduler.get_lr())

        epoch_loss = running_loss / num_batches
        loss_history.append(epoch_loss)

        # Validate
        val_metrics = evaluate(model, val_loader, device)
        val_acc_history.append(val_metrics['accuracy'])

        current_lr = scheduler.get_lr()
        print(
            f"  {epoch:>2}/{epochs:<2}  "
            f"{epoch_loss:>10.4f}  "
            f"{val_metrics['accuracy']:>7.4f}  "
            f"{current_lr:>8.1e}"
        )

    return {
        'loss_history': loss_history,
        'val_acc_history': val_acc_history,
        'lr_history': lr_history,
    }


def evaluate(model, data_loader, device):
    """
    Evaluate the model on a dataset.

    Args:
        model: SmallCNN instance
        data_loader: DataLoader to evaluate on
        device: torch device

    Returns:
        dict with 'loss' and 'accuracy'
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
    }


def predict(model, data_loader, device):
    """
    Generate predictions for all samples in data_loader.

    Args:
        model: SmallCNN instance
        data_loader: DataLoader
        device: torch device

    Returns:
        predictions tensor (int64)
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu())

    return torch.cat(all_preds)


def save_artifacts(results, output_dir='output'):
    """
    Save metrics JSON with all three runs' results and plots.

    Args:
        results: dict mapping run name to metrics/histories
        output_dir: output directory path
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert lr_history floats for JSON serialisation
    serialisable = {}
    for name, data in results.items():
        serialisable[name] = {
            'val_accuracy': data['val_accuracy'],
            'val_loss': data['val_loss'],
            'lr_history': [float(v) for v in data['lr_history']],
            'val_acc_history': [float(v) for v in data['val_acc_history']],
        }

    path = os.path.join(output_dir, 'metrics.json')
    with open(path, 'w') as f:
        json.dump(serialisable, f, indent=2)

    print(f"Artifacts saved to {output_dir}")

    # --- Plots ---
    run_labels = {'fixed_lr': 'Fixed LR', 'cosine': 'Cosine Annealing', 'warmup_cosine': 'Warmup + Cosine'}
    run_colors = {'fixed_lr': '#F44336', 'cosine': '#2196F3', 'warmup_cosine': '#4CAF50'}

    # 1. LR schedules
    plt.figure(figsize=(10, 5))
    for name, data in results.items():
        label = run_labels.get(name, name)
        color = run_colors.get(name, None)
        plt.plot(data['lr_history'], label=label, color=color, alpha=0.8)
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedules')
    plt.legend()
    plt.grid(True, alpha=0.3)
    p = os.path.join(output_dir, 'lr_schedules.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {p}")

    # 2. Val accuracy curves
    plt.figure(figsize=(8, 5))
    for name, data in results.items():
        label = run_labels.get(name, name)
        color = run_colors.get(name, None)
        epochs = range(1, len(data['val_acc_history']) + 1)
        plt.plot(epochs, data['val_acc_history'], label=label, color=color, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    p = os.path.join(output_dir, 'val_accuracy_curves.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {p}")

    # 3. Comparison bars
    plt.figure(figsize=(8, 5))
    names_list = list(results.keys())
    labels_list = [run_labels.get(n, n) for n in names_list]
    accs_list = [results[n]['val_accuracy'] for n in names_list]
    colors_list = [run_colors.get(n, '#999999') for n in names_list]
    bars = plt.bar(labels_list, accs_list, color=colors_list)
    for bar, acc in zip(bars, accs_list):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{acc:.4f}', ha='center', va='bottom', fontsize=11)
    plt.ylabel('Final Validation Accuracy')
    plt.title('Final Accuracy Comparison')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    p = os.path.join(output_dir, 'comparison_bars.png')
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {p}")


def main():
    """Main function to run the LR scheduling comparison task."""
    print("=" * 60)
    print("Cosine Annealing + Warmup LR Schedule")
    print("FashionMNIST - SmallCNN - SGD")
    print("=" * 60)

    device = get_device()
    print(f"Using device: {device}")

    print("\nDownloading / loading FashionMNIST...")
    train_loader, val_loader = make_dataloaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    results = {}

    for name, sched_type in [
        ('fixed_lr', 'fixed'),
        ('cosine', 'cosine'),
        ('warmup_cosine', 'warmup_cosine'),
    ]:
        print(f"\n{'=' * 60}")
        print(f"Run: {name} (scheduler: {sched_type})")
        print(f"{'=' * 60}")

        set_seed(42)  # Same init for fair comparison
        model = build_model(device)
        history = train(
            model, train_loader, val_loader, device,
            epochs=10, scheduler_type=sched_type
        )
        metrics = evaluate(model, val_loader, device)

        results[name] = {
            'val_accuracy': metrics['accuracy'],
            'val_loss': metrics['loss'],
            'lr_history': history['lr_history'],
            'val_acc_history': history['val_acc_history'],
        }
        print(f"  Final val accuracy: {metrics['accuracy']:.4f}")

    save_artifacts(results, output_dir='output')

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(f"{'Run':<16} {'Val Acc':>10} {'Val Loss':>10} {'Final LR':>12}")
    print("-" * 50)
    for name, data in results.items():
        final_lr = data['lr_history'][-1]
        print(
            f"{name:<16} {data['val_accuracy']:>10.4f} "
            f"{data['val_loss']:>10.4f} {final_lr:>12.6f}"
        )

    # ------------------------------------------------------------------
    # Quality checks
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    checks_passed = True

    # 1. warmup_cosine accuracy >= fixed_lr accuracy
    check1 = results['warmup_cosine']['val_accuracy'] >= results['fixed_lr']['val_accuracy']
    tag1 = "PASS" if check1 else "FAIL"
    print(
        f"  [{tag1}] warmup_cosine acc ({results['warmup_cosine']['val_accuracy']:.4f}) "
        f">= fixed_lr acc ({results['fixed_lr']['val_accuracy']:.4f})"
    )
    checks_passed = checks_passed and check1

    # 2. All three runs achieve val accuracy > 0.80
    for name in ['fixed_lr', 'cosine', 'warmup_cosine']:
        ok = results[name]['val_accuracy'] > 0.80
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name} val_acc > 0.80: {results[name]['val_accuracy']:.4f}")
        checks_passed = checks_passed and ok

    # 3. Final LR for cosine and warmup_cosine is near zero
    for name in ['cosine', 'warmup_cosine']:
        final_lr = results[name]['lr_history'][-1]
        ok = final_lr < 1e-3
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name} final LR < 1e-3: {final_lr:.8f}")
        checks_passed = checks_passed and ok

    # 4. Fixed LR stayed constant
    fixed_lrs = results['fixed_lr']['lr_history']
    ok = abs(fixed_lrs[-1] - fixed_lrs[0]) < 1e-8
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] fixed_lr stayed constant: {fixed_lrs[0]:.6f} -> {fixed_lrs[-1]:.6f}")
    checks_passed = checks_passed and ok

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if checks_passed:
        print("PASS: All quality checks passed!")
    else:
        print("FAIL: Some quality checks failed!")
    print("=" * 60)

    return 0 if checks_passed else 1


if __name__ == '__main__':
    sys.exit(main())
