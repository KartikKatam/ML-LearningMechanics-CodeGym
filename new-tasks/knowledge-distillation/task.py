"""
Knowledge Distillation - PyTorch Implementation

Knowledge distillation is a model compression technique where a smaller "student"
network learns to mimic a larger, pre-trained "teacher" network. Instead of training
the student solely on hard one-hot labels, the teacher produces soft probability
distributions by scaling its logits with a temperature parameter T. These softened
outputs capture richer inter-class relationships (e.g., "this shirt looks a bit like
a coat") that hard labels discard.

The student minimizes a weighted combination of two losses:
  1. Standard cross-entropy against the true hard labels.
  2. KL divergence between the temperature-scaled softmax outputs of the student
     and the teacher. This term is multiplied by T^2 to compensate for the reduced
     gradient magnitudes that result from temperature scaling.

Alpha controls the balance: alpha * CE + (1 - alpha) * KL_soft.
With alpha=0.3 the student draws 70% of its learning signal from the teacher's
soft targets and 30% from the ground-truth labels.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'kd_lvl3_distillation',
        'task_type': 'classification',
        'description': (
            'Knowledge distillation: train a small student MLP to mimic a larger '
            'teacher CNN on FashionMNIST using temperature-scaled soft targets '
            'and KL divergence.'
        ),
        'input_type': 'float32',
        'output_type': 'int64',
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


def make_dataloaders(batch_size=128):
    """
    Create FashionMNIST train and validation dataloaders.

    Returns:
        train_loader, val_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform,
    )
    val_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class TeacherCNN(nn.Module):
    """Larger CNN teacher model (~800k parameters)."""

    def __init__(self):
        super(TeacherCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class StudentMLP(nn.Module):
    """Smaller MLP student model (~100k parameters)."""

    def __init__(self):
        super(StudentMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def build_model(model_type, device):
    """
    Build a teacher or student model.

    Args:
        model_type: 'teacher' or 'student'
        device: torch device

    Returns:
        model on the specified device
    """
    if model_type == 'teacher':
        model = TeacherCNN()
    elif model_type == 'student':
        model = StudentMLP()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model.to(device)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def distillation_loss(student_logits, teacher_logits, true_labels,
                      temperature=4.0, alpha=0.3):
    """
    Combined distillation loss.

    Args:
        student_logits: raw logits from the student
        teacher_logits: raw logits from the teacher (detached)
        true_labels: ground-truth class indices
        temperature: softmax temperature for soft targets
        alpha: weight for the hard-label CE loss (1-alpha for KL)

    Returns:
        scalar loss
    """
    # Soft targets: KL divergence between temperature-scaled distributions
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)

    # Hard targets: standard cross entropy
    ce_loss = F.cross_entropy(student_logits, true_labels)

    # Combined loss
    return alpha * ce_loss + (1 - alpha) * kl_loss


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train(model, train_loader, val_loader, device, epochs=5,
          teacher_model=None, temperature=4.0, alpha=0.3):
    """
    Train a model with optional knowledge distillation.

    If teacher_model is provided, use distillation_loss.
    Otherwise, use standard cross-entropy.

    Args:
        model: the model to train
        train_loader: training DataLoader
        val_loader: validation DataLoader
        device: torch device
        epochs: number of training epochs
        teacher_model: optional pre-trained teacher (in eval mode)
        temperature: temperature for distillation
        alpha: weight for CE vs KL in distillation loss

    Returns:
        dict with 'loss_history' and 'val_acc_history'
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_history = []
    val_acc_history = []

    print(f"  {'Epoch':>5}  {'Train Loss':>10}  {'Val Acc':>7}")
    print(f"  {'─'*5}  {'─'*10}  {'─'*7}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            student_logits = model(images)

            if teacher_model is not None:
                with torch.no_grad():
                    teacher_logits = teacher_model(images)
                loss = distillation_loss(
                    student_logits, teacher_logits, labels,
                    temperature=temperature, alpha=alpha,
                )
            else:
                loss = F.cross_entropy(student_logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches
        loss_history.append(avg_loss)

        # Validation accuracy
        val_metrics = evaluate(model, val_loader, device)
        val_acc = val_metrics['accuracy']
        val_acc_history.append(val_acc)

        print(
            f"  {epoch + 1:>2}/{epochs:<2}  "
            f"{avg_loss:>10.4f}  "
            f"{val_acc:>7.4f}"
        )

    return {'loss_history': loss_history, 'val_acc_history': val_acc_history}


def evaluate(model, data_loader, device):
    """
    Evaluate model on a data loader.

    Returns:
        dict with 'loss' and 'accuracy'
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
    }


def predict(model, data_loader, device):
    """
    Return predicted class indices for all samples in data_loader.

    Returns:
        predictions: 1-D torch.Tensor of int64
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            logits = model(images)
            all_preds.append(logits.argmax(dim=1).cpu())

    return torch.cat(all_preds)


def save_artifacts(model, metrics, output_dir='output',
                   teacher_history=None, baseline_history=None, distill_history=None):
    """
    Save model state dict, metrics JSON, and plots.

    Args:
        model: trained model
        metrics: dict of metrics
        output_dir: directory to write to
        teacher_history: training history for teacher
        baseline_history: training history for baseline student
        distill_history: training history for distilled student
    """
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model_state.pt'))

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Artifacts saved to {output_dir}")

    # --- Plots ---
    teacher_acc = metrics.get('teacher', {}).get('accuracy', 0)
    baseline_acc = metrics.get('baseline_student', {}).get('accuracy', 0)
    distilled_acc = metrics.get('distilled_student', {}).get('accuracy', 0)

    # 1. Accuracy comparison bar chart
    plt.figure(figsize=(8, 5))
    names = ['Teacher', 'Baseline Student', 'Distilled Student']
    accs = [teacher_acc, baseline_acc, distilled_acc]
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    bars = plt.bar(names, accs, color=colors)
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{acc:.4f}', ha='center', va='bottom', fontsize=11)
    plt.ylabel('Validation Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    path = os.path.join(output_dir, 'accuracy_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {path}")

    # 2. Loss curves
    if teacher_history and baseline_history and distill_history:
        plt.figure(figsize=(8, 5))
        epochs_t = range(1, len(teacher_history['loss_history']) + 1)
        epochs_s = range(1, len(baseline_history['loss_history']) + 1)
        epochs_d = range(1, len(distill_history['loss_history']) + 1)
        plt.plot(epochs_t, teacher_history['loss_history'], label='Teacher', marker='o')
        plt.plot(epochs_s, baseline_history['loss_history'], label='Baseline Student', marker='s')
        plt.plot(epochs_d, distill_history['loss_history'], label='Distilled Student', marker='^')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        path = os.path.join(output_dir, 'loss_curves.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {path}")

    # 3. Improvement summary
    improvement = distilled_acc - baseline_acc
    plt.figure(figsize=(7, 4))
    bar_names = ['Baseline Student', 'Distilled Student']
    bar_accs = [baseline_acc, distilled_acc]
    bar_colors = ['#FF9800', '#4CAF50']
    bars = plt.bar(bar_names, bar_accs, color=bar_colors)
    for bar, acc in zip(bars, bar_accs):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{acc:.4f}', ha='center', va='bottom', fontsize=11)
    plt.title(f'Distillation Improvement: {improvement:+.4f}')
    plt.ylabel('Validation Accuracy')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    path = os.path.join(output_dir, 'improvement_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Main function to run the knowledge distillation task."""
    print("=" * 60)
    print("Knowledge Distillation - PyTorch Implementation")
    print("=" * 60)

    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = make_dataloaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # ------------------------------------------------------------------
    # Phase 1: Train teacher
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 1: Training Teacher CNN")
    print("=" * 60)
    teacher = build_model('teacher', device)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"Teacher parameters: {teacher_params:,}")
    teacher_history = train(teacher, train_loader, val_loader, device, epochs=5)
    teacher_metrics = evaluate(teacher, val_loader, device)
    print(f"Teacher val accuracy: {teacher_metrics['accuracy']:.4f}")
    teacher.eval()  # freeze for distillation

    # ------------------------------------------------------------------
    # Phase 2: Train student WITHOUT distillation (baseline)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 2: Training Student MLP (baseline, no distillation)")
    print("=" * 60)
    set_seed(42)  # reset seed so both students start identically
    student_baseline = build_model('student', device)
    student_params = sum(p.numel() for p in student_baseline.parameters())
    print(f"Student parameters: {student_params:,}")
    baseline_history = train(student_baseline, train_loader, val_loader, device, epochs=5)
    baseline_metrics = evaluate(student_baseline, val_loader, device)
    print(f"Baseline student val accuracy: {baseline_metrics['accuracy']:.4f}")

    # ------------------------------------------------------------------
    # Phase 3: Train student WITH distillation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 3: Training Student MLP (with distillation)")
    print("=" * 60)
    set_seed(42)  # reset seed again for fair comparison
    student_distilled = build_model('student', device)
    distill_history = train(
        student_distilled, train_loader, val_loader, device, epochs=5,
        teacher_model=teacher, temperature=4.0, alpha=0.3,
    )
    distill_metrics = evaluate(student_distilled, val_loader, device)
    print(f"Distilled student val accuracy: {distill_metrics['accuracy']:.4f}")

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    print()
    save_artifacts(student_distilled, {
        'teacher': teacher_metrics,
        'baseline_student': baseline_metrics,
        'distilled_student': distill_metrics,
    }, output_dir='output',
        teacher_history=teacher_history,
        baseline_history=baseline_history,
        distill_history=distill_history)

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Model':<30} {'Val Accuracy':>12}  {'Val Loss':>10}")
    print("-" * 56)
    print(f"{'Teacher CNN':<30} {teacher_metrics['accuracy']:>12.4f}  {teacher_metrics['loss']:>10.4f}")
    print(f"{'Student MLP (baseline)':<30} {baseline_metrics['accuracy']:>12.4f}  {baseline_metrics['loss']:>10.4f}")
    print(f"{'Student MLP (distilled)':<30} {distill_metrics['accuracy']:>12.4f}  {distill_metrics['loss']:>10.4f}")
    improvement = distill_metrics['accuracy'] - baseline_metrics['accuracy']
    print(f"\nDistillation improvement: {improvement:+.4f}")

    # ------------------------------------------------------------------
    # Quality checks
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    checks_passed = True

    # Check 1: Teacher accuracy > 0.88
    c1 = teacher_metrics['accuracy'] > 0.88
    print(f"{'✓' if c1 else '✗'} Teacher accuracy > 0.88: {teacher_metrics['accuracy']:.4f}")
    checks_passed = checks_passed and c1

    # Check 2: Distilled student > baseline student
    c2 = distill_metrics['accuracy'] > baseline_metrics['accuracy']
    print(f"{'✓' if c2 else '✗'} Distilled > baseline: "
          f"{distill_metrics['accuracy']:.4f} > {baseline_metrics['accuracy']:.4f}")
    checks_passed = checks_passed and c2

    # Check 3: Distilled student accuracy > 0.82
    c3 = distill_metrics['accuracy'] > 0.82
    print(f"{'✓' if c3 else '✗'} Distilled accuracy > 0.82: {distill_metrics['accuracy']:.4f}")
    checks_passed = checks_passed and c3

    # Final summary
    print("\n" + "=" * 60)
    if checks_passed:
        print("PASS: All quality checks passed!")
    else:
        print("FAIL: Some quality checks failed!")
    print("=" * 60)

    return 0 if checks_passed else 1


if __name__ == '__main__':
    sys.exit(main())
