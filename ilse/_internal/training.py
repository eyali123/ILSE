"""
Training loop for ILSE classifiers.
"""
import time
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(
    encoder: nn.Module,
    head: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 50,
    patience: int = 10,
    device: torch.device = None,
    is_gnn: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Train encoder + classification head.

    Args:
        encoder: The ILSE encoder (GNNEncoder or SetEncoder).
        head: ClassificationHead.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader (optional; if None, uses train loss for stopping).
        lr: Learning rate.
        weight_decay: L2 regularization.
        epochs: Maximum training epochs.
        patience: Early stopping patience.
        device: torch device.
        is_gnn: True for GNN encoders (PyG batches), False for SetEncoder (tensor batches).
        verbose: Print progress.

    Returns:
        Dict with best_val_acc, best_epoch, train_time_sec, best_state_dict.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = encoder.to(device)
    head = head.to(device)

    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=3, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    epochs_no_improve = 0

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # --- Train ---
        encoder.train()
        head.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            optimizer.zero_grad()

            if is_gnn:
                batch = batch.to(device)
                h = encoder(batch)
                labels = batch.y
            else:
                x, labels = batch
                x, labels = x.to(device), labels.to(device)
                h = encoder(x)

            logits = head(h)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # --- Validate ---
        if val_loader is not None:
            val_acc = _evaluate(encoder, head, val_loader, device, is_gnn)
        else:
            val_acc = train_acc

        scheduler.step(val_acc)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(
                f"Epoch {epoch:3d} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.1e}"
            )

        # Early stopping
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {
                "encoder": {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                "head": {k: v.cpu().clone() for k, v in head.state_dict().items()},
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (patience={patience})")
                break

    train_time = time.time() - start_time

    # Restore best weights
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        head.load_state_dict(best_state["head"])

    if verbose:
        total_params = sum(p.numel() for p in params if p.requires_grad)
        print(
            f"Done: best_val_acc={best_val_acc:.4f} at epoch {best_epoch} | "
            f"{total_params:,} params | {train_time:.1f}s"
        )

    return {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "train_time_sec": train_time,
        "best_state": best_state,
    }


@torch.no_grad()
def _evaluate(encoder, head, loader, device, is_gnn):
    encoder.eval()
    head.eval()
    correct = 0
    total = 0
    for batch in loader:
        if is_gnn:
            batch = batch.to(device)
            h = encoder(batch)
            labels = batch.y
        else:
            x, labels = batch
            x, labels = x.to(device), labels.to(device)
            h = encoder(x)
        logits = head(h)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return correct / total if total > 0 else 0.0
