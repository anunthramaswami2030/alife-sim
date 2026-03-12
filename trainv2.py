import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from new_model import MorphologyDataset, CNNEncoder


class AcceptClassifier(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 64):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=in_channels, latent_dim=latent_dim)

        self.head = nn.Sequential(
            nn.Linear(3 * latent_dim + 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        parent_mask: torch.Tensor,
        child_mask: torch.Tensor,
        parent_fitness: torch.Tensor,
        edit_distance: torch.Tensor,
    ) -> torch.Tensor:
        z_parent = self.encoder(parent_mask)
        z_child = self.encoder(child_mask)
        z_diff = z_child - z_parent

        scalar_feats = torch.stack([parent_fitness, edit_distance], dim=1)
        feat = torch.cat([z_parent, z_child, z_diff, scalar_feats], dim=1)

        logits = self.head(feat).squeeze(1)
        return logits


def grouped_train_val_split(parent_ids: torch.Tensor, train_split: float = 0.8, seed: int = 0):
    unique_parent_ids = torch.unique(parent_ids)

    if unique_parent_ids.numel() < 2:
        raise ValueError(
            f"Need at least 2 unique parent_id groups for train/val split, got {unique_parent_ids.numel()}."
        )

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(unique_parent_ids.numel(), generator=g)
    unique_parent_ids = unique_parent_ids[perm]

    n_train_groups = int(train_split * unique_parent_ids.numel())
    n_train_groups = max(1, min(n_train_groups, unique_parent_ids.numel() - 1))

    train_parent_ids = unique_parent_ids[:n_train_groups]
    val_parent_ids = unique_parent_ids[n_train_groups:]

    train_mask = torch.isin(parent_ids, train_parent_ids)
    val_mask = torch.isin(parent_ids, val_parent_ids)

    train_indices = torch.where(train_mask)[0]
    val_indices = torch.where(val_mask)[0]

    return train_indices, val_indices


def binary_metrics_from_logits(logits: torch.Tensor, target: torch.Tensor):
    probs = torch.sigmoid(logits)
    pred = (probs >= 0.5).float()

    acc = (pred == target).float().mean().item()

    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return acc, precision, recall, f1


def train_accept_classifier(
    dataset_path: str,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    latent_dim: int = 64,
    train_split: float = 0.8,
    seed: int = 0,
    checkpoint_path: str = "best_accept_classifier.pt",
):
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dataset = MorphologyDataset(dataset_path)

    n_total = len(dataset)

    train_indices, val_indices = grouped_train_val_split(
        parent_ids=dataset.parent_id,
        train_split=train_split,
        seed=seed,
    )

    n_train = train_indices.numel()
    n_val = val_indices.numel()

    train_ds = Subset(dataset, train_indices.tolist())
    val_ds = Subset(dataset, val_indices.tolist())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    train_targets = dataset.fitness_delta[train_indices]
    train_accept = (train_targets > 0).float()
    pos_count = train_accept.sum().item()
    neg_count = train_accept.numel() - pos_count

    pos_weight = torch.tensor(
        [neg_count / max(pos_count, 1.0)],
        dtype=torch.float32,
        device=device,
    )

    model = AcceptClassifier(in_channels=1, latent_dim=latent_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")

    n_train_groups = torch.unique(dataset.parent_id[train_indices]).numel()
    n_val_groups = torch.unique(dataset.parent_id[val_indices]).numel()

    print(f"dataset size: {n_total}")
    print(f"train size:   {n_train}")
    print(f"val size:     {n_val}")
    print(f"train parent groups: {n_train_groups}")
    print(f"val parent groups:   {n_val_groups}")
    print(f"positive train samples: {int(pos_count)}")
    print(f"negative train samples: {int(neg_count)}")
    print(f"pos_weight: {pos_weight.item():.4f}")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_prec_sum = 0.0
        train_rec_sum = 0.0
        train_f1_sum = 0.0
        train_count = 0

        for batch in train_loader:
            parent = batch["parent_mask"].to(device)
            child = batch["child_mask"].to(device)
            parent_fitness = batch["parent_fitness"].to(device)
            edit_distance = batch["edit_distance"].to(device)
            accepted = (batch["fitness_delta"].to(device) > 0).float()

            logits = model(parent, child, parent_fitness, edit_distance)
            loss = criterion(logits, accepted)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = parent.shape[0]
            acc, prec, rec, f1 = binary_metrics_from_logits(logits.detach(), accepted)

            train_loss_sum += loss.item() * bs
            train_acc_sum += acc * bs
            train_prec_sum += prec * bs
            train_rec_sum += rec * bs
            train_f1_sum += f1 * bs
            train_count += bs

        train_loss = train_loss_sum / max(train_count, 1)
        train_acc = train_acc_sum / max(train_count, 1)
        train_prec = train_prec_sum / max(train_count, 1)
        train_rec = train_rec_sum / max(train_count, 1)
        train_f1 = train_f1_sum / max(train_count, 1)

        model.eval()
        val_loss_sum = 0.0
        val_acc_sum = 0.0
        val_prec_sum = 0.0
        val_rec_sum = 0.0
        val_f1_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                parent = batch["parent_mask"].to(device)
                child = batch["child_mask"].to(device)
                parent_fitness = batch["parent_fitness"].to(device)
                edit_distance = batch["edit_distance"].to(device)
                accepted = (batch["fitness_delta"].to(device) > 0).float()

                logits = model(parent, child, parent_fitness, edit_distance)
                loss = criterion(logits, accepted)

                bs = parent.shape[0]
                acc, prec, rec, f1 = binary_metrics_from_logits(logits, accepted)

                val_loss_sum += loss.item() * bs
                val_acc_sum += acc * bs
                val_prec_sum += prec * bs
                val_rec_sum += rec * bs
                val_f1_sum += f1 * bs
                val_count += bs

        val_loss = val_loss_sum / max(val_count, 1)
        val_acc = val_acc_sum / max(val_count, 1)
        val_prec = val_prec_sum / max(val_count, 1)
        val_rec = val_rec_sum / max(val_count, 1)
        val_f1 = val_f1_sum / max(val_count, 1)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"train_prec={train_prec:.4f} "
            f"train_rec={train_rec:.4f} "
            f"train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_prec={val_prec:.4f} "
            f"val_rec={val_rec:.4f} "
            f"val_f1={val_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "latent_dim": latent_dim,
                    "in_channels": 1,
                    "seed": seed,
                    "pos_weight": pos_weight.detach().cpu(),
                },
                checkpoint_path,
            )

    print(f"Best model saved to: {os.path.abspath(checkpoint_path)}")
    return model


if __name__ == "__main__":
    train_accept_classifier(
        dataset_path="dataset_7hr.npz",
        epochs=30,
        batch_size=256,
        lr=1e-3,
        latent_dim=64,
        checkpoint_path="best_accept_classifier_dataset_7hr.pt",
    )