import torch
from torch.utils.data import DataLoader, Subset

from new_model import MorphologyDataset, CNNEncoder


class AcceptClassifier(torch.nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 64):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=in_channels, latent_dim=latent_dim)

        self.head = torch.nn.Sequential(
            torch.nn.Linear(3 * latent_dim + 2, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 1),
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


def metrics_at_threshold(probs: torch.Tensor, target: torch.Tensor, threshold: float):
    pred = (probs >= threshold).float()

    acc = (pred == target).float().mean().item()

    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    specificity = tn / max(tn + fp, 1)

    return {
        "threshold": threshold,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def collect_val_probs(
    dataset_path: str,
    checkpoint_path: str,
    batch_size: int = 256,
    train_split: float = 0.8,
    seed: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    latent_dim = checkpoint["latent_dim"]
    in_channels = checkpoint["in_channels"]

    dataset = MorphologyDataset(dataset_path)
    _, val_indices = grouped_train_val_split(
        parent_ids=dataset.parent_id,
        train_split=train_split,
        seed=seed,
    )
    val_ds = Subset(dataset, val_indices.tolist())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = AcceptClassifier(in_channels=in_channels, latent_dim=latent_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            parent = batch["parent_mask"].to(device)
            child = batch["child_mask"].to(device)
            parent_fitness = batch["parent_fitness"].to(device)
            edit_distance = batch["edit_distance"].to(device)
            accepted = (batch["fitness_delta"].to(device) > 0).float()

            logits = model(parent, child, parent_fitness, edit_distance)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_targets.append(accepted.cpu())

    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)

    print(f"val samples: {all_probs.numel()}")
    print(f"positive val samples: {int(all_targets.sum().item())}")
    print(f"negative val samples: {int((all_targets == 0).sum().item())}")

    return all_probs, all_targets


def main():
    dataset_path = "dataset_7hr.npz"
    checkpoint_path = "best_accept_classifier_dataset_7hr.pt"
    batch_size = 256
    train_split = 0.8
    seed = 0

    probs, targets = collect_val_probs(
        dataset_path=dataset_path,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        train_split=train_split,
        seed=seed,
    )

    thresholds = torch.linspace(0.05, 0.95, steps=19).tolist()
    results = [metrics_at_threshold(probs, targets, t) for t in thresholds]

    best_f1 = max(results, key=lambda x: x["f1"])
    best_prec = max(results, key=lambda x: x["precision"])
    best_rec = max(results, key=lambda x: x["recall"])

    print("\nTop thresholds by F1:")
    results_sorted = sorted(results, key=lambda x: x["f1"], reverse=True)
    for r in results_sorted[:10]:
        print(
            f"thr={r['threshold']:.2f} | "
            f"acc={r['acc']:.4f} "
            f"prec={r['precision']:.4f} "
            f"rec={r['recall']:.4f} "
            f"f1={r['f1']:.4f} "
            f"spec={r['specificity']:.4f}"
        )

    print("\nBest by F1:")
    print(best_f1)

    print("\nBest by precision:")
    print(best_prec)

    print("\nBest by recall:")
    print(best_rec)


if __name__ == "__main__":
    main()